"""TAMI LTE and TRC, adapted from the authors' models/modules.py.

The equations and parameter names follow upstream. See UPSTREAM.md and licenses/.
Memory updates are explicit so every query at a timestamp sees the same past.
"""
import numpy as np
import torch
from torch import nn


class TimeEncoder(nn.Module):
    """LTE: cos(W log(1 + abs(delta_t)) + b), with learned frequencies."""

    def __init__(self, time_dim, parameter_requires_grad=True):
        super().__init__()
        self.w = nn.Linear(1, time_dim)
        self.w.weight = nn.Parameter(torch.from_numpy(
            1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float32)
        ).reshape(time_dim, 1), requires_grad=parameter_requires_grad)
        self.w.bias = nn.Parameter(torch.zeros(time_dim), requires_grad=parameter_requires_grad)

    def forward(self, timestamps):
        return torch.cos(self.w(torch.log1p(timestamps.abs()).unsqueeze(2)))


class TRCMemory(nn.Module):
    """Detached embeddings keyed by directed (source, destination) pairs."""

    def __init__(self, dim, device='cpu'):
        super().__init__()
        self.register_buffer('PAD_ZERO', torch.zeros(dim, device=device), persistent=False)
        self.most_recent_hist_emb = {}

    def get_memories(self, keys):
        return [self.most_recent_hist_emb.get((int(u), int(v)), self.PAD_ZERO).to(self.PAD_ZERO.device)
                for u, v in keys]

    def reset_memory(self):
        self.most_recent_hist_emb.clear()

    def update_memories(self, keys, embeddings):
        # Equal-time repeats have no chronological order: average their proposed
        # updates rather than letting the last file row decide the memory.
        grouped = {}
        for key, embedding in zip(keys, embeddings.detach()):
            grouped.setdefault(tuple(map(int, key)), []).append(embedding)
        for key, values in grouped.items():
            self.most_recent_hist_emb[key] = torch.stack(values).mean(0).detach().cpu()


class HistEmbAggregatorWeightedSum(nn.Module):
    def __init__(self, gamma=0.9):
        super().__init__()
        if not 0 <= gamma <= 1:
            raise ValueError('gamma must be between zero and one.')
        self.gamma = gamma

    def forward(self, current_emb, hist_emb):
        return self.gamma * current_emb + (1 - self.gamma) * hist_emb


class HistoricalDecoder(nn.Module):
    """Logit = fc2([ReLU(fc1([z_u,z_v])), previous_pair_memory])."""

    def __init__(self, input_dim1, input_dim2, hidden_dim, output_dim=1, device='cpu', gamma=0.9):
        super().__init__()
        self.fc1 = nn.Linear(input_dim1 + input_dim2, hidden_dim)
        self.fc2 = nn.Linear(2 * hidden_dim, output_dim)
        self.act = nn.ReLU()
        self.aggregate_hist_emb = HistEmbAggregatorWeightedSum(gamma)
        self.historical_interaction_memory = TRCMemory(hidden_dim, device)

    def score(self, src_ids, dst_ids, src_emb, dst_emb):
        current = self.act(self.fc1(torch.cat([src_emb, dst_emb], 1)))
        previous = torch.stack(self.historical_interaction_memory.get_memories(zip(src_ids, dst_ids)))
        logits = self.fc2(torch.cat([current, previous], 1))
        proposed = self.aggregate_hist_emb(current, previous)
        return logits, proposed

    def forward(self, src_ids, dst_ids, src_emb, dst_emb, update_memories=False):
        logits, proposed = self.score(src_ids, dst_ids, src_emb, dst_emb)
        if update_memories:
            self.historical_interaction_memory.update_memories(zip(src_ids, dst_ids), proposed)
        return logits
