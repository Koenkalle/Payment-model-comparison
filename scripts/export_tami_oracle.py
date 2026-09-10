"""Generate an independent numerical reference using a pinned upstream checkout.

Usage: python scripts/export_tami_oracle.py --upstream /path/to/TAMI_temporal_graph
Only the authors' classes execute here; no local model or sampling code is used.
"""
import argparse
import ast
import copy
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F, MultiheadAttention

COMMIT = 'b64bfa53731c623d6fa46e44ebb5aa8b09d90d1e'


def classes(path, names, namespace):
    tree = ast.parse(path.read_text(), filename=str(path))
    selected = ast.Module(body=[node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names], type_ignores=[])
    exec(compile(selected, str(path), 'exec'), namespace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path('tests/temporal/upstream-oracle.npz'))
    args = parser.parse_args()
    actual = subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != COMMIT:
        parser.error('Reference generation requires pinned TAMI commit ' + COMMIT)
    subprocess.run(['git', '-C', str(args.upstream), 'diff', '--exit-code', 'HEAD', '--',
                    'utils/utils.py', 'models/modules.py', 'models/DyGFormer.py'], check=True)
    namespace = dict(np=np, torch=torch, nn=nn, F=F, MultiheadAttention=MultiheadAttention, copy=copy)
    classes(args.upstream / 'utils/utils.py', {'NeighborSampler'}, namespace)
    classes(args.upstream / 'models/modules.py', {'TimeEncoder', 'HistEmbAggregatorWeightedSum', 'TRCMemory', 'HistoricalDecoder'}, namespace)
    classes(args.upstream / 'models/DyGFormer.py', {'DyGFormer', 'NeighborCooccurrenceEncoder', 'TransformerEncoder'}, namespace)
    torch.set_num_threads(1)
    torch.manual_seed(771)
    rng = np.random.default_rng(129)
    sources = np.array([1, 2, 1, 3, 1, 1, 2, 1])
    destinations = np.array([2, 3, 2, 1, 3, 2, 1, 2])
    times = np.array([0., 1., 2., 2., 3., 4., 5., 6.])
    nodes = rng.normal(size=(5, 4)).astype(np.float32)
    edges = rng.normal(size=(len(times) + 1, 2)).astype(np.float32)
    nodes[0] = 0
    edges[0] = 0
    adjacency = [[] for _ in nodes]
    for i, (u, v, t) in enumerate(zip(sources, destinations, times)):
        adjacency[u].append((v, i + 1, t))
        adjacency[v].append((u, i + 1, t))
    sampler = namespace['NeighborSampler'](adjacency, sample_neighbor_strategy='recent')
    parameters = dict(time_feat_dim=4, channel_embedding_dim=4, patch_size=2, num_layers=1,
                      num_heads=2, dropout=0.0, max_input_sequence_length=4, device='cpu')
    network = nn.Sequential(namespace['DyGFormer'](nodes, edges, sampler, **parameters),
                            namespace['HistoricalDecoder'](4, 4, 4, 1, gamma=0.7))
    network.eval()
    output = {key: value.detach().numpy().copy() for key, value in network.state_dict().items()}
    pos_logits, neg_logits, source_embeddings, destination_embeddings = [], [], [], []
    losses = []
    for time in np.unique(times):
        indices = np.flatnonzero(times == time)
        u, v, t = sources[indices], destinations[indices], times[indices]
        a, b = network[0].compute_src_dst_node_temporal_embeddings(u, v, t)
        an, bn = network[0].compute_src_dst_node_temporal_embeddings(u, np.full(len(u), 4), t)
        neg = network[1](u, np.full(len(u), 4), an, bn)
        pos = network[1](u, v, a, b, update_memories=True)
        pos_logits.extend(pos.detach().numpy().flatten())
        neg_logits.extend(neg.detach().numpy().flatten())
        source_embeddings.extend(a.detach().numpy())
        destination_embeddings.extend(b.detach().numpy())
        losses.append(F.binary_cross_entropy_with_logits(torch.cat([pos, neg]), torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])))
    torch.stack(losses).sum().backward()
    output.update({'grad_' + name: parameter.grad.detach().numpy() for name, parameter in network.named_parameters()})
    output.update(sources=sources, destinations=destinations, times=times, nodes=nodes, edges=edges,
                  positive=np.asarray(pos_logits), negative=np.asarray(neg_logits),
                  source_embeddings=np.asarray(source_embeddings), destination_embeddings=np.asarray(destination_embeddings),
                  metadata=np.asarray(json.dumps({'commit': COMMIT, 'torch': torch.__version__, 'parameters': {**parameters, 'gamma': 0.7}})))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    print(args.output)


if __name__ == '__main__':
    main()
