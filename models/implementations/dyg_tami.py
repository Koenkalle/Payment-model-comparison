"""Trainable DyGFormer + TAMI: published backbone, LTE and directed-pair TRC.

This is a dynamic link predictor. Its outputs are not fraud probabilities.
See UPSTREAM.md for pinned sources, licenses and experiment-protocol adaptations.
"""
import json
import logging
import time
import numpy as np
import torch
from torch import nn
from .dygformer import DyGFormer
from .tami import HistoricalDecoder
from .temporal_history import NeighborSampler
from framework.temporal_sampling import timestamp_groups, DestinationSampler

DEFAULTS = dict(
    time_feat_dim=32,
    channel_embedding_dim=32,
    patch_size=2,
    num_layers=2,
    num_heads=2,
    dropout=0.1,
    max_input_sequence_length=64,
    gamma=0.9,
    epochs=10,
    batch_size=32,
    learning_rate=0.0001,
    weight_decay=0.0,
    patience=3,
    seed=42,
    device="cpu",
    num_threads=2,
)
LOGGER = logging.getLogger(__name__)


class Model:
    def __init__(self):
        self.library_version = torch.__version__
        self.network = None
        self.training_history = []

    def _initialize(self, dataset, parameters):
        unknown = set(parameters) - set(DEFAULTS)
        if unknown:
            raise ValueError(
                "Unknown DyGFormer + TAMI parameters: " + ", ".join(sorted(unknown))
            )
        self.parameters = {**DEFAULTS, **parameters}
        p = self.parameters
        for key in (
            "time_feat_dim",
            "channel_embedding_dim",
            "patch_size",
            "num_layers",
            "num_heads",
            "max_input_sequence_length",
            "epochs",
            "batch_size",
            "patience",
            "num_threads",
        ):
            if type(p[key]) is not int or p[key] < 1:
                raise ValueError(key + " must be a positive integer.")
        if (
            p["max_input_sequence_length"] < 2
            or p["max_input_sequence_length"] % p["patch_size"]
        ):
            raise ValueError(
                "max_input_sequence_length must be >= 2 and divisible by patch_size."
            )
        if (4 * p["channel_embedding_dim"]) % p["num_heads"]:
            raise ValueError("Four channel dimensions must be divisible by num_heads.")
        if not 0 <= p["dropout"] < 1 or not 0 <= p["gamma"] <= 1:
            raise ValueError("Invalid dropout or gamma.")
        if (
            not np.isfinite(p["learning_rate"])
            or p["learning_rate"] <= 0
            or not np.isfinite(p["weight_decay"])
            or p["weight_decay"] < 0
        ):
            raise ValueError(
                "learning_rate must be positive and weight_decay nonnegative."
            )
        torch.manual_seed(p["seed"])
        torch.set_num_threads(p["num_threads"])
        backbone = DyGFormer(
            dataset.node_features.copy(),
            dataset.edge_features.copy(),
            NeighborSampler(dataset, history_limit=p["max_input_sequence_length"] - 1),
            **{
                key: p[key]
                for key in (
                    "time_feat_dim",
                    "channel_embedding_dim",
                    "patch_size",
                    "num_layers",
                    "num_heads",
                    "dropout",
                    "max_input_sequence_length",
                    "device",
                )
            },
        )
        dim = dataset.node_features.shape[1]
        self.network = nn.Sequential(
            backbone,
            HistoricalDecoder(dim, dim, dim, device=p["device"], gamma=p["gamma"]),
        ).to(p["device"])
        self.feature_schema = {
            "node": list(dataset.node_feature_names),
            "edge": list(dataset.edge_feature_names),
        }
        self.node_ids = dataset.node_ids

    @property
    def memory(self):
        return self.network[1].historical_interaction_memory

    def _bind(self, dataset, indices=None):
        if self.feature_schema != {
            "node": list(dataset.node_feature_names),
            "edge": list(dataset.edge_feature_names),
        }:
            raise ValueError(
                "Graph feature names or order differ from the fitted model."
            )
        backbone = self.network[0]
        backbone.node_raw_features = torch.from_numpy(dataset.node_features.copy()).to(
            self.parameters["device"]
        )
        backbone.edge_raw_features = torch.from_numpy(dataset.edge_features.copy()).to(
            self.parameters["device"]
        )
        backbone.neighbor_sampler = NeighborSampler(
            dataset, indices, self.parameters["max_input_sequence_length"] - 1
        )
        self.node_ids = dataset.node_ids
        self.memory.reset_memory()

    def score_group(self, dataset, group, negative_destinations=None):
        """Pure scoring; caller commits the returned positive updates afterwards."""
        u, v, t = (
            dataset.sources[group],
            dataset.destinations[group],
            dataset.times[group],
        )
        source, destination = self.network[0].compute_src_dst_node_temporal_embeddings(
            u, v, t
        )
        positive, proposed = self.network[1].score(u, v, source, destination)
        negative = None
        if negative_destinations is not None:
            valid = negative_destinations >= 0
            if valid.any():
                source, destination = self.network[
                    0
                ].compute_src_dst_node_temporal_embeddings(
                    u[valid], negative_destinations[valid], t[valid]
                )
                negative = self.network[1](
                    u[valid], negative_destinations[valid], source, destination
                ).flatten()
        return positive.flatten(), negative, proposed

    def _commit(self, dataset, group, proposed):
        self.memory.update_memories(
            zip(dataset.sources[group], dataset.destinations[group]), proposed
        )

    def fit_graph(self, dataset, training, validation, parameters):
        self._initialize(dataset, parameters)
        p = self.parameters
        if (
            not len(training)
            or not len(validation)
            or dataset.times[training].max() >= dataset.times[validation].min()
        ):
            raise ValueError(
                "Graph fitting needs strictly separated train and validation intervals."
            )
        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=p["learning_rate"],
            weight_decay=p["weight_decay"],
        )
        loss_function = nn.BCEWithLogitsLoss(reduction="sum")
        best_loss, best_weights, stale = float("inf"), None, 0
        self.training_history = []
        started = time.perf_counter()
        LOGGER.info(
            "Link pretraining: %d training links, %d validation links, up to %d epochs on %s.",
            len(training),
            len(validation),
            p["epochs"],
            p["device"],
        )
        for epoch in range(p["epochs"]):
            epoch_started = time.perf_counter()
            self._bind(dataset, training)
            self.network.train()
            sampler = DestinationSampler(p["seed"] + epoch)
            optimizer.zero_grad()
            pending, total, count = 0, 0.0, 0

            def step():
                for parameter in self.network.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(pending)
                optimizer.step()
                optimizer.zero_grad()

            for group in timestamp_groups(dataset, training):
                negatives = sampler.sample(dataset, group)
                positive, negative, proposed = self.score_group(
                    dataset, group, negatives
                )
                if negative is not None:
                    logits = torch.cat([positive[negatives >= 0], negative])
                    labels = torch.cat(
                        [torch.ones_like(negative), torch.zeros_like(negative)]
                    )
                    loss = loss_function(logits, labels)
                    loss.backward()
                    total += loss.item()
                    count += len(logits)
                    pending += len(logits)
                self._commit(dataset, group, proposed)
                if pending >= 2 * p["batch_size"]:
                    step()
                    pending = 0
            if pending:
                step()
            if not count:
                raise ValueError(
                    "Training has no valid negative links; need alternative observed destinations."
                )
            # Rebuild memories using frozen epoch weights before validation.
            # Test interactions are never used for selection or gradient updates.
            evaluation = self.predict_graph(dataset, validation, seed=p["seed"] + 10000)
            valid = np.isfinite(evaluation["negative_logits"])
            if not valid.any():
                raise ValueError("Validation has no valid negative links.")
            positive, negative = (
                evaluation["positive_logits"][valid],
                evaluation["negative_logits"][valid],
            )
            val_loss = float(
                np.concatenate(
                    [np.logaddexp(0, -positive), np.logaddexp(0, negative)]
                ).mean()
            )
            self.training_history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": total / count,
                    "validation_loss": val_loss,
                }
            )
            improved = val_loss < best_loss
            if improved:
                best_loss = val_loss
                best_weights = {
                    key: value.detach().cpu().clone()
                    for key, value in self.network.state_dict().items()
                }
                self.best_epoch = epoch + 1
                stale = 0
            else:
                stale += 1
            LOGGER.info(
                "Link epoch %d/%d: train_loss=%.6f validation_loss=%.6f%s epoch=%.1fs total=%.1fs",
                epoch + 1,
                p["epochs"],
                total / count,
                val_loss,
                " new_best" if improved else f' patience={stale}/{p["patience"]}',
                time.perf_counter() - epoch_started,
                time.perf_counter() - started,
            )
            if stale >= p["patience"]:
                LOGGER.info(
                    "Link pretraining stopped early after epoch %d; best epoch was %d.",
                    epoch + 1,
                    self.best_epoch,
                )
                break
        self.network.load_state_dict(best_weights)
        self.network.eval()
        self.memory.reset_memory()
        LOGGER.info(
            "Link pretraining complete: best_epoch=%d best_validation_loss=%.6f elapsed=%.1fs",
            self.best_epoch,
            best_loss,
            time.perf_counter() - started,
        )

    def predict_graph(self, dataset, indices, seed=10042):
        """Replay the observed prefix with frozen weights; no outcomes are read.

        State resets each call so test, validation, and repeated evaluations cannot
        contaminate one another. Queries at equal times all precede memory updates.
        """
        indices = np.asarray(indices, dtype=np.int64)
        if not len(indices) or np.any(np.diff(indices) <= 0):
            raise ValueError(
                "Evaluation indices must be nonempty, unique and chronological."
            )
        self._bind(dataset)
        self.network.eval()
        lookup = {int(index): offset for offset, index in enumerate(indices)}
        positive = np.empty(len(indices))
        negative = np.full(len(indices), np.nan)
        destinations = np.full(len(indices), -1, dtype=np.int64)
        sampler = DestinationSampler(seed)
        prefix = np.flatnonzero(dataset.times <= dataset.times[indices[-1]])
        with torch.no_grad():
            for group in timestamp_groups(dataset, prefix):
                # Sample for all prefix groups, keeping a row's negatives invariant
                # to the caller's choice of output partition.
                sampled = sampler.sample(dataset, group)
                pos, neg, proposed = self.score_group(dataset, group, sampled)
                expanded = np.full(len(group), np.nan)
                if neg is not None:
                    expanded[sampled >= 0] = neg.cpu().numpy()
                for offset, index in enumerate(group):
                    if int(index) in lookup:
                        target = lookup[int(index)]
                        positive[target] = float(pos[offset])
                        negative[target] = expanded[offset]
                        destinations[target] = sampled[offset]
                self._commit(dataset, group, proposed)
        return {
            "positive_logits": positive,
            "negative_logits": negative,
            "negative_destinations": destinations,
        }

    def save(self, path):
        """Save tensor arrays and JSON metadata without pickle/object arrays."""
        weights = {
            key: value.detach().cpu().numpy()
            for key, value in self.network.state_dict().items()
        }
        memory = self.memory.most_recent_hist_emb
        keys = sorted(memory)
        weights["memory_keys"] = np.asarray(keys, dtype=np.int64).reshape(-1, 2)
        dim = self.network[1].fc1.out_features
        weights["memory_values"] = (
            np.stack([memory[key].cpu().numpy() for key in keys])
            if keys
            else np.empty((0, dim), dtype=np.float32)
        )
        weights["metadata"] = np.asarray(
            json.dumps(
                {
                    "version": 1,
                    "parameters": self.parameters,
                    "features": self.feature_schema,
                    "node_ids": self.node_ids,
                    "history": self.training_history,
                    "best_epoch": self.best_epoch,
                }
            )
        )
        with open(path, "wb") as stream:
            np.savez_compressed(stream, **weights)

    def load_graph(self, path, dataset):
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"]))
            if metadata.get("version") != 1:
                raise ValueError("Unsupported DyGFormer + TAMI checkpoint version.")
            schema = {
                "node": list(dataset.node_feature_names),
                "edge": list(dataset.edge_feature_names),
            }
            if schema != metadata["features"]:
                raise ValueError(
                    "Graph feature names or order differ from the fitted model."
                )
            parameters = {**metadata["parameters"], "device": "cpu"}
            self._initialize(dataset, parameters)
            weights = {
                key: torch.from_numpy(saved[key].copy())
                for key in self.network.state_dict()
            }
            if any(not torch.isfinite(value).all() for value in weights.values()):
                raise ValueError("Checkpoint contains nonfinite weights.")
            self.network.load_state_dict(weights, strict=True)
            # Keep a saved inspection snapshot when node identities are unchanged.
            # Prediction always resets and reconstructs history, even after load.
            if list(dataset.node_ids) == metadata["node_ids"]:
                self.memory.update_memories(
                    saved["memory_keys"],
                    torch.from_numpy(saved["memory_values"].copy()),
                )
            self.training_history = metadata["history"]
            self.best_epoch = metadata["best_epoch"]
            self.network.eval()


def create():
    return Model()
