"""Temporal links with mapped CSV columns and optional DyGLib NumPy features."""
from pathlib import Path
import hashlib
import numpy as np
from framework.contracts import TemporalGraphDataset
from ._validation import source, provenance, rows, number, timestamp


def graph_dataset(
    ids,
    times,
    sources,
    destinations,
    features,
    feature_names,
    metadata,
    node_dimension=32,
    raw_nodes=None,
    raw_edges=None,
    edge_indices=None,
):
    if not ids or any(not str(i).strip() for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Interaction IDs must be nonempty and unique.")
    if any(not str(v).strip() for v in [*sources, *destinations]):
        raise ValueError("Every interaction needs a source and destination identity.")
    if not isinstance(node_dimension, int) or node_dimension < 1:
        raise ValueError("node_feature_dim must be positive.")
    identities = tuple(
        dict.fromkeys([str(v) for pair in zip(sources, destinations) for v in pair])
    )
    mapping = {name: i + 1 for i, name in enumerate(identities)}
    node_ids = ("", *identities)
    times = np.asarray(times, dtype=np.float64)
    if not np.isfinite(times).all():
        raise ValueError("Interaction times must be finite.")
    # Relative seconds retain precision when the upstream time encoder casts to float32.
    origin = float(times.min())
    times = times - origin
    metadata = {
        **metadata,
        "time_origin_seconds": origin,
        "task": "dynamic-link-prediction",
    }
    order = np.argsort(times, kind="stable")
    source_ids = np.asarray([mapping[str(v)] for v in sources], dtype=np.int64)[order]
    destination_ids = np.asarray(
        [mapping[str(v)] for v in destinations], dtype=np.int64
    )[order]
    if raw_nodes is None:
        node_features = np.zeros((len(node_ids), node_dimension), dtype=np.float32)
        node_names = tuple(f"constant_zero_{i}" for i in range(node_dimension))
    else:
        try:
            native_ids = [int(v) for v in identities]
        except ValueError:
            raise ValueError(
                "NumPy node features require integer source IDs indexing their rows."
            ) from None
        if min(native_ids) < 0 or max(native_ids) >= len(raw_nodes):
            raise ValueError("Node feature array does not cover the CSV node IDs.")
        node_features = np.vstack(
            [np.zeros((1, raw_nodes.shape[1])), raw_nodes[native_ids]]
        ).astype(np.float32)
        node_names = tuple(f"node_{i}" for i in range(raw_nodes.shape[1]))
    if raw_edges is not None:
        edge_indices = np.asarray(edge_indices, dtype=np.int64)
        if edge_indices.min() < 0 or edge_indices.max() >= len(raw_edges):
            raise ValueError("Edge feature array does not cover the CSV edge indices.")
        values = raw_edges[edge_indices][order]
        feature_names = tuple(f"edge_{i}" for i in range(raw_edges.shape[1]))
    else:
        values = np.asarray(features, dtype=np.float32)[order]
        if not feature_names:
            values = np.zeros((len(ids), 1), dtype=np.float32)
            feature_names = ("constant_zero",)
    edge_features = np.vstack([np.zeros((1, values.shape[1])), values]).astype(
        np.float32
    )
    sorted_times = times[order]
    for array in (
        sorted_times,
        source_ids,
        destination_ids,
        node_features,
        edge_features,
    ):
        if not np.isfinite(array).all():
            raise ValueError("Graph features and times must be finite.")
        array.setflags(write=False)
    return TemporalGraphDataset(
        tuple(ids[i] for i in order),
        sorted_times,
        source_ids,
        destination_ids,
        node_ids,
        node_features,
        edge_features,
        node_names,
        tuple(feature_names),
        metadata,
    )


def load(config, base):
    path = source(config, base)
    columns = config.get(
        "columns", {"id": "idx", "source": "u", "destination": "i", "time": "ts"}
    )
    if any(key not in columns for key in ("id", "source", "destination", "time")):
        raise ValueError("columns must map id, source, destination and time.")
    features = config.get("features", [])
    if len(set(features)) != len(features) or set(features) & set(columns.values()):
        raise ValueError(
            "Feature columns must be distinct from mapped identities, times and labels."
        )
    data = rows(path, [*columns.values(), *features])
    canonical = dict(config)
    for key in ("path", "node_features_path", "edge_features_path"):
        if key in canonical:
            filename = Path(canonical[key])
            canonical[key] = str(
                (filename if filename.is_absolute() else base / filename).resolve()
            )
    metadata = provenance(canonical, path)

    def matrix(key):
        if key not in config:
            return None
        filename = Path(config[key])
        filename = filename if filename.is_absolute() else base / filename
        values = np.load(filename, allow_pickle=False)
        if (
            values.ndim != 2
            or min(values.shape) < 1
            or not np.issubdtype(values.dtype, np.number)
            or not np.isfinite(values).all()
        ):
            raise ValueError(key + " must be a finite numeric matrix.")
        metadata[key + "_sha256"] = hashlib.sha256(filename.read_bytes()).hexdigest()
        return values

    raw_nodes, raw_edges = matrix("node_features_path"), matrix("edge_features_path")
    if raw_edges is not None and features:
        raise ValueError("Choose CSV features or edge_features_path, not both.")
    try:
        edge_indices = (
            [int(row[columns.get("edge_index", columns["id"])]) for row in data]
            if raw_edges is not None
            else None
        )
    except ValueError:
        raise ValueError("NumPy edge features require integer edge indices.") from None
    return graph_dataset(
        [row[columns["id"]].strip() for row in data],
        [
            timestamp(row[columns["time"]], config.get("time_unit", "seconds"))
            for row in data
        ],
        [row[columns["source"]].strip() for row in data],
        [row[columns["destination"]].strip() for row in data],
        [[number(row[f], f) for f in features] for row in data],
        features,
        metadata,
        config.get("node_feature_dim", 32),
        raw_nodes,
        raw_edges,
        edge_indices,
    )
