"""Export frozen native link logits and replaceable fraud heads for comparison.

The model remains a link predictor. Calibration compares observed-link scores,
without consulting outcomes, and decisions only flag unusual transactions. The
browser replays these recorded scores in shadow mode; it does not execute Torch.
"""
import dataclasses
import json
from pathlib import Path

import numpy as np

from .contracts import EventDataset, TemporalGraphDataset
from .experiments import digest
from .registry import create_model, load_dataset
from .temporal_experiments import fingerprint, source_hashes
from .temporal_sampling import timestamp_groups
from prediction_heads.registry import create_head, manifest as head_manifest


BROWSER_MAX_ACCOUNTS = 256
BROWSER_MAX_EVENTS = 20000


def _artifact(path):
    metadata = json.loads((path / "manifest.json").read_text())
    if (
        metadata.get("version") != 1
        or metadata.get("model_file") != "model.npz"
        or metadata.get("task") != "dynamic-link-prediction"
        or metadata.get("input_schema") != "temporal-graph/v1"
        or metadata.get("model_id") != "dyg_tami_native"
    ):
        raise ValueError(
            "export-fraud requires a native DyGFormer + TAMI link-model artifact."
        )
    if digest(path / "model.npz") != metadata.get("model_sha256"):
        raise ValueError("Model artifact checksum does not match.")
    return metadata


def _partitions(metadata, graph):
    """Validate saved partition membership before reserving calibration rows."""
    split = metadata.get("split", {})
    groups = [split.get(name) for name in ("train", "validation", "test")]
    if any(not isinstance(group, list) or not group for group in groups):
        raise ValueError(
            "Artifact requires nonempty saved train, validation and test splits."
        )
    flat = [identifier for group in groups for identifier in group]
    if flat != list(graph.ids):
        raise ValueError(
            "Saved splits must cover the original graph once in chronological order."
        )
    offsets = np.cumsum([0, *map(len, groups)])
    for boundary in offsets[1:-1]:
        if graph.times[boundary - 1] >= graph.times[boundary]:
            raise ValueError("Saved splits must not divide simultaneous transactions.")
    return {
        name: np.arange(offsets[i], offsets[i + 1])
        for i, name in enumerate(("train", "validation", "test"))
    }


def _calibration_prefix(graph, indices, max_rows):
    if type(max_rows) is not int or max_rows < 1:
        raise ValueError("calibration.max_rows must be a positive integer.")
    chosen = []
    for group in timestamp_groups(graph, indices):
        chosen.extend(group.tolist())
        # A complete timestamp group may exceed the requested row count.
        if len(chosen) >= max_rows:
            break
    if not chosen:
        raise ValueError("The saved test partition has no calibration transactions.")
    return np.asarray(chosen, dtype=np.int64)


def _same_graph(left, right):
    # Re-exporting a file or changing truth/provenance does not create new data.
    # The network sees relative time, so a uniform timestamp offset also cannot
    # make an otherwise identical graph into independent evaluation data.
    def canonical(graph):
        return dataclasses.replace(graph, provenance={})

    return fingerprint(canonical(left)) == fingerprint(canonical(right))


def _payment_signature(document, event):
    accounts = document["accounts"]
    return (
        str(accounts[event["u"]]["external_id"]),
        str(accounts[event["v"]]["external_id"]),
        float(event["t"]),
        float(event["amount"]),
    )


def _evaluation_ids(
    metadata,
    calibration_indices,
    original_graph,
    original_events,
    target_graph,
    target_events,
):
    same = _same_graph(original_graph, target_graph)
    reserved = (
        set(metadata["split"]["train"])
        | set(metadata["split"]["validation"])
        | {original_graph.ids[i] for i in calibration_indices}
    )
    if same:
        evaluation = [
            identifier
            for identifier in metadata["split"]["test"]
            if identifier not in reserved
        ]
        if not evaluation:
            raise ValueError(
                "Calibration consumes the entire test partition; reduce calibration.max_rows or use a separate target dataset."
            )
        return evaluation, True

    # Detect copying/renaming/reserializing already-used observations. Labels and
    # transaction IDs are deliberately excluded from this identity comparison.
    used = {
        _payment_signature(original_events, event)
        for event in original_events["events"]
        if event["kind"] == "payment" and event["id"] in reserved
    }
    overlaps = [
        event["id"]
        for event in target_events["events"]
        if event["kind"] == "payment"
        and _payment_signature(target_events, event) in used
    ]
    if overlaps:
        raise ValueError(
            "Target overlaps training, validation or calibration transactions; use the original full dataset for automatic held-out selection, or a separate dataset. Example: "
            + overlaps[0]
        )
    source_hash = original_graph.provenance.get("source_sha256")
    if source_hash is not None and source_hash == target_graph.provenance.get(
        "source_sha256"
    ):
        raise ValueError(
            "The original source was reinterpreted with a different graph configuration; this cannot be treated as independent evaluation data."
        )
    return list(target_graph.ids), False


def _positive_logits(model, graph, indices, seed):
    values = np.asarray(
        model.predict_graph(graph, indices, seed)["positive_logits"], dtype=float
    )
    if values.shape != (len(indices),) or not np.isfinite(values).all():
        raise ValueError("The native model must return one finite logit per payment.")
    return values


def _metrics(document, ids, logits, head, alpha):
    truth = document["truth"]
    flags = head.score(logits)["score"] > head.get_threshold(alpha)
    actual = np.asarray(
        [int(truth[identifier]) if identifier in truth else -1 for identifier in ids]
    )
    known = actual >= 0
    tp = int(np.sum((actual == 1) & flags))
    fp = int(np.sum((actual == 0) & flags))
    fn = int(np.sum((actual == 1) & ~flags))
    tn = int(np.sum((actual == 0) & ~flags))
    return {
        "rows": len(ids),
        "known": int(known.sum()),
        "unknown": int((~known).sum()),
        "flagged": int(flags.sum()),
        "flag_rate": float(flags.mean()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn)
        if 2 * tp + fp + fn
        else (0 if known.any() else None),
    }


def export_fraud(config, base_dir=None):
    """Return a JSON-safe comparison bundle; config paths are relative to base_dir.

    Config: {artifact, dataset: payment-event config,
             calibration: {dataset: original payment_graph config, max_rows: 100},
             head: 'empirical_tail', alpha: 0.02}.
    """
    base = Path(base_dir or ".").resolve()
    artifact = Path(config["artifact"])
    artifact = artifact if artifact.is_absolute() else base / artifact
    metadata = _artifact(artifact)
    head_ids = [entry["id"] for entry in head_manifest()["prediction_heads"]]
    selected = config.get("head", "empirical_tail")
    if selected not in head_ids:
        raise ValueError("Unknown fraud prediction head: " + str(selected))
    alpha = config.get("alpha", 0.02)
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not np.isfinite(alpha)
        or not 0 < alpha < 1
    ):
        raise ValueError("alpha must be a finite number strictly between zero and one.")
    calibration_config = config["calibration"]
    graph_config = calibration_config["dataset"]
    if graph_config.get("loader") != "payment_graph":
        raise ValueError(
            "Calibration requires the original payment_graph dataset and configuration."
        )
    original = load_dataset(graph_config, base)
    if not isinstance(original, TemporalGraphDataset) or fingerprint(
        original
    ) != metadata.get("dataset_fingerprint"):
        raise ValueError(
            "Calibration requires the original graph dataset and configuration used to fit the artifact."
        )
    parts = _partitions(metadata, original)
    calibration = _calibration_prefix(
        original, parts["test"], calibration_config.get("max_rows", 100)
    )
    target_config = config["dataset"]
    target = load_dataset(target_config, base)
    if not isinstance(target, EventDataset):
        if hasattr(target, "close"):
            target.close()
        raise ValueError("Comparison export requires payment-events/v1 input.")
    if (
        len(target.document["accounts"]) > BROWSER_MAX_ACCOUNTS
        or len(target.document["events"]) > BROWSER_MAX_EVENTS
    ):
        raise ValueError(
            "The browser comparison supports at most 256 accounts and 20,000 events; export a bounded dataset."
        )
    original_events = load_dataset(graph_config["dataset"], base)
    node_names = metadata.get("features", {}).get("node", [])
    if not node_names:
        raise ValueError("Artifact is missing its node feature schema.")
    target_graph = load_dataset(
        {
            "loader": "payment_graph",
            "dataset": target_config,
            "node_feature_dim": len(node_names),
        },
        base,
    )
    evaluation, same_dataset = _evaluation_ids(
        metadata,
        calibration,
        original,
        original_events.document,
        target_graph,
        target.document,
    )
    model, descriptor = create_model(metadata["model_id"], original.schema)
    if source_hashes(descriptor) != metadata.get("implementation_sha256"):
        raise ValueError(
            "Model implementation changed since this artifact was created."
        )
    model.load_graph(artifact / "model.npz", original)
    seed = metadata.get("evaluation_seed", 10042)
    reference = _positive_logits(model, original, calibration, seed)
    heads = {
        identifier: create_head(identifier).fit(reference) for identifier in head_ids
    }
    logits = _positive_logits(
        model, target_graph, np.arange(len(target_graph.ids)), seed
    )
    positions = {identifier: i for i, identifier in enumerate(target_graph.ids)}
    evaluation_logits = logits[[positions[identifier] for identifier in evaluation]]
    count = len(reference)
    resolution = 1 / (count + 1)
    result = {
        "version": 1,
        "schema": "native-fraud-comparison/v1",
        "dataset": target.document,
        "model": {
            "id": descriptor["id"],
            "label": "DyGFormer + TAMI · native",
            "checkpoint_id": metadata["model_sha256"],
            "implementation": descriptor,
        },
        "predictions": [
            {"id": identifier, "logit": float(logit)}
            for identifier, logit in zip(target_graph.ids, logits)
        ],
        "heads": {identifier: head.to_dict() for identifier, head in heads.items()},
        "head": selected,
        "alpha": float(alpha),
        "evaluation_ids": evaluation,
        "calibration": {
            "ids": [original.ids[i] for i in calibration],
            "count": count,
            "partition": "test-prefix",
            "dataset_fingerprint": metadata["dataset_fingerprint"],
            "requested_max_rows": calibration_config.get("max_rows", 100),
            "minimum_tail_probability": resolution,
            "alpha_can_flag": resolution < alpha,
            "note": "Frozen observed-link reference from complete timestamp groups after model selection; no outcomes used. Calibration rows are excluded from same-dataset evaluation.",
        },
        "provenance": {
            "artifact_model_sha256": metadata["model_sha256"],
            "artifact_dataset": metadata["dataset"],
            "target_dataset": target.provenance,
            "same_dataset": same_dataset,
            "implementation_sha256": metadata["implementation_sha256"],
            "model_parameters": metadata.get("parameters", {}),
            "evaluation_scope": "saved-test-after-calibration"
            if same_dataset
            else "separate-target-dataset",
            "training_rows_in_evaluation": 0 if same_dataset else None,
            "overlap_check": "Graph content equality ignores labels and file paths; separate targets reject matching external endpoints, relative event time and amount from train, validation or calibration. Independently transformed copies cannot be ruled out.",
        },
        "history": {
            "mode": "shadow",
            "payments": "observed-attempts",
            "timestamps": "strictly-before",
            "deposits": False,
            "reports": False,
        },
        "score_meaning": "Low observed-link likelihood flags unusual payments; neither link likelihood nor empirical tail rank is fraud probability. Outcomes are used only for evaluation metrics.",
        "metrics": {
            identifier: _metrics(
                target.document, evaluation, evaluation_logits, head, alpha
            )
            for identifier, head in heads.items()
        },
    }
    # Reject nonfinite metadata as well as scores before a caller writes a file.
    json.dumps(result, allow_nan=False)
    return result
