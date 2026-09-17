"""Train/evaluate registered temporal link models with explicit task semantics."""
import hashlib
import json
import os
import tempfile
from pathlib import Path
import numpy as np
from .registry import ROOT, create_model, load_dataset
from .contracts import TemporalGraphDataset
from .experiments import chronological_split, choose_threshold, digest


def fingerprint(dataset):
    hasher = hashlib.sha256()
    hasher.update(
        json.dumps(
            {
                "ids": dataset.ids,
                "nodes": dataset.node_ids,
                "node_features": dataset.node_feature_names,
                "edge_features": dataset.edge_feature_names,
                "provenance": dataset.provenance,
            },
            sort_keys=True,
        ).encode()
    )
    for array in (
        dataset.times,
        dataset.sources,
        dataset.destinations,
        dataset.node_features,
        dataset.edge_features,
    ):
        hasher.update(str((array.shape, array.dtype)).encode())
        hasher.update(array.tobytes())
    return hasher.hexdigest()


def source_hashes(descriptor):
    return {name: digest(ROOT / name) for name in descriptor["implementation_sources"]}


def probabilities(logits):
    return np.exp(-np.logaddexp(0, -np.asarray(logits)))


def paired_predictions(prediction):
    positive, negative = prediction["positive_logits"], prediction["negative_logits"]
    valid = np.isfinite(negative)
    if not np.isfinite(positive).all() or np.isinf(negative).any():
        raise ValueError("Temporal model returned nonfinite link logits.")
    labels = np.concatenate([np.ones(valid.sum()), np.zeros(valid.sum())])
    logits = np.concatenate([positive[valid], negative[valid]])
    return labels, logits, valid


def report(model, descriptor, dataset, indices, threshold, seed, partition):
    from sklearn.metrics import average_precision_score, roc_auc_score

    prediction = model.predict_graph(dataset, indices, seed)
    labels, logits, valid = paired_predictions(prediction)
    scores = probabilities(logits)
    metrics = {
        "observed_links": len(indices),
        "paired_links": int(valid.sum()),
        "unpaired_links": int((~valid).sum()),
        "negative_links": int(valid.sum()),
        "average_precision": float(average_precision_score(labels, scores))
        if len(labels)
        else None,
        "roc_auc": float(roc_auc_score(labels, scores)) if len(labels) else None,
        "accuracy": float(np.mean((scores > threshold) == labels))
        if len(labels)
        else None,
        "log_loss": float(np.mean(np.logaddexp(0, logits) - labels * logits))
        if len(labels)
        else None,
    }
    rows = []
    for offset, i in enumerate(indices):
        negative = int(prediction["negative_destinations"][offset])
        rows.append(
            {
                "id": dataset.ids[i],
                "source": dataset.node_ids[dataset.sources[i]],
                "destination": dataset.node_ids[dataset.destinations[i]],
                "elapsed_seconds": float(dataset.times[i]),
                "link_probability": float(
                    probabilities(prediction["positive_logits"][offset])
                ),
                "link_logit": float(prediction["positive_logits"][offset]),
                "negative_destination": dataset.node_ids[negative]
                if negative >= 0
                else None,
                "negative_probability": float(
                    probabilities(prediction["negative_logits"][offset])
                )
                if negative >= 0
                else None,
            }
        )
    return {
        "version": 1,
        "schema": "temporal-link-evaluation/v1",
        "task": "dynamic-link-prediction",
        "model_id": descriptor["id"],
        "partition": partition,
        "threshold": float(threshold),
        "negative_sampling": {
            "strategy": "causal-random-destination",
            "seed": seed,
            "per_positive": 1,
            "exclude": ["self-links", "all observed links at the query timestamp"],
            "no_candidate": "positive updates history but is excluded from paired metrics",
        },
        "history": "observed interactions strictly before each query; frozen weights; reset and replay per evaluation",
        "score_meaning": "Observed-versus-sampled-link classification; not fraud probability.",
        "dataset": dataset.provenance,
        "metrics": metrics,
        "rows": rows,
    }


def train(config, output, dataset):
    model, descriptor = create_model(config["model"], dataset.schema)
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Experiment output already exists; choose a new directory.")
    splits = chronological_split(dataset, config.get("split", {}))
    parameters = dict(config.get("parameters", {}))
    model.fit_graph(dataset, splits["train"], splits["validation"], parameters)
    evaluation_seed = int(config.get("evaluation_seed", 10042))
    threshold = config.get("decision_threshold")
    if threshold is None:
        prediction = model.predict_graph(dataset, splits["validation"], evaluation_seed)
        labels, logits, _ = paired_predictions(prediction)
        threshold = choose_threshold(labels, probabilities(logits))
    elif (
        not isinstance(threshold, (int, float))
        or not np.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("decision_threshold must be between zero and one.")
    results = report(
        model, descriptor, dataset, splits["test"], threshold, evaluation_seed, "test"
    )
    results["training_rows_in_evaluation"] = 0
    metadata = {
        "version": 1,
        "task": "dynamic-link-prediction",
        "input_schema": dataset.schema,
        "model_id": descriptor["id"],
        "implementation": descriptor,
        "implementation_sha256": source_hashes(descriptor),
        "library_version": model.library_version,
        "parameters": model.parameters,
        "features": model.feature_schema,
        "dataset": dataset.provenance,
        "dataset_fingerprint": fingerprint(dataset),
        "split": {
            name: [dataset.ids[i] for i in indices] for name, indices in splits.items()
        },
        "threshold": float(threshold),
        "threshold_source": "validation_f1"
        if config.get("decision_threshold") is None
        else "configured",
        "evaluation_seed": evaluation_seed,
        "best_epoch": model.best_epoch,
        "training_history": model.training_history,
        "model_file": "model.npz",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".payment-graph-", dir=output.parent
    ) as temporary:
        stage = Path(temporary)
        model.save(stage / "model.npz")
        metadata["model_sha256"] = digest(stage / "model.npz")
        (stage / "manifest.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n"
        )
        (stage / "test-report.json").write_text(
            json.dumps(results, indent=2, allow_nan=False) + "\n"
        )
        os.rename(stage, output)
    return metadata, results


def evaluate(artifact, dataset_config, base_dir=None, partition="all"):
    artifact = Path(artifact)
    metadata = json.loads((artifact / "manifest.json").read_text())
    if metadata.get("version") != 1 or metadata.get("model_file") != "model.npz":
        raise ValueError("Unsupported temporal model artifact.")
    if digest(artifact / "model.npz") != metadata["model_sha256"]:
        raise ValueError("Model artifact checksum does not match.")
    dataset = load_dataset(dataset_config, base_dir)
    if not isinstance(dataset, TemporalGraphDataset):
        raise ValueError("Temporal link models require temporal-graph/v1 input.")
    model, descriptor = create_model(metadata["model_id"], dataset.schema)
    if source_hashes(descriptor) != metadata["implementation_sha256"]:
        raise ValueError(
            "Model implementation changed since this artifact was created."
        )
    same_dataset = fingerprint(dataset) == metadata["dataset_fingerprint"]
    if partition == "all":
        indices = np.arange(len(dataset.ids))
    elif partition in metadata["split"]:
        if not same_dataset:
            raise ValueError(
                "Saved split membership requires the original graph dataset and configuration."
            )
        wanted = set(metadata["split"][partition])
        indices = np.asarray(
            [i for i, identifier in enumerate(dataset.ids) if identifier in wanted]
        )
        if len(indices) != len(wanted):
            raise ValueError("Saved partition IDs are missing.")
    else:
        raise ValueError("Unknown evaluation partition.")
    model.load_graph(artifact / "model.npz", dataset)
    result = report(
        model,
        descriptor,
        dataset,
        indices,
        metadata["threshold"],
        metadata["evaluation_seed"],
        partition,
    )
    trained = set(metadata["split"]["train"])
    result["training_rows_in_evaluation"] = (
        sum(dataset.ids[i] in trained for i in indices) if same_dataset else None
    )
    return result
