"""Export reference counts for the existing booster without fitting any weights.

The first floor(80%) historical episodes are the original fitting partition.
All events update causal history; only labeled payments become reference rows.
Counts are unweighted row counts, explicitly not native XGBoost Hessian cover.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE_PATHS = (
    "models/xgboost.json",
    "xgb-training.json",
    "training/train.py",
    "training/autodiff.py",
    "shared/runtime/xgboost.js",
    "scripts/export_xgboost_explanations.py",
    "models/implementations/_common.py",
    "models/implementations/xgboost_numpy.py",
    "models/implementations/xgboost_numpy_core.js",
    "datasets/payment_features.py",
    "datasets/payment_features.js",
)
OUTPUT = ROOT / "models/xgboost-explanations.json"


def canonical(value):
    """Cross-language semantic encoding, including exact IEEE-754 numbers."""
    if value is None or isinstance(value, (str, bool)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, (int, float)):
        return "~" + struct.pack(">d", float(value) if value else 0.0).hex()
    if isinstance(value, list):
        return "[" + ",".join(canonical(item) for item in value) + "]"
    if isinstance(value, dict):
        return (
            "{"
            + ",".join(
                canonical(key) + ":" + canonical(value[key]) for key in sorted(value)
            )
            + "}"
        )
    raise TypeError(type(value))


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def provenance():
    return {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in SOURCE_PATHS
    }


def model_fingerprint(model):
    return fingerprint(
        {
            key: model[key]
            for key in (
                "features",
                "amount_bins",
                "base_score",
                "learning_rate",
                "trees",
            )
        }
    )


def count_tree(tree, rows):
    """Preorder node counts; independently route each original fitting row."""
    nodes = []

    def collect(node):
        index = len(nodes)
        nodes.append({"tree": node, "count": 0})
        if "leaf" not in node:
            nodes[index]["left"] = collect(node["left"])
            nodes[index]["right"] = collect(node["right"])
        return index

    collect(tree)
    for row in rows:
        index = 0
        while True:
            current = nodes[index]
            current["count"] += 1
            node = current["tree"]
            if "leaf" in node:
                break
            index = current[
                "left" if row[node["feature"]] <= node["threshold"] else "right"
            ]
    return [node["count"] for node in nodes]


def create_sidecar(hashes=None):
    from training.train import AMOUNT, XGB_FEATURES, xgb_rows

    hashes = hashes or provenance()
    model = json.loads((ROOT / "models/xgboost.json").read_text())
    history = json.loads((ROOT / "xgb-training.json").read_text())
    if model["features"] != XGB_FEATURES or model["amount_bins"] != AMOUNT:
        raise ValueError(
            "Checkpoint feature schema differs from the training feature extractor."
        )
    if (
        model["training"]["fraud_seed"] != history["seed"]
        or model["training"]["fraud_flags_requested"] != history["flags_requested"]
    ):
        raise ValueError(
            "Historical data does not match the checkpoint training configuration."
        )
    rows, labels, episode_ids = xgb_rows(history)
    split = max(1, int(len(history["episodes"]) * 0.8))
    fitting = episode_ids < split
    reference_rows = rows[fitting]
    if (
        len(reference_rows) != model["training"]["training_rows"]
        or int(labels[fitting].sum()) != model["training"]["fraud_labels_used"]
    ):
        raise ValueError(
            "Reconstructed fitting population differs from checkpoint metadata."
        )
    sidecar = {
        "version": 1,
        "feature_schema_version": 1,
        "features": model["features"],
        "checkpoint_id": model["checkpoint_id"],
        "checkpoint_sha256": hashes["models/xgboost.json"],
        "model_fingerprint": model_fingerprint(model),
        "reference": {
            "convention": "unweighted_labeled_fitting_rows",
            "description": "Path-dependent branch proportions from unweighted labeled fitting rows; not Hessian cover.",
            "history_sha256": hashes["xgb-training.json"],
            "row_count": len(reference_rows),
            "fraud_row_count": int(labels[fitting].sum()),
            "episode_count": len(history["episodes"]),
            "split_episode": split,
            "split": "episode_index < max(1, floor(episode_count * 0.8))",
            "unknown_outcomes": "Excluded from reference rows; included when constructing preceding history.",
            "heldout_outcomes": "Excluded from the reference population.",
        },
        "provenance": hashes,
        "node_counts": [count_tree(tree, reference_rows) for tree in model["trees"]],
    }
    sidecar["reference_id"] = fingerprint(sidecar)
    return sidecar


def is_fresh(sidecar, hashes):
    unsigned = {key: value for key, value in sidecar.items() if key != "reference_id"}
    return (
        sidecar.get("version") == 1
        and sidecar.get("feature_schema_version") == 1
        and sidecar.get("provenance") == hashes
        and sidecar.get("reference_id") == fingerprint(unsigned)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the sidecar is absent or stale; do not write files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reconstruct the fitting rows even if the sidecar is fresh.",
    )
    args = parser.parse_args()
    hashes = provenance()
    try:
        existing = json.loads(OUTPUT.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        existing = {}
    if not args.force and is_fresh(existing, hashes):
        print(
            "XGBoost explanation reference is current ("
            + str(existing["reference"]["row_count"])
            + " fitting rows)."
        )
        return
    if args.check:
        raise SystemExit(
            "XGBoost explanation reference is stale; run python scripts/export_xgboost_explanations.py."
        )
    result = create_sidecar(hashes)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print(
        "Exported XGBoost explanation reference from "
        + str(result["reference"]["row_count"])
        + " fitting rows; model weights unchanged."
    )


if __name__ == "__main__":
    main()
