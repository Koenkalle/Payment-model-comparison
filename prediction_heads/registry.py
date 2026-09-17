"""Allowlisted heads with separate calibration and learned-input contracts."""

import importlib
import json
from pathlib import Path


_IMPLEMENTATIONS = {
    "empirical_tail": "prediction_heads.implementations.empirical_tail",
    "fixed_likelihood": "prediction_heads.implementations.fixed_likelihood",
}
_LEARNED_IMPLEMENTATIONS = {
    "fraud_linear": "prediction_heads.implementations.fraud_linear",
}


def manifest(kind="link-logits"):
    """Return inspectable descriptions, checking they match the code allowlist."""
    payload = json.loads(Path(__file__).with_name("registry.json").read_text())
    if payload.get("version") != 1:
        raise ValueError("Unsupported prediction-head registry version.")
    entries = payload.get("prediction_heads", [])
    ids = [entry["id"] for entry in entries]
    implementations = {**_IMPLEMENTATIONS, **_LEARNED_IMPLEMENTATIONS}
    if len(set(ids)) != len(ids) or set(ids) != set(implementations):
        raise ValueError(
            "Prediction-head registry does not match the implementation allowlist."
        )
    for entry in entries:
        if entry.get("python_module") != implementations[entry["id"]]:
            raise ValueError("Prediction-head module is not allowlisted.")
    if kind not in (None, "link-logits", "transaction-representation"):
        raise ValueError("Unknown prediction-head input kind.")
    if kind is not None:
        payload["prediction_heads"] = [
            entry for entry in entries if entry["kind"] == kind
        ]
    return payload


def _implementation(identifier):
    if not isinstance(identifier, str) or identifier not in _IMPLEMENTATIONS:
        raise ValueError(f"Unknown prediction head: {identifier!r}")
    return importlib.import_module(_IMPLEMENTATIONS[identifier]).Head


def create_head(identifier="empirical_tail"):
    """Create an unfitted head by its registry ID, never an arbitrary module path."""
    return _implementation(identifier)()


def load_head(payload):
    """Restore fitted state from a JSON object, without importing supplied code."""
    if not isinstance(payload, dict):
        raise ValueError("Prediction-head state must be a JSON object.")
    return _implementation(payload.get("id")).from_dict(payload)


def create_learned_head(identifier, input_dim, parameters=None):
    """Construct a trainable head; the experiment trainer owns optimization.

    Tensor state belongs to the model's safe NPZ checkpoint. This deliberately
    does not overload the likelihood calibrators' fit(reference_logits) API.
    """
    if not isinstance(identifier, str) or identifier not in _LEARNED_IMPLEMENTATIONS:
        raise ValueError(f"Unknown learned prediction head: {identifier!r}")
    return importlib.import_module(_LEARNED_IMPLEMENTATIONS[identifier]).Head(
        input_dim, parameters
    )
