"""A frozen empirical lower-tail head for link-prediction logits.

Higher logits mean more likely links. For reference logits r and query q,
    p(q) = (1 + number of r <= q) / (1 + number of r)
    score(q) = -log2(p(q))

The smoothing prevents zero tail probabilities. Using <= treats ties
conservatively: equal likelihoods are never arbitrarily ordered. A transaction
is suspected when score > -log2(alpha), equivalently p < alpha. The reference
distribution is fixed after fit; scoring does not adapt to evaluated events.

This is a rank-based anomaly measure, not the probability that fraud occurred.
"""

import numpy as np


def _logits(values, *, nonempty=False):
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Link logits must be a finite one-dimensional numeric array."
        ) from exc
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError("Link logits must be a finite one-dimensional numeric array.")
    if nonempty and not array.size:
        raise ValueError("Reference logits must not be empty.")
    return array


def _threshold(alpha):
    if isinstance(alpha, (bool, np.bool_)) or not np.isscalar(alpha):
        raise ValueError("Tail cutoff alpha must be a finite number in (0, 1].")
    try:
        alpha = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Tail cutoff alpha must be a finite number in (0, 1]."
        ) from exc
    if not np.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("Tail cutoff alpha must be a finite number in (0, 1].")
    return float(-np.log2(alpha))


def _state(payload, identifier, fields):
    if not isinstance(payload, dict):
        raise ValueError("Prediction-head state must be a JSON object.")
    if type(payload.get("version")) is not int or payload["version"] != 1:
        raise ValueError("Unsupported prediction-head state version.")
    if payload.get("id") != identifier:
        raise ValueError("Prediction-head state has the wrong implementation ID.")
    if set(payload) != {"version", "id", *fields}:
        raise ValueError("Prediction-head state contains missing or unexpected fields.")


class Head:
    id = "empirical_tail"

    def __init__(self):
        self._reference_logits = None

    @property
    def reference_count(self):
        return 0 if self._reference_logits is None else len(self._reference_logits)

    def fit(self, reference_logits):
        # np.sort creates a copy, so later caller mutations cannot change state.
        reference = np.sort(_logits(reference_logits, nonempty=True))
        reference.flags.writeable = False
        self._reference_logits = reference
        return self

    def score(self, logits):
        if self._reference_logits is None:
            raise ValueError("Prediction head must be fitted before scoring.")
        logits = _logits(logits)
        ranks = np.searchsorted(self._reference_logits, logits, side="right")
        tail = (ranks + 1) / (self.reference_count + 1)
        return {"tail_probability": tail, "score": -np.log2(tail)}

    def get_threshold(self, alpha):
        return _threshold(alpha)

    def to_dict(self):
        if self._reference_logits is None:
            raise ValueError("Prediction head must be fitted before serialization.")
        return {
            "version": 1,
            "id": self.id,
            "reference_logits": self._reference_logits.tolist(),
        }

    @classmethod
    def from_dict(cls, payload):
        _state(payload, cls.id, {"reference_logits"})
        return cls().fit(payload["reference_logits"])
