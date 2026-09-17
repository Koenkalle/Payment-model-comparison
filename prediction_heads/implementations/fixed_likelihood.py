"""Alternative head using a fixed cutoff on the link model's sigmoid output.

This deliberately simple alternative demonstrates head swapping. Unlike the
empirical head it does not compare queries with the reference distribution.
Link probabilities learned against sampled negative edges need not be calibrated
real-world link probabilities, and are not calibrated fraud probabilities.
"""

import numpy as np

from .empirical_tail import _logits, _state, _threshold


class Head:
    id = "fixed_likelihood"

    def __init__(self):
        self._reference_count = 0

    @property
    def reference_count(self):
        return self._reference_count

    def fit(self, reference_logits):
        self._reference_count = len(_logits(reference_logits, nonempty=True))
        return self

    def score(self, logits):
        if not self._reference_count:
            raise ValueError("Prediction head must be fitted before scoring.")
        logits = _logits(logits)
        # Evaluate the sigmoid with positive and negative branches to avoid
        # overflow. Clip only the numerically unrepresentable zero tail so that
        # score stays finite and remains exactly -log2(tail_probability).
        tail = np.empty_like(logits)
        nonnegative = logits >= 0
        tail[nonnegative] = 1 / (1 + np.exp(-logits[nonnegative]))
        exp_negative = np.exp(logits[~nonnegative])
        tail[~nonnegative] = exp_negative / (1 + exp_negative)
        tail = np.maximum(tail, np.nextafter(0.0, 1.0))
        return {"tail_probability": tail, "score": -np.log2(tail)}

    def get_threshold(self, alpha):
        return _threshold(alpha)

    def to_dict(self):
        if not self._reference_count:
            raise ValueError("Prediction head must be fitted before serialization.")
        return {"version": 1, "id": self.id, "reference_count": self._reference_count}

    @classmethod
    def from_dict(cls, payload):
        _state(payload, cls.id, {"reference_count"})
        count = payload["reference_count"]
        if type(count) is not int or count <= 0:
            raise ValueError("Saved reference count must be a positive integer.")
        head = cls()
        head._reference_count = count
        return head
