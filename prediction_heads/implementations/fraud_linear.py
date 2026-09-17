"""A trainable fraud head and its stateless, monotone policy-score transform."""
import numpy as np
import torch
from torch import nn


class Head(nn.Module):
    id = "fraud_linear"
    input_contract = "tami-payment-representation/v1"
    output_contract = "fraud-logit/v1"

    def __init__(self, input_dim, parameters=None):
        super().__init__()
        if type(input_dim) is not int or input_dim < 1:
            raise ValueError("Learned head input dimension must be a positive integer.")
        if parameters:
            raise ValueError(
                "fraud_linear has no configurable architecture parameters."
            )
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, representation):
        if representation.ndim != 2 or representation.shape[1] != self.input_dim:
            raise ValueError(
                "Fraud-head transaction representation has incompatible dimensions."
            )
        return self.linear(representation).flatten()

    def descriptor(self):
        return {
            "id": self.id,
            "input_contract": self.input_contract,
            "output_contract": self.output_contract,
            "input_dim": self.input_dim,
            "parameters": {},
        }


class FraudLogitScorer:
    """Convert learned logits to estimated fraud probability and policy score.

    No calibration is fitted here. Learned weights are owned by the model; this
    adapter describes only the output transformation consumed by policies.
    """

    id = "fraud_linear"

    def __init__(self, head_id="fraud_linear"):
        from prediction_heads.registry import manifest

        ids = {
            entry["id"]
            for entry in manifest(kind="transaction-representation")["prediction_heads"]
        }
        if head_id not in ids:
            raise ValueError("Unknown learned fraud prediction head.")
        self.id = head_id

    def score(self, logits):
        values = np.asarray(logits, dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("Fraud logits must be a finite one-dimensional array.")
        probability = np.empty_like(values)
        positive = values >= 0
        probability[positive] = 1 / (1 + np.exp(-values[positive]))
        exponential = np.exp(values[~positive])
        probability[~positive] = exponential / (1 + exponential)
        return {
            "fraud_probability": probability,
            "score": np.logaddexp(0, values) / np.log(2),
        }

    def to_dict(self):
        return {
            "version": 1,
            "id": self.id,
            "kind": "learned-fraud",
            "input": "transaction-representation",
            "output": "fraud-logit",
            "score_input": "fraud-logit",
            "score_output": "fraud-probability",
            "score_transform": "softplus(logit)/ln(2)",
        }
