"""GRU + graph attention: explicit architecture using shared differentiable primitives."""
from ._neural import NeuralModel, train_design

MODEL_ID = "gru_attention"


class Model(NeuralModel):
    architecture = {
        "id": "gru_attention",
        "label": "GRU + graph attention",
        "hidden": 8,
        "memory": "gru",
        "readout": "attention",
        "layers": 2,
        "description": "Original design: GRU account memory plus two layers of learned temporal graph attention.",
    }


def train(args, data, labeled=None):
    return train_design(args, data, MODEL_ID, labeled)
