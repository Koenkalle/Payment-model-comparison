"""GRU memory: explicit architecture using shared differentiable primitives."""
from ._neural import NeuralModel,train_design
MODEL_ID='gru'

class Model(NeuralModel):
    architecture={'id': 'gru', 'label': 'GRU memory', 'hidden': 8, 'memory': 'gru', 'readout': 'none', 'layers': 0, 'description': 'Role-aware account memory with counterparty state messages; no neighborhood readout.'}

def train(args,data,labeled=None):
    return train_design(args,data,MODEL_ID,labeled)
