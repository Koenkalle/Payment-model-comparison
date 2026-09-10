"""GRU + graph mean: explicit architecture using shared differentiable primitives."""
from ._neural import NeuralModel,train_design
MODEL_ID='gru_mean'

class Model(NeuralModel):
    architecture={'id': 'gru_mean', 'label': 'GRU + graph mean', 'hidden': 8, 'memory': 'gru', 'readout': 'mean', 'layers': 2, 'description': 'GRU account memory plus two layers averaging direction-, amount-, and age-aware neighbor messages.'}

def train(args,data,labeled=None):
    return train_design(args,data,MODEL_ID,labeled)
