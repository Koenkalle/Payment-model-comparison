"""History statistics: explicit architecture using shared differentiable primitives."""
from ._neural import NeuralModel,train_design
MODEL_ID='statistics'

class Model(NeuralModel):
    architecture={'id': 'statistics', 'label': 'History statistics', 'hidden': 0, 'memory': 'none', 'readout': 'none', 'layers': 0, 'description': 'Learned categorical predictors using account and pair statistics; no neural memory or neighborhood aggregation.'}

def train(args,data,labeled=None):
    return train_design(args,data,MODEL_ID,labeled)
