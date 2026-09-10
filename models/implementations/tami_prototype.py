"""Pair history prototype (TAMI-inspired): fixed encoder, optional trained logistic head."""
from ._temporal import train_sequence_design,sequence_prediction,sequence_payload
CONFIG={'use_pair': True, 'use_gnn': False, 'recipient_pair_weight': 1.2, 'recipient_activity_weight': 0.2, 'recipient_temporal_weight': 0.25, 'recipient_pair_state_weight': 1.0, 'recipient_graph_weight': 0.0}
MODEL_ID='tami'

def train(args,data,labeled=None):
    return train_sequence_design(args,data,MODEL_ID,labeled)
