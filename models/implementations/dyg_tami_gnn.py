"""Temporal + pair + graph prototype: fixed encoder, optional trained logistic head."""
from ._temporal import train_sequence_design, sequence_prediction, sequence_payload

CONFIG = {
    "use_pair": True,
    "use_gnn": True,
    "recipient_pair_weight": 0.65,
    "recipient_activity_weight": 0.15,
    "recipient_temporal_weight": 0.6,
    "recipient_pair_state_weight": 0.8,
    "recipient_graph_weight": 0.8,
}
MODEL_ID = "dyg_tami_gnn"


def train(args, data, labeled=None):
    return train_sequence_design(args, data, MODEL_ID, labeled)
