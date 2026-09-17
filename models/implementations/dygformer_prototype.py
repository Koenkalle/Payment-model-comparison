"""Temporal history prototype (DyGFormer-inspired): fixed encoder, optional trained logistic head."""
from ._temporal import train_sequence_design, sequence_prediction, sequence_payload

CONFIG = {
    "use_pair": False,
    "use_gnn": False,
    "recipient_pair_weight": 0.4,
    "recipient_activity_weight": 0.2,
    "recipient_temporal_weight": 1.2,
    "recipient_pair_state_weight": 0.0,
    "recipient_graph_weight": 0.0,
}
MODEL_ID = "dygformer"


def train(args, data, labeled=None):
    return train_sequence_design(args, data, MODEL_ID, labeled)
