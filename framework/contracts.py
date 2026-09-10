"""Data and model contracts independent of file formats, algorithms and tools."""
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import numpy as np

@dataclass(frozen=True)
class NumericDataset:
    ids: tuple[str,...]
    times: np.ndarray
    features: np.ndarray
    labels: np.ndarray  # -1 unknown, 0 legitimate, 1 flagged
    feature_names: tuple[str,...]
    provenance: dict
    schema: str='numeric-table/v1'

@dataclass(frozen=True)
class EventDataset:
    document: dict
    provenance: dict
    schema: str='payment-events/v1'

@dataclass(frozen=True)
class TemporalGraphDataset:
    """Observed links; row zero of node/edge features is padding, never a node.

    Edge feature row i+1 belongs to interaction i. Outcome labels, if available,
    are metadata only: all observed interactions are positives for link prediction.
    """
    ids: tuple[str,...]
    times: np.ndarray
    sources: np.ndarray
    destinations: np.ndarray
    node_ids: tuple[str,...]
    node_features: np.ndarray
    edge_features: np.ndarray
    node_feature_names: tuple[str,...]
    edge_feature_names: tuple[str,...]
    provenance: dict
    schema: str='temporal-graph/v1'

@dataclass(frozen=True)
class Predictions:
    probabilities: np.ndarray
    margins: np.ndarray|None=None
    contributions: np.ndarray|None=None  # final column is the expected margin

class FittedModel(Protocol):
    def fit(self,x:np.ndarray,y:np.ndarray,feature_names:tuple[str,...],parameters:dict)->None: ...
    def predict(self,x:np.ndarray,feature_names:tuple[str,...],explain:bool=False)->Predictions: ...
    def save(self,path:Path)->None: ...
    def load(self,path:Path,feature_names:tuple[str,...])->None: ...

class FittedTemporalModel(Protocol):
    def fit_graph(self,dataset:TemporalGraphDataset,training:np.ndarray,validation:np.ndarray,parameters:dict)->None: ...
    def predict_graph(self,dataset:TemporalGraphDataset,indices:np.ndarray,seed:int)->dict: ...
    def save(self,path:Path)->None: ...
    def load_graph(self,path:Path,dataset:TemporalGraphDataset)->None: ...
