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
class Predictions:
    probabilities: np.ndarray
    margins: np.ndarray|None=None
    contributions: np.ndarray|None=None  # final column is the expected margin

class FittedModel(Protocol):
    def fit(self,x:np.ndarray,y:np.ndarray,feature_names:tuple[str,...],parameters:dict)->None: ...
    def predict(self,x:np.ndarray,feature_names:tuple[str,...],explain:bool=False)->Predictions: ...
    def save(self,path:Path)->None: ...
    def load(self,path:Path,feature_names:tuple[str,...])->None: ...
