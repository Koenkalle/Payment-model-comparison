"""Swappable transformations from link likelihood to transaction anomaly scores.

These heads do not estimate a calibrated fraud probability. Larger scores mean
that a transaction is less likely under the fitted link model and reference set.
"""

from .registry import create_head, load_head

__all__ = ['create_head', 'load_head']
