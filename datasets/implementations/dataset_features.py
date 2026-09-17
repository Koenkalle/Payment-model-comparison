"""Load stored dataset feature columns without running a feature recipe.

The numeric archive owns row IDs, source seconds, targets and selected columns.
Graph views retain their original identities, clock and label availability and
replace only their edge attributes with these same stored column values.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from framework.contracts import NumericDataset, TemporalGraphDataset, LabeledTemporalGraphDataset
from framework.registry import load_dataset


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_numeric(path, dataset):
    """A pickle-free, lossless archive with an explicit, ordered column schema."""
    np.savez_compressed(path, ids=np.asarray(dataset.ids, dtype=str), times=dataset.times,
                        features=dataset.features, labels=dataset.labels,
                        feature_names=np.asarray(dataset.feature_names, dtype=str),
                        provenance=np.asarray(json.dumps(dataset.provenance, allow_nan=False)))


def read_numeric(path, checksum=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Invalid stored feature payload.')
    if checksum and file_hash(path) != checksum:
        raise ValueError('Stored feature payload checksum changed. Create a new dataset snapshot.')
    try:
        with np.load(path, allow_pickle=False) as archive:
            ids = tuple(archive['ids'].tolist())
            names = tuple(archive['feature_names'].tolist())
            times = np.asarray(archive['times'], dtype=np.float64)
            values = np.asarray(archive['features'], dtype=np.float64)
            labels = np.asarray(archive['labels'], dtype=np.int64)
            provenance = json.loads(str(archive['provenance'].item()))
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError('Invalid stored feature payload.') from error
    if (not ids or not all(isinstance(value, str) and value for value in ids)
            or len(set(ids)) != len(ids) or not names
            or not all(isinstance(value, str) and value for value in names)
            or len(set(names)) != len(names) or values.shape != (len(ids), len(names))
            or times.shape != (len(ids),) or labels.shape != (len(ids),)
            or not np.isfinite(values).all() or not np.isfinite(times).all()
            or not np.isin(labels, [-1, 0, 1]).all() or not isinstance(provenance, dict)):
        raise ValueError('Invalid stored feature matrix, row identities or targets.')
    for array in (times, values, labels):
        array.setflags(write=False)
    return NumericDataset(ids, times, values, labels, names, provenance)


def load(config, base):
    path = Path(config['path'])
    numeric = read_numeric(path if path.is_absolute() else base / path, config.get('payload_sha256'))
    if list(numeric.feature_names) != config.get('feature_names'):
        raise ValueError('Stored feature columns do not match the dataset configuration.')
    metadata = {**numeric.provenance, 'feature_recipe_version': config['recipe_version'],
                'dataset_feature_fingerprint': config['fingerprint'], 'features_stored': True,
                'outcomes_used': False}
    view = config.get('view', 'numeric')
    if view == 'numeric':
        return replace(numeric, provenance=metadata)
    if view not in ('graph', 'labeled_graph'):
        raise ValueError('Unsupported stored feature view: ' + str(view))
    source = dict(config['dataset'])
    if 'node_feature_dim' in config:
        source['node_feature_dim'] = config['node_feature_dim']
    data = load_dataset(source, base)
    graph = data.graph if isinstance(data, LabeledTemporalGraphDataset) else data
    if not isinstance(graph, TemporalGraphDataset):
        raise ValueError('Stored graph feature columns require a graph dataset.')
    mapping = {identifier: index for index, identifier in enumerate(numeric.ids)}
    if len(graph.ids) != len(numeric.ids) or set(graph.ids) != set(mapping):
        raise ValueError('Stored feature rows do not match the source graph.')
    order = np.asarray([mapping[identifier] for identifier in graph.ids], dtype=np.int64)
    origin = graph.provenance.get('time_origin_seconds', 0.)
    if not np.allclose(numeric.times[order] - origin, graph.times, rtol=0, atol=1e-7):
        raise ValueError('Stored feature timestamps do not match the source graph.')
    # The graph contract reserves row zero for padding. Retain its native dtype
    # so temporal models receive the float32 inputs their networks expect.
    with np.errstate(over='ignore'):
        edges = np.vstack([np.zeros((1, len(numeric.feature_names))), numeric.features[order]]).astype(graph.edge_features.dtype)
    if not np.isfinite(edges).all():
        raise ValueError('Selected feature values exceed the numeric range supported by graph models.')
    edges.setflags(write=False)
    graph = replace(graph, edge_features=edges, edge_feature_names=numeric.feature_names,
                    provenance={**graph.provenance, **metadata})
    if isinstance(data, LabeledTemporalGraphDataset):
        if not np.array_equal(numeric.labels[order], data.labels):
            raise ValueError('Stored feature targets do not match the source graph.')
        return replace(data, graph=graph, provenance={**data.provenance, **metadata})
    return graph
