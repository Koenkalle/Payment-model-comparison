"""Build fraud-task targets from ordinary payment providers, outside graph inputs."""
import hashlib
import json
import numpy as np
from .contracts import EventDataset, LabeledTemporalGraphDataset
from .registry import load_dataset


def labeled_payments(events, graph_config=None):
    from datasets.implementations.payment_json import normalize
    from datasets.implementations.payment_graph import to_graph
    if not isinstance(events, EventDataset):
        raise ValueError('Fraud training requires a payment JSON, CSV, or generated event dataset.')
    graph_config = graph_config or {}
    if set(graph_config) - {'node_feature_dim'}:
        raise ValueError('Unknown fraud graph conversion option.')
    document = normalize(events.document, events.provenance)
    graph = to_graph(EventDataset(document, events.provenance), graph_config.get('node_feature_dim', 32))
    truth = document.get('truth', {})
    availability = document.get('label_available_at', {})
    origin = graph.provenance['time_origin_seconds']
    labels = np.asarray([int(truth[i]) if i in truth else -1 for i in graph.ids], dtype=np.int64)
    times = np.asarray([availability[i] * 60 - origin if i in availability else np.nan
                        for i in graph.ids], dtype=np.float64)
    labels.setflags(write=False)
    times.setflags(write=False)
    explicit = int(np.isfinite(times[labels >= 0]).sum())
    known = int((labels >= 0).sum())
    assumption = 'as-of' if explicit == known and known else ('mixed' if explicit else 'retrospective')
    return LabeledTemporalGraphDataset(graph, labels, times, {
        **events.provenance, 'task': 'temporal-fraud-classification',
        'label_semantics': {'fraud': 1, 'legitimate': 0, 'unknown': -1},
        'label_availability': assumption, 'explicit_label_times': explicit,
        'retrospective_labels': known - explicit,
    }, document=document)


def load_labeled_dataset(config, base_dir=None, graph_config=None):
    effective = config
    if config.get('view') == 'labeled_graph' and graph_config:
        if set(graph_config) - {'node_feature_dim'}:
            raise ValueError('Unknown fraud graph conversion option.')
        if 'node_feature_dim' in graph_config:
            if 'node_feature_dim' in config and config['node_feature_dim'] != graph_config['node_feature_dim']:
                raise ValueError('Conflicting dataset and graph node_feature_dim settings.')
            effective = {**config, **graph_config}
    dataset = load_dataset(effective, base_dir)
    if isinstance(dataset, LabeledTemporalGraphDataset):
        if graph_config:
            if (set(graph_config) - {'node_feature_dim'} or
                    graph_config.get('node_feature_dim', dataset.graph.node_features.shape[1])
                    != dataset.graph.node_features.shape[1]):
                raise ValueError('Configure node_feature_dim in the labeled_graph dataset view.')
        return dataset
    if dataset.schema == 'transaction-stream/v1':
        from datasets.views import payments_view
        try:
            return labeled_payments(payments_view(dataset, config), graph_config)
        finally:
            dataset.close()
    return labeled_payments(dataset, graph_config)


def available_labels(dataset, cutoff=None):
    """Apply outcome availability without mutating targets or historical context."""
    labels = dataset.labels.copy()
    if cutoff is not None:
        if not np.isfinite(cutoff):
            raise ValueError('Label cutoff must be finite relative seconds.')
        labels[np.isfinite(dataset.label_available_at) & (dataset.label_available_at > cutoff)] = -1
    return labels


def class_counts(labels):
    labels = np.asarray(labels)
    return {'rows': len(labels), 'fraud': int((labels == 1).sum()),
            'legitimate': int((labels == 0).sum()), 'unknown': int((labels < 0).sum())}


def fingerprint(dataset, include_labels=True):
    """Content identity independent of file location, including outcome metadata."""
    graph = dataset.graph
    hasher = hashlib.sha256(json.dumps({
        'ids': graph.ids, 'nodes': graph.node_ids, 'node_features': graph.node_feature_names,
        'edge_features': graph.edge_feature_names,
        'time_origin_seconds': graph.provenance['time_origin_seconds'],
        'source_time_origin': dataset.provenance.get('time_origin_unix_or_numeric_seconds'),
    }, sort_keys=True).encode())
    arrays = [graph.times, graph.sources, graph.destinations, graph.node_features, graph.edge_features]
    if include_labels:
        arrays += [dataset.labels, dataset.label_available_at]
    for array in arrays:
        hasher.update(str((array.shape, array.dtype)).encode())
        hasher.update(array.tobytes())
    return hasher.hexdigest()
