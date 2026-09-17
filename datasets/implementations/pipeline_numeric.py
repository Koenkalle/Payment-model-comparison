"""Causal numeric features and graph views of saved payment event datasets."""
from collections import Counter
from itertools import groupby

import numpy as np

from framework.contracts import EventDataset, NumericDataset
from framework.registry import load_dataset


def numeric_payments(dataset):
    """Use observed payments before each timestamp; outcomes are targets only.

    A simultaneous payment never changes another payment's features. Deposits
    and reports are not supervised payment rows or payment-count history.
    """
    events = [event for event in dataset.document['events'] if event['kind'] == 'payment']
    if not events:
        raise ValueError('The dataset contains no payment rows.')
    source_counts, destination_counts = Counter(), Counter()
    features = []
    for _, group in groupby(events, key=lambda event: event['t']):
        simultaneous = list(group)
        for event in simultaneous:
            features.append([np.log1p(event['amount']), source_counts[event['u']],
                             destination_counts[event['v']]])
        for event in simultaneous:
            source_counts[event['u']] += 1
            destination_counts[event['v']] += 1
    truth = dataset.document.get('truth', {})
    ids = tuple(event['id'] for event in events)
    times = np.asarray([event['t'] * 60 for event in events], dtype=float)
    values = np.asarray(features, dtype=float)
    labels = np.asarray([int(truth[identifier]) if identifier in truth else -1 for identifier in ids], dtype=np.int64)
    for array in (times, values, labels):
        array.setflags(write=False)
    return NumericDataset(ids, times, values, labels,
                          ('log1p_amount', 'prior_source_count', 'prior_destination_count'),
                          {**dataset.provenance, 'numeric_conversion': 'causal-payments/v1',
                           'feature_history': 'strictly-earlier-payment-timestamps', 'outcomes_used': False})


def load(config, base):
    dataset = load_dataset(config['dataset'], base)
    if not isinstance(dataset, EventDataset):
        if hasattr(dataset, 'close'):
            dataset.close()
        raise ValueError('pipeline_numeric requires a payment event dataset.')
    view = config.get('view', 'numeric')
    if view == 'numeric':
        return numeric_payments(dataset)
    if view == 'labeled_graph':
        from framework.temporal_fraud_data import labeled_payments
        return labeled_payments(dataset, {'node_feature_dim': config.get('node_feature_dim', 32)})
    raise ValueError('Unsupported saved payment view: ' + str(view))
