"""Use observed payments as temporal links; fraud outcomes never become link labels."""
from framework.registry import load_dataset
from framework.contracts import EventDataset
from .temporal_csv import graph_dataset
from ._validation import provenance
import numpy as np


def load(config, base):
    nested = config['dataset']
    if nested.get('loader') not in ('payment_csv', 'payment_json', 'synthetic_payments'):
        raise ValueError('payment_graph requires a payment event provider.')
    dataset = load_dataset(nested, base)
    if not isinstance(dataset, EventDataset):
        raise ValueError('payment_graph requires payment events.')
    events = [event for event in dataset.document['events'] if event['kind'] == 'payment']
    names = [str(a.get('external_id', a['id'])) for a in dataset.document['accounts']]
    metadata = dict(dataset.provenance)
    # Equivalent configs in different directories should identify the same file.
    if 'path' in nested:
        from pathlib import Path
        path = Path(nested['path'])
        path = path if path.is_absolute() else base / path
        metadata.update(provenance({**nested, 'path': str(path.resolve())}, path))
    return graph_dataset(
        [e['id'] for e in events], [e['t'] * 60 for e in events],
        [names[e['u']] for e in events], [names[e['v']] for e in events],
        [[np.log1p(e['amount'])] for e in events], ['log1p_amount'],
        {**metadata, 'graph_conversion': 'observed-payments/v1',
         'outcomes_used': False, 'omitted_event_kinds': ['deposit', 'report']},
        config.get('node_feature_dim', 32))
