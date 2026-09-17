"""Explicit bridges from observable transactions to the existing model contracts.

The prepared stream is disk backed. These views intentionally materialize the
selected interval because the existing fitted models and browser use arrays.
"""
import hashlib
import json
import math

import numpy as np

from framework.contracts import EventDataset, NumericDataset


def _selection(config):
    selection = config.get('selection', {})
    if not isinstance(selection, dict) or set(selection) - {'start', 'stop'}:
        raise ValueError('selection accepts start and stop in source seconds (half-open).')
    for value in selection.values():
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('Selection bounds must be finite seconds.')
    if 'start' in selection and 'stop' in selection and selection['start'] >= selection['stop']:
        raise ValueError('Selection start must precede stop.')
    return selection


def _events(stream, config):
    events = list(stream.iter_events(**_selection(config)))
    if not events:
        raise ValueError('Dataset selection is empty.')
    return events


def _metadata(stream, config, view):
    configuration = {**config, 'view': view}
    return {**stream.provenance, 'descriptor': stream.descriptor,
            'loader': config.get('loader', stream.provenance.get('loader')),
            'source_configuration': stream.provenance.get('configuration', {}),
            'view': view, 'configuration': configuration,
            'configuration_sha256': hashlib.sha256(
                json.dumps(configuration, sort_keys=True).encode()).hexdigest()}


def _feature_names(stream, config, default=None):
    available = tuple(stream.descriptor['feature_names'])
    names = config.get('features', default if default is not None else available)
    if (not isinstance(names, (list, tuple)) or not names
            or not all(isinstance(name, str) for name in names)
            or len(set(names)) != len(names)):
        raise ValueError('features must list distinct observable numeric feature names.')
    permitted = set(available) | ({'log1p_amount'} if 'amount' in available else set())
    if set(names) - permitted:
        raise ValueError('Features must be observable inputs; available: ' + ', '.join(sorted(permitted)))
    return tuple(names)


def _values(events, names, factor=1.):
    def value(event, name):
        raw = event.features['amount'] * factor if name in ('amount', 'log1p_amount') else event.features[name]
        return math.log1p(raw) if name == 'log1p_amount' else raw
    values = np.asarray([[value(event, name) for name in names] for event in events], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Selected features must be finite numeric values.')
    return values


def numeric_view(stream, config=None):
    config = config or {}
    names = _feature_names(stream, config)
    events = _events(stream, config)
    outcomes = stream.truth_for(event.event_id for event in events)
    times = np.asarray([event.event_time for event in events], dtype=float)
    features = _values(events, names)
    labels = np.asarray([outcomes[event.event_id].label for event in events], dtype=np.int64)
    for array in (times, features, labels):
        array.setflags(write=False)
    return NumericDataset(tuple(event.event_id for event in events), times, features,
                          labels, names, _metadata(stream, config, 'numeric'))


def _require_graph(stream):
    graph = stream.descriptor.get('graph')
    if not graph:
        raise ValueError(f"{stream.descriptor['dataset_id']} has no stable relational identities; use view='numeric' or 'stream'.")
    return graph


def _endpoints(stream, events):
    graph = _require_graph(stream)
    # JSON tuple encoding prevents collisions between types, roles, and releases.
    # Truth/diagnostic edits must not rename observed entities or bypass the
    # comparison's training-overlap checks. Content hashes belong in provenance.
    namespace = [stream.descriptor['dataset_id'], stream.descriptor['version']]
    sources, destinations = [], []
    for event in events:
        roles = {entity.role: entity for entity in event.entities}
        for role, target in ((graph['source_role'], sources), (graph['destination_role'], destinations)):
            entity = roles.get(role)
            if entity is None:
                raise ValueError('Missing required graph endpoint for event ' + event.event_id)
            target.append(json.dumps([*namespace, entity.kind, entity.id], separators=(',', ':')))
    return sources, destinations


def graph_view(stream, config=None):
    from .implementations.temporal_csv import graph_dataset
    config = config or {}
    _require_graph(stream)
    names = _feature_names(stream, config, ['log1p_amount'])
    events = _events(stream, config)
    sources, destinations = _endpoints(stream, events)
    return graph_dataset([event.event_id for event in events],
                         [event.event_time for event in events], sources, destinations,
                         _values(events, names), names,
                         {**_metadata(stream, config, 'graph'), 'outcomes_used': False,
                          'amount_currency': stream.descriptor.get('currency')},
                         config.get('node_feature_dim', 32))


def payments_view(stream, config=None):
    from .implementations.payment_json import normalize
    config = config or {}
    if 'features' in config:
        raise ValueError('payments and labeled_graph views use the fixed canonical amount feature; select numeric or graph for configurable features.')
    _require_graph(stream)
    factor = config.get('amount_to_eur')
    if factor is None:
        if stream.descriptor.get('currency') != 'EUR':
            raise ValueError('The comparison uses EUR. Supply an explicit positive amount_to_eur conversion factor, or use numeric/graph/stream views in source units.')
        factor = 1.
    if type(factor) not in (int, float) or not math.isfinite(factor) or factor <= 0:
        raise ValueError('amount_to_eur must be a finite positive conversion factor.')
    events = _events(stream, config)
    sources, destinations = _endpoints(stream, events)
    outcomes = stream.truth_for(event.event_id for event in events)
    accounts, mapping = [], {}
    for event, source, destination in zip(events, sources, destinations):
        roles = {entity.role: entity for entity in event.entities}
        graph = stream.descriptor['graph']
        for identity, role in ((source, graph['source_role']), (destination, graph['destination_role'])):
            if identity not in mapping:
                entity = roles[role]
                mapping[identity] = len(accounts)
                accounts.append({'id': mapping[identity], 'external_id': identity,
                                 'name': f'{entity.kind} {entity.id}'})
    # Preserve the source-relative clock across selections. Rebasing each subset
    # would change transaction signatures and hide overlap with training data.
    origin = 0.
    metadata = {**_metadata(stream, config, 'payments'),
                'time_origin_unix_or_numeric_seconds': origin,
                'amount_conversion': {'source_currency': stream.descriptor.get('currency'),
                                      'target_currency': 'EUR', 'factor': factor},
                'event_projection': 'source transaction types projected to payment attempts'}
    document = {'schema': 'payment-events/v1',
                'name': config.get('name', stream.descriptor['dataset_id']),
                'units': {'time': 'minutes', 'currency': 'EUR'}, 'accounts': accounts,
                'events': [{'id': event.event_id, 'kind': 'payment',
                            't': (event.event_time - origin) / 60,
                            'u': mapping[source], 'v': mapping[destination],
                            'amount': event.features['amount'] * factor}
                           for event, source, destination in zip(events, sources, destinations)],
                'truth': {event.event_id: bool(outcomes[event.event_id].label)
                          for event in events if outcomes[event.event_id].label >= 0},
                'label_available_at': {event.event_id: (outcomes[event.event_id].available_at - origin) / 60
                                       for event in events if outcomes[event.event_id].label >= 0
                                       and outcomes[event.event_id].available_at is not None}}
    return EventDataset(normalize(document, metadata), metadata)


def project(stream, view, config=None):
    config = config or {}
    if view == 'stream':
        if config.get('selection'):
            raise ValueError('Use iter_events(start=..., stop=...) to select a stream interval; selection is for materialized views.')
        return stream
    if view == 'numeric':
        return numeric_view(stream, config)
    if view == 'graph':
        return graph_view(stream, config)
    if view == 'payments':
        return payments_view(stream, config)
    if view == 'labeled_graph':
        from framework.temporal_fraud_data import labeled_payments
        return labeled_payments(payments_view(stream, config),
                                {'node_feature_dim': config.get('node_feature_dim', 32)})
    raise ValueError('Unknown dataset view: ' + str(view))
