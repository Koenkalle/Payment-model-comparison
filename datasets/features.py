"""Extensible, outcome-free feature recipes for materialized dataset snapshots.

Add a FeatureRecipe to FEATURE_RECIPES with metadata, observable requirements,
and a calculate(context) callable. A context exposes the current payment, raw
payment statistics, and strictly earlier history; it never contains outcomes or
settlement decisions. Bump RECIPE_VERSION whenever a recipe's meaning changes.
Source columns are preserved and feature selection is a separate dataset view.
"""
from bisect import bisect_right
from collections import defaultdict, deque
from dataclasses import dataclass, field
from itertools import groupby
import math
from pathlib import Path
from types import MappingProxyType

import numpy as np

from framework.contracts import NumericDataset
from .configuration import resolved_config
from .payment_features import AMOUNT_BINS, FEATURE_SPECS, _log1p


RECIPE_VERSION = 'payment-features/v1'
HISTORY_POLICY = 'strictly-earlier-timestamps; observed payment attempts; deposits activity only; reports ignored'
WINDOW_SECONDS = 60 * 60


@dataclass(frozen=True)
class FeatureRecipe:
    definition: dict
    calculate: object


@dataclass(frozen=True)
class FeatureEvent:
    id: str
    time: float  # seconds, without rebasing the dataset clock
    amount: float
    source: object = None
    destination: object = None
    kind: str = 'payment'
    features: object = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, 'features', MappingProxyType(dict(self.features)))


@dataclass
class AccountHistory:
    out_count: int = 0
    in_count: int = 0
    out_value: float = 0.
    in_value: float = 0.
    seen: int = 0
    last: float = 0.
    recent_out_count: int = 0
    recent_in_count: int = 0
    recent_out_value: float = 0.
    recent_in_value: float = 0.


class PaymentHistory:
    """Linear sparse state: one account/pair entry and a 60-minute event queue.

    Lifetime counters retain no event records. The queue drops old transactions
    globally, including accounts that never appear again.
    """
    def __init__(self):
        self.accounts = defaultdict(AccountHistory)
        self.pairs = defaultdict(int)
        self.recent = deque()

    def advance(self, time):
        lower = time - WINDOW_SECONDS
        while self.recent and self.recent[0].time < lower:
            event = self.recent.popleft()
            source, destination = self.accounts[event.source], self.accounts[event.destination]
            source.recent_out_count -= 1
            destination.recent_in_count -= 1
            source.recent_out_value -= event.amount
            destination.recent_in_value -= event.amount
            if not source.recent_out_count:
                source.recent_out_value = 0.
            if not destination.recent_in_count:
                destination.recent_in_value = 0.

    def observe(self, event):
        if event.kind not in ('payment', 'deposit'):
            return
        for identity in {event.source, event.destination} - {None}:
            account = self.accounts[identity]
            account.seen += 1
            account.last = event.time
        if event.kind != 'payment' or event.source is None or event.destination is None:
            return
        source, destination = self.accounts[event.source], self.accounts[event.destination]
        source.out_count += 1
        destination.in_count += 1
        source.out_value += event.amount
        destination.in_value += event.amount
        source.recent_out_count += 1
        destination.recent_in_count += 1
        source.recent_out_value += event.amount
        destination.recent_in_value += event.amount
        self.pairs[event.source, event.destination] += 1
        self.recent.append(event)


@dataclass(frozen=True)
class FeatureContext:
    event: FeatureEvent
    history: PaymentHistory
    raw: dict = field(default_factory=dict)
    graph_values: dict = field(default_factory=dict)


def _context(event, history, graph_history=None):
    raw = {'log_amount': event.amount,
           'amount_bin': bisect_right(AMOUNT_BINS, event.amount)}
    if event.source is not None and event.destination is not None:
        source, destination = history.accounts[event.source], history.accounts[event.destination]
        for prefix, account in (('sender', source), ('recipient', destination)):
            for attribute in ('out_count', 'in_count', 'out_value', 'in_value', 'seen',
                              'recent_in_count', 'recent_out_count',
                              'recent_in_value', 'recent_out_value'):
                raw[prefix + '_' + attribute] = getattr(account, attribute)
            raw[prefix + '_gap'] = max(0., event.time - account.last) / 60 if account.seen else 0.
        directed = history.pairs.get((event.source, event.destination), 0)
        total = directed + history.pairs.get((event.destination, event.source), 0)
        raw.update(pair_out_count=directed, pair_total_count=total, prior_contact=float(total > 0),
                   amount_vs_sender_out_mean=event.amount / max(1., source.out_value / max(1, source.out_count)),
                   amount_vs_recipient_in_mean=event.amount / max(1., destination.in_value / max(1, destination.in_count)))
    graph_values = (graph_history.values(event.source, event.destination)
                    if graph_history is not None else {})
    return FeatureContext(event, history, raw, graph_values)


def _payment_recipe(spec):
    identifier, scale = spec['id'], spec['scale']
    def calculate(context):
        value = context.raw[identifier]
        if identifier == 'amount_bin':
            return value / len(AMOUNT_BINS)
        return _log1p(value, scale) if scale else value
    return FeatureRecipe({**spec, 'kind': 'derived', 'version': '1'}, calculate)


def _graph_recipe(spec):
    identifier = spec['id']
    return FeatureRecipe(dict(spec), lambda context: context.graph_values[identifier])


# This is the extension point; a new recipe can use any observable context field.
FEATURE_RECIPES = [_payment_recipe(spec) for spec in FEATURE_SPECS]


def _definitions(names, currency, capabilities):
    unit = currency or 'source currency units (unspecified)'
    definitions = []
    for name in names:
        definitions.append({'id': name, 'label': name.replace('_', ' '),
                            'description': 'Original numeric dataset column, preserved without modification.',
                            'group': 'source', 'kind': 'source',
                            'unit': unit if name == 'amount' else 'source units',
                            'transform': 'identity', 'window': 'source row',
                            'requirements': [], 'available': True,
                            'unavailable_reason': None, 'version': '1'})
        if name == 'log1p_amount':
            definitions[-1].update(description='Natural logarithm of one plus the source amount.',
                                   transform='log1p(amount)', unit=unit, window='current payment')
    identifiers = set(names)
    for recipe in FEATURE_RECIPES:
        definition = dict(recipe.definition)
        identifier = definition['id']
        if identifier in identifiers:
            raise ValueError('Feature recipes must have unique IDs distinct from source columns: ' + identifier)
        identifiers.add(identifier)
        missing = set(definition['requirements']) - capabilities
        definition.update(available=not missing,
                          unavailable_reason=('Requires ' + ', '.join(sorted(missing)) + '; these are not present in this dataset.') if missing else None)
        if definition['unit'] == 'currency units':
            definition['unit'] = unit
        if identifier == 'amount_bin':
            definition['amount_bins'] = list(AMOUNT_BINS)
            definition['amount_unit_assumption'] = unit
            definition['description'] += ' Fixed edges use ' + unit + '; no currency conversion or fitting is applied.'
        definitions.append(definition)
    return definitions


def _materialize(numeric, events, currency, capabilities):
    definitions = _definitions(numeric.feature_names, currency, capabilities)
    available = {definition['id'] for definition in definitions if definition['available']}
    recipes = [recipe for recipe in FEATURE_RECIPES if recipe.definition['id'] in available]
    values = np.empty((len(numeric.ids), len(recipes)), dtype=float)
    index = {identifier: row for row, identifier in enumerate(numeric.ids)}
    history = PaymentHistory()
    # Optional providers do no work for sources that lack their capabilities.
    graph_history = None
    produced = 0
    for time, group in groupby(events, key=lambda event: event.time):
        simultaneous = list(group)
        history.advance(time)
        if graph_history is not None:
            graph_history.begin_group()
        # All candidates at this time observe exactly the same preceding history.
        for event in simultaneous:
            row = index.get(event.id)
            if row is not None:
                context = _context(event, history, graph_history)
                values[row] = [recipe.calculate(context) for recipe in recipes]
                produced += 1
        for event in simultaneous:
            history.observe(event)
        if graph_history is not None:
            graph_history.observe_batch(simultaneous)
    if produced != len(numeric.ids):
        raise ValueError('Feature rows do not match source dataset identities.')
    if not np.isfinite(values).all():
        raise ValueError('Feature recipes must produce finite numeric values.')
    combined = np.column_stack((numeric.features, values))
    combined.setflags(write=False)
    return NumericDataset(numeric.ids, numeric.times, combined, numeric.labels,
                          (*numeric.feature_names, *(recipe.definition['id'] for recipe in recipes)),
                          {**numeric.provenance, 'feature_recipe_version': RECIPE_VERSION,
                           'feature_history': HISTORY_POLICY, 'feature_window_seconds': WINDOW_SECONDS,
                           'feature_history_scope': 'all preceding events in this stored dataset',
                           **({'graph_feature_history': 'strictly-earlier-payment-topology; unique edges; no future vertices',
                               'graph_feature_definitions': [definition for definition in definitions
                                                             if definition.get('group') == 'graph']}
                              if graph_history is not None else {}),
                           'feature_amount_currency': currency, 'outcomes_used': False}), definitions


def _prepared_events(stream, config):
    from .views import _selection
    selection = _selection(config)
    # A selected interval keeps its earlier observable context, including rows
    # before selection.start. No row at or after selection.stop is read.
    graph = stream.descriptor.get('graph')
    for event in stream.iter_events(stop=selection.get('stop')):
        roles = {entity.role: (entity.kind, entity.id) for entity in event.entities}
        yield FeatureEvent(event.event_id, event.event_time, event.features.get('amount', 0.),
                           roles.get(graph['source_role']) if graph else None,
                           roles.get(graph['destination_role']) if graph else None,
                           features=event.features)


def prepare_features(config, base=None):
    """Return all original numeric columns plus applicable registered features.

    Unavailable recipes remain inspectable in the returned definitions but
    never fabricate a numeric column. Prepared streams retain source amounts;
    canonical payment JSON retains EUR. Labels are copied as separate targets.
    """
    config = resolved_config(config, base)
    base = Path(base or '.')
    loader = config.get('loader')
    if loader == 'prepared_fraud':
        from .stream import open_prepared
        from .views import numeric_view
        source_config = {key: value for key, value in config.items() if key != 'features'}
        with open_prepared(config['path']) as stream:
            # Preserve the existing optional unscaled amount transform too.
            source_config['features'] = list(stream.descriptor['feature_names'])
            added_amount_transform = 'amount' in source_config['features'] and 'log1p_amount' not in source_config['features']
            if added_amount_transform:
                source_config['features'].append('log1p_amount')
            numeric = numeric_view(stream, source_config)
            capabilities = {'timestamps', *stream.descriptor['feature_names']}
            if stream.descriptor.get('graph'):
                capabilities.add('identities')
            result, definitions = _materialize(numeric, _prepared_events(stream, source_config),
                                               stream.descriptor.get('currency'), capabilities)
            if added_amount_transform:
                next(definition for definition in definitions if definition['id'] == 'log1p_amount').update(
                    kind='derived', group='payment', requirements=['amount'])
            return result, definitions
    if loader == 'payment_json':
        from .implementations.payment_json import load
        from .implementations.pipeline_numeric import numeric_payments
        dataset = load(config, base)
        numeric = numeric_payments(dataset)
        payments = [event for event in dataset.document['events'] if event['kind'] == 'payment']
        # Retain the existing three-column numeric view and expose raw amount.
        values = np.column_stack((numeric.features, [event['amount'] for event in payments]))
        values.setflags(write=False)
        numeric = NumericDataset(numeric.ids, numeric.times, values, numeric.labels,
                                 (*numeric.feature_names, 'amount'), numeric.provenance)
        events = (FeatureEvent(event['id'], event['t'] * 60, event['amount'],
                               event['u'] if event['u'] >= 0 else None, event['v'], event['kind'],
                               {'amount': event['amount']})
                  for event in dataset.document['events'])
        result, definitions = _materialize(numeric, events, 'EUR', {'amount', 'identities', 'timestamps'})
        source_info = {
            'log1p_amount': ('Natural logarithm of one plus the requested EUR amount.', 'log1p(amount)', 'current payment'),
            'prior_source_count': ('Strictly earlier observed outgoing payment attempts.', 'identity', 'all preceding history'),
            'prior_destination_count': ('Strictly earlier observed incoming payment attempts.', 'identity', 'all preceding history'),
        }
        for definition in definitions:
            if definition['id'] in source_info:
                description, transform, window = source_info[definition['id']]
                definition.update(description=description, transform=transform, window=window,
                                  unit='EUR' if definition['id'] == 'log1p_amount' else 'payments')
        return result, definitions
    raise ValueError('Feature preparation requires a prepared_fraud or payment_json source dataset.')
