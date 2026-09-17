"""Compose independent payment episodes into a reproducible chronological history.

Every event and label is retained. Identities are namespaced per episode, so
synthetic repeated account IDs cannot accidentally share learned graph history.
"""
import copy
from framework.contracts import EventDataset
from framework.registry import load_dataset
from .payment_json import normalize


def load(config, base):
    episodes = config.get('episodes')
    if not isinstance(episodes, list) or not episodes:
        raise ValueError('payment_episodes requires a nonempty list of event-provider configurations.')
    if set(config) - {'loader', 'episodes', 'name', 'gap_minutes'}:
        raise ValueError('Unknown payment episode configuration option.')
    gap = config.get('gap_minutes', 60)
    if type(gap) not in (int, float) or not 0 < gap < 1e9:
        raise ValueError('Episode gap_minutes must be finite and positive.')
    document = {'schema': 'payment-events/v1', 'units': {'time': 'minutes', 'currency': 'EUR'},
                'name': config.get('name', 'Independent historical payment episodes'),
                'accounts': [], 'events': [], 'truth': {}, 'label_available_at': {}}
    offset, sources, boundaries = 0., [], []
    for index, provider in enumerate(episodes):
        if not isinstance(provider, dict):
            raise ValueError('Each episode requires a payment-event provider configuration.')
        dataset = load_dataset(provider, base)
        if not isinstance(dataset, EventDataset):
            if hasattr(dataset, 'close'):
                dataset.close()
            raise ValueError('Each episode requires a payment-events/v1 provider; choose view="payments" for fraud datasets.')
        episode = normalize(dataset.document, dataset.provenance)
        prefix = f'episode-{index + 1}:'
        first = min(event['t'] for event in episode['events'])
        shift = offset - first
        account_offset = len(document['accounts'])
        for account in episode['accounts']:
            document['accounts'].append({**account, 'id': account_offset + account['id'],
                                         'external_id': prefix + account['external_id']})
        for event in episode['events']:
            converted = {**event, 'id': prefix + event['id'], 't': event['t'] + shift,
                         'u': account_offset + event['u'] if event['u'] >= 0 else -1,
                         'v': account_offset + event['v']}
            if event.get('reference'):
                converted['reference'] = prefix + event['reference']
            document['events'].append(converted)
        document['truth'].update({prefix + key: value for key, value in episode['truth'].items()})
        document['label_available_at'].update({prefix + key: value + shift for key, value in episode.get('label_available_at', {}).items()})
        boundaries.append({'episode': index + 1, 'start_minutes': offset,
                           'end_minutes': document['events'][-1]['t']})
        offset = document['events'][-1]['t'] + gap
        sources.append(copy.deepcopy(dataset.provenance))
    synthetic = all(source.get('origin') == 'synthetic' for source in sources)
    metadata = {'loader': 'payment_episodes', 'origin': 'synthetic' if synthetic else 'composed',
                'configuration': copy.deepcopy(config), 'episodes': sources, 'episode_bounds': boundaries}
    document = normalize(document, metadata)
    return EventDataset(document, metadata)
