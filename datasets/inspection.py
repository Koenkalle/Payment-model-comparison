"""Dataset discovery without importing a model or materializing feature arrays."""
from framework.registry import dataset_entry, load_dataset, manifest
from .views import _selection


def inspect_dataset(config, base=None):
    entry = dataset_entry(config['loader'])
    if not entry.get('views'):
        dataset = load_dataset(config, base)
        return {'schema': dataset.schema, 'provenance': dataset.provenance}
    source_config = {key: value for key, value in config.items() if key != 'selection'}
    with load_dataset({**source_config, 'view': 'stream'}, base) as stream:
        counts = {'rows': 0, 'known': 0, 'fraud': 0, 'legitimate': 0, 'unknown': 0,
                  'recorded_availability': 0}
        first = last = None
        for batch in stream.iter_batches(1024, **_selection(config)):
            truth = stream.truth_for(event.event_id for event in batch)
            if first is None:
                first = batch[0].event_time
            last = batch[-1].event_time
            for event in batch:
                outcome = truth[event.event_id]
                counts['rows'] += 1
                counts['unknown' if outcome.label < 0 else 'fraud' if outcome.label else 'legitimate'] += 1
                counts['known'] += int(outcome.label >= 0)
                counts['recorded_availability'] += int(outcome.label >= 0 and outcome.available_at is not None)
        views = dict(entry['views'])
        if not stream.descriptor.get('graph'):
            views = {key: value for key, value in views.items() if key in ('numeric', 'stream')}
        models = {}
        for view, schema in views.items():
            models[view] = [model['id'] for model in manifest('models')['models']
                            if 'python' in model['execution'] and
                            (schema in model['inputs'] or any(schema in task.get('inputs', [])
                                                             for task in model.get('tasks', {}).values()))]
        return {'schema': stream.schema, 'descriptor': stream.descriptor,
                'provenance': stream.provenance, 'total_rows': stream.count,
                'selection': config.get('selection', {}), 'label_counts': counts,
                'time_range_seconds': [first, last], 'views': views, 'models': models,
                'comparison_conversion_required': stream.descriptor.get('currency') != 'EUR'}
