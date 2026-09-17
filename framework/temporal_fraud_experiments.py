"""Chronological supervised temporal experiments with separate policy validation."""
import copy
import dataclasses
import json
import logging
import os
import tempfile
from pathlib import Path
import numpy as np
from .registry import create_model
from .experiments import digest, choose_threshold, metrics
from .temporal_experiments import source_hashes
from .temporal_fraud_data import load_labeled_dataset, available_labels, class_counts, fingerprint

TASK = 'temporal-fraud-classification'
PARTITIONS = ('train', 'model_validation', 'policy_validation', 'test')
LOGGER = logging.getLogger(__name__)


def chronological_splits(dataset, config=None):
    config = config or {}
    unique = np.unique(dataset.times)
    if 'boundaries' in config:
        if set(config) != {'boundaries'}:
            raise ValueError('Choose explicit split boundaries or fractions, not both.')
        boundaries = np.asarray(config['boundaries'], dtype=float)
        if boundaries.shape != (3,) or not np.isfinite(boundaries).all() or np.any(np.diff(boundaries) <= 0):
            raise ValueError('Split boundaries must be three increasing finite relative seconds.')
    else:
        if set(config) - set(PARTITIONS[:-1]):
            raise ValueError('Fraud splits use train, model_validation and policy_validation fractions.')
        fractions = [config.get(key, default) for key, default in zip(PARTITIONS, (.6, .15, .1))]
        if any(type(v) not in (int, float) or not np.isfinite(v) or v <= 0 for v in fractions) or sum(fractions) >= 1:
            raise ValueError('Positive fraud split fractions must leave a nonempty test period.')
        positions = np.floor(np.cumsum(fractions) * len(unique)).astype(int)
        if not len(unique) or positions[0] < 1 or positions[-1] >= len(unique) or np.any(np.diff(positions) <= 0):
            raise ValueError('Need enough distinct timestamps for four nonempty fraud partitions.')
        boundaries = unique[positions]
    membership = np.searchsorted(boundaries, dataset.times, side='right')
    result = {name: np.flatnonzero(membership == i) for i, name in enumerate(PARTITIONS)}
    if any(not len(indices) for indices in result.values()):
        raise ValueError('Every fraud partition must contain payment timestamp groups.')
    return result


def label_cutoffs(dataset, splits, configured=None):
    configured = configured or {}
    if set(configured) - set(PARTITIONS[:-1]):
        raise ValueError('Label cutoffs apply to train, model_validation and policy_validation.')
    result = {}
    for name, following in zip(PARTITIONS, PARTITIONS[1:]):
        limit = float(np.nextafter(dataset.times[splits[following][0]], -np.inf))
        value = configured.get(name, limit)
        if type(value) not in (int, float) or not np.isfinite(value) or value > limit:
            raise ValueError(f'{name} label cutoff must precede the next partition in relative seconds.')
        result[name] = float(value)
    return result


def _path_config(config, base, relative_to=None):
    result = dict(config)
    if 'path' in result:
        path = Path(result['path'])
        path = (path if path.is_absolute() else base / path).resolve()
        result['path'] = os.path.relpath(path, relative_to) if relative_to is not None else str(path)
    if 'paths' in result:
        result['paths'] = [_path_config({'path': path}, base, relative_to)['path'] for path in result['paths']]
    if 'episodes' in result:
        result['episodes'] = [_path_config(episode, base, relative_to) for episode in result['episodes']]
    return result


def _initialize(config, dataset, splits, parameters, base, fraud_model):
    initialization = config.get('initialization', {'method': 'link_pretrain_train_partition'})
    if not isinstance(initialization, dict) or set(initialization) - {'method', 'parameters', 'artifact'}:
        raise ValueError('Invalid encoder initialization configuration.')
    method = initialization.get('method', 'link_pretrain_train_partition')
    if method == 'random':
        if parameters['encoder_training'] == 'frozen':
            raise ValueError('Frozen encoder training requires a pretrained link encoder.')
        LOGGER.info('Encoder initialization: random weights.')
        return None, {'method': 'random'}
    model, descriptor = create_model(config['model'], dataset.graph.schema, task='dynamic-link-prediction')
    if method == 'link_pretrain_train_partition':
        timestamps = np.unique(dataset.times[splits['train']])
        boundary = int(len(timestamps) * .8)
        if not 0 < boundary < len(timestamps):
            raise ValueError('Link initialization needs at least two training timestamp groups.')
        train = splits['train'][dataset.times[splits['train']] < timestamps[boundary]]
        validation = splits['train'][dataset.times[splits['train']] >= timestamps[boundary]]
        settings = fraud_model.pretraining_parameters(parameters, initialization.get('parameters', {}))
        LOGGER.info('Encoder initialization: link pretraining on the first 80%% of the fraud training period; '
                    'the remaining 20%% selects its epoch.')
        model.fit_graph(dataset.graph, train, validation, settings)
        return model, {'method': method, 'training_ids': [dataset.ids[i] for i in train],
                       'selection_ids': [dataset.ids[i] for i in validation],
                       'best_epoch': model.best_epoch, 'parameters': model.parameters,
                       'implementation_sha256': source_hashes(descriptor)}
    if method == 'checkpoint':
        path = Path(initialization['artifact'])
        path = (path if path.is_absolute() else base / path).resolve()
        saved = json.loads((path / 'manifest.json').read_text())
        if saved.get('task') != 'dynamic-link-prediction' or saved.get('model_id') != config['model']:
            raise ValueError('Initialization requires a link checkpoint for the same model.')
        if saved.get('implementation_sha256') != source_hashes(descriptor) or saved.get('model_sha256') != digest(path / 'model.npz'):
            raise ValueError('Initialization checkpoint source or weights checksum does not match.')
        checksum = dataset.provenance.get('source_sha256')
        if not checksum or saved.get('dataset', {}).get('source_sha256') != checksum:
            raise ValueError('Checkpoint initialization requires verified source data matching this training dataset.')
        from .temporal_experiments import fingerprint as link_fingerprint
        saved_graph = dataclasses.replace(dataset.graph, provenance=saved['dataset'])
        if link_fingerprint(saved_graph) != saved.get('dataset_fingerprint'):
            raise ValueError('Initialization checkpoint graph or conversion differs from the fraud dataset.')
        allowed = {dataset.ids[i] for i in splits['train']}
        used = set(saved['split']['train']) | set(saved['split']['validation'])
        if not used or not used <= allowed:
            raise ValueError('Initialization checkpoint used transactions outside the fraud training interval.')
        model.load_graph(path / 'model.npz', dataset.graph)
        if model.feature_schema != saved['features'] or model.parameters != {**saved['parameters'], 'device': model.parameters['device']}:
            raise ValueError('Initialization checkpoint metadata differs from saved features or parameters.')
        LOGGER.info('Encoder initialization: loaded verified link checkpoint %s.', path)
        return model, {'method': method, 'model_sha256': saved['model_sha256'],
                       'training_ids': saved['split']['train'], 'selection_ids': saved['split']['validation'],
                       'implementation_sha256': saved['implementation_sha256']}
    raise ValueError('Unknown encoder initialization method: ' + str(method))


def report(model, dataset, indices, threshold, partition, cutoff=None):
    from sklearn.metrics import average_precision_score, roc_auc_score
    prediction = model.predict_fraud(dataset.graph, indices)
    logits = np.asarray(prediction['fraud_logits'], dtype=np.float64)
    if logits.shape != (len(indices),) or not np.isfinite(logits).all():
        raise ValueError('Fraud model must return one finite fraud logit per payment.')
    scores = np.logaddexp(0, logits) / np.log(2)
    probabilities = np.exp(-np.logaddexp(0, -logits))
    labels = available_labels(dataset, cutoff)[indices]
    known = labels >= 0
    both = set(labels[known].tolist()) == {0, 1}
    counts = class_counts(labels)
    statistics = metrics(labels, scores, threshold)
    statistics.update({**counts, 'average_precision': float(average_precision_score(labels[known], scores[known])) if both else None,
                       'roc_auc': float(roc_auc_score(labels[known], scores[known])) if both else None,
                       'log_loss': float(np.mean(np.logaddexp(0, logits[known]) - labels[known] * logits[known])) if known.any() else None,
                       'false_block_rate': statistics['fp'] / counts['legitimate'] if counts['legitimate'] else None})
    return {'version': 1, 'schema': 'temporal-fraud-evaluation/v1', 'task': TASK,
            'partition': partition, 'threshold': float(threshold), 'threshold_units': 'fraud-surprise-bits',
            'history': 'observed attempts; strictly earlier context; frozen weights and reset state',
            'label_cutoff': cutoff, 'dataset': dataset.provenance, 'metrics': statistics,
            'metric_definitions': {'average_precision': 'Non-interpolated average precision on known payment outcomes; fraud is positive.'},
            'rows': [{'id': dataset.ids[i], 'timestamp_seconds': float(dataset.times[i]),
                      'fraud_logit': float(logits[j]), 'fraud_probability': float(probabilities[j]),
                      'score': float(scores[j]), 'label': int(labels[j]) if labels[j] >= 0 else None,
                      'decision': 'BLOCK' if scores[j] > threshold else 'ALLOW'} for j, i in enumerate(indices)]}


def train(config, output, base_dir=None):
    base = Path(base_dir or '.').resolve()
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Experiment output already exists; choose a new directory.')
    graph_config = config.get('graph', {})
    dataset = load_labeled_dataset(config['dataset'], base, graph_config)
    splits = chronological_splits(dataset, config.get('split'))
    cutoffs = label_cutoffs(dataset, splits, config.get('label_cutoffs'))
    counts = {name: class_counts(available_labels(dataset, cutoffs.get(name))[indices]) for name, indices in splits.items()}
    LOGGER.info('Fraud dataset: %d payments; train=%d (%d fraud), model_validation=%d (%d fraud), '
                'policy_validation=%d (%d fraud), test=%d (%d fraud).',
                len(dataset.ids), counts['train']['rows'], counts['train']['fraud'],
                counts['model_validation']['rows'], counts['model_validation']['fraud'],
                counts['policy_validation']['rows'], counts['policy_validation']['fraud'],
                counts['test']['rows'], counts['test']['fraud'])
    for name in ('train', 'model_validation'):
        if not counts[name]['fraud'] or not counts[name]['legitimate']:
            raise ValueError(f'{name} requires both confirmed fraud and legitimate labels available by its cutoff; counts: {counts[name]}. Supply a labeled history or adjust chronological boundaries.')
    if 'decision_threshold' in config:
        raise ValueError('Fraud experiments use score_threshold in bits, not a link probability decision_threshold.')
    threshold = config.get('score_threshold')
    if threshold is not None and (type(threshold) not in (int, float) or not np.isfinite(threshold)):
        raise ValueError('score_threshold must be finite fraud-surprise bits.')
    model, descriptor = create_model(config['model'], dataset.schema, task=TASK)
    parameters = {**config.get('parameters', {}),
                  'encoder_training': config.get('encoder_training', 'finetune'),
                  'prediction_head': config.get('prediction_head', {'id': 'fraud_linear'}),
                  'training_label_cutoff': cutoffs['train'], 'validation_label_cutoff': cutoffs['model_validation']}
    pretrained, lineage = _initialize(config, dataset, splits, parameters, base, model)
    model.fit_fraud(dataset, splits['train'], splits['model_validation'], parameters, pretrained_model=pretrained)
    LOGGER.info('Selecting the decision threshold on %d policy-validation payments.',
                len(splits['policy_validation']))
    if threshold is not None:
        threshold_source = 'configured'
    elif counts['policy_validation']['fraud'] and counts['policy_validation']['legitimate']:
        policy = model.predict_fraud(dataset.graph, splits['policy_validation'])
        scores = np.logaddexp(0, np.asarray(policy['fraud_logits'])) / np.log(2)
        threshold = choose_threshold(available_labels(dataset, cutoffs['policy_validation'])[splits['policy_validation']], scores)
        threshold_source = 'policy_validation_f1'
    else:
        threshold, threshold_source = 1., 'default_probability_0.5_missing_policy_classes'
    LOGGER.info('Evaluating the frozen best model on %d test payments.', len(splits['test']))
    result = report(model, dataset, splits['test'], threshold, 'test')
    result['training_rows_in_evaluation'] = 0
    result['threshold_source'] = threshold_source
    metadata = {'version': 1, 'task': TASK, 'model_id': descriptor['id'], 'input_schema': dataset.schema,
                'model_file': 'model.npz', 'implementation': descriptor,
                'implementation_sha256': source_hashes(descriptor), 'library_version': model.library_version,
                'parameters': model.parameters, 'encoder_training': parameters['encoder_training'],
                'prediction_head': model.parameters['prediction_head'], 'initialization': lineage,
                'feature_schema': model.feature_schema, 'graph': graph_config,
                'head_input_schema': model.head_input_schema,
                **({'feature_normalization': model.feature_normalization} if model.uses_dataset_features
                   else {'amount_normalization': model.amount_normalization}),
                'positive_class_weight': model.positive_class_weight,
                'dataset': dataset.provenance, 'dataset_config': _path_config(config['dataset'], base),
                'dataset_fingerprint': fingerprint(dataset), 'graph_fingerprint': fingerprint(dataset, include_labels=False),
                'split': {name: [dataset.ids[i] for i in indices] for name, indices in splits.items()},
                'split_bounds': {name: [float(dataset.times[indices[0]]), float(dataset.times[indices[-1]])] for name, indices in splits.items()},
                'label_cutoffs': cutoffs, 'label_counts': counts, 'threshold': float(threshold),
                'threshold_source': threshold_source, 'threshold_units': 'fraud-surprise-bits',
                'best_epoch': model.best_epoch, 'training_history': model.training_history}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.payment-fraud-', dir=output.parent) as temporary:
        stage = Path(temporary)
        model.experiment_metadata = copy.deepcopy(metadata)
        model.save(stage / 'model.npz')
        metadata['model_sha256'] = digest(stage / 'model.npz')
        (stage / 'manifest.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
        (stage / 'test-report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        (stage / 'calibration.config.json').write_text(json.dumps({
            'dataset': _path_config(config['dataset'], base, output), 'graph': graph_config}, indent=2) + '\n')
        os.rename(stage, output)
    return metadata, result


def load_artifact(artifact, dataset_config=None, base_dir=None):
    artifact = Path(artifact).resolve()
    metadata = json.loads((artifact / 'manifest.json').read_text())
    if metadata.get('version') != 1 or metadata.get('task') != TASK or metadata.get('model_file') != 'model.npz':
        raise ValueError('Unsupported supervised temporal artifact.')
    if digest(artifact / 'model.npz') != metadata['model_sha256']:
        raise ValueError('Fraud model weights checksum does not match.')
    model, descriptor = create_model(metadata['model_id'], metadata['input_schema'], task=TASK)
    if source_hashes(descriptor) != metadata['implementation_sha256']:
        raise ValueError('Fraud model implementation changed since training; regenerate the artifact.')
    if dataset_config is None:
        path = artifact / 'calibration.config.json'
        config = json.loads(path.read_text()) if path.exists() else {'dataset': metadata['dataset_config']}
        dataset_config, base_dir = config['dataset'], artifact
        if config.get('graph', metadata['graph']) != metadata['graph']:
            raise ValueError('Historical graph configuration differs from the artifact.')
    dataset = load_labeled_dataset(dataset_config, base_dir, metadata['graph'])
    if fingerprint(dataset) != metadata['dataset_fingerprint']:
        raise ValueError('Fraud artifact requires the original historical graph and outcomes.')
    if metadata['dataset'].get('source_sha256') != dataset.provenance.get('source_sha256'):
        raise ValueError('Historical dataset source checksum differs from the artifact.')
    model.load_graph(artifact / 'model.npz', dataset.graph)
    expected_metadata = {key: value for key, value in metadata.items() if key != 'model_sha256'}
    if model.experiment_metadata != expected_metadata:
        raise ValueError('Fraud experiment metadata differs from the provenance saved with its weights.')
    if model.feature_schema != metadata['feature_schema']:
        raise ValueError('Fraud artifact feature schema differs from saved weights.')
    if model.parameters != metadata['parameters']:
        # Loading on CPU is deliberately supported for checkpoints trained on GPU.
        expected = {**metadata['parameters'], 'device': model.parameters.get('device')}
        if model.parameters != expected:
            raise ValueError('Fraud artifact parameters differ from saved weights.')
    if model.parameters.get('prediction_head') != metadata['prediction_head'] or model.parameters.get('encoder_training') != metadata['encoder_training']:
        raise ValueError('Fraud artifact head or training mode differs from saved weights.')
    # Saved membership must remain complete, unique, and strictly chronological.
    indices = {identifier: i for i, identifier in enumerate(dataset.ids)}
    flattened = [identifier for name in PARTITIONS for identifier in metadata['split'][name]]
    if flattened != list(dataset.ids):
        raise ValueError('Fraud artifact partition membership is invalid.')
    splits = {name: np.asarray([indices[i] for i in metadata['split'][name]]) for name in PARTITIONS}
    if any(not len(v) for v in splits.values()) or any(dataset.times[splits[a][-1]] >= dataset.times[splits[b][0]] for a, b in zip(PARTITIONS, PARTITIONS[1:])):
        raise ValueError('Fraud artifact partitions overlap or split equal timestamps.')
    label_cutoffs(dataset, splits, metadata['label_cutoffs'])
    return metadata, dataset, model


def evaluate(artifact, dataset_config, base_dir=None, partition='all'):
    metadata, historical, model = load_artifact(artifact)
    dataset = load_labeled_dataset(dataset_config, base_dir, metadata['graph'])
    same = fingerprint(dataset) == metadata['dataset_fingerprint']
    if partition == 'all':
        indices = np.arange(len(dataset.ids))
    elif partition in PARTITIONS:
        if not same:
            raise ValueError('Saved fraud partition membership requires the original dataset and outcomes.')
        wanted = set(metadata['split'][partition])
        indices = np.asarray([i for i, identifier in enumerate(dataset.ids) if identifier in wanted])
    else:
        raise ValueError('Unknown supervised temporal partition: ' + str(partition))
    result = report(model, dataset, indices, metadata['threshold'], partition,
                    metadata['label_cutoffs'].get(partition) if same else None)
    trained = set(metadata['split']['train'])
    result['training_rows_in_evaluation'] = sum(dataset.ids[i] in trained for i in indices) if same else None
    result['threshold_source'] = metadata['threshold_source']
    return result
