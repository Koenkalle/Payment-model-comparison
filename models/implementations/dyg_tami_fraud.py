"""Fraud-supervised DyGFormer + TAMI with an independently registered head.

The published backbone and detached directed-pair memory are reused without
changing the link-prediction implementation or its existing artifact hashes.
Outcomes enter only the supervised loss, never the transaction representation.
"""
from dataclasses import dataclass
import json
import logging
import time

import numpy as np
import torch
from torch import nn

from .dyg_tami import Model as LinkModel, DEFAULTS
from framework.temporal_sampling import timestamp_groups
from prediction_heads.registry import create_learned_head
from prediction_heads.implementations.fraud_linear import FraudLogitScorer


FRAUD_DEFAULTS = dict(encoder_training='finetune', prediction_head={'id': 'fraud_linear'},
                      class_weight=None, training_label_cutoff=None, validation_label_cutoff=None)
ARCHITECTURE = ('time_feat_dim', 'channel_embedding_dim', 'patch_size', 'num_layers',
                'num_heads', 'dropout', 'max_input_sequence_length', 'gamma')
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TransactionRepresentation:
    """Head inputs and proposed updates computed against strictly earlier state."""
    inputs: torch.Tensor
    proposed_memory: torch.Tensor


def _graph(dataset):
    return getattr(dataset, 'graph', dataset)


def _indices(graph, values, name):
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind not in 'iu' or not len(raw):
        raise ValueError(name + ' indices must be nonempty chronological integers.')
    result = raw.astype(np.int64)
    if result[0] < 0 or result[-1] >= len(graph.ids) or np.any(np.diff(result) <= 0):
        raise ValueError(name + ' indices must be valid, unique and chronological.')
    return result


def _known(dataset, indices, cutoff):
    labels = np.asarray(dataset.labels)
    availability = np.asarray(dataset.label_available_at, dtype=np.float64)
    if labels.shape != (len(dataset.graph.ids),) or availability.shape != labels.shape:
        raise ValueError('Fraud outcomes and availability must align with graph rows.')
    if not np.isin(labels, [-1, 0, 1]).all():
        raise ValueError('Fraud labels must be -1 unknown, 0 legitimate or 1 fraud.')
    if np.isinf(availability).any():
        raise ValueError('Label availability must be finite or NaN for retrospective outcomes.')
    return (labels[indices] >= 0) & (np.isnan(availability[indices]) | (availability[indices] <= cutoff))


class Model(LinkModel):
    """One real temporal encoder and a swappable learned binary fraud head."""

    @staticmethod
    def pretraining_parameters(parameters, overrides=None):
        """Choose this model's link-initialization schedule without task coupling."""
        settings = {key: value for key, value in parameters.items() if key in DEFAULTS}
        epochs = settings.get('epochs', DEFAULTS['epochs'])
        if type(epochs) is not int or epochs < 1:
            raise ValueError('epochs must be a positive integer.')
        settings['epochs'] = min(epochs, 3)
        if overrides is not None:
            if not isinstance(overrides, dict) or set(overrides) - set(DEFAULTS):
                raise ValueError('Invalid native link-pretraining parameter override.')
            settings.update(overrides)
        return settings

    def _initialize_fraud(self, graph, parameters, pretrained_model=None):
        unknown = set(parameters) - set(DEFAULTS) - set(FRAUD_DEFAULTS)
        if unknown:
            raise ValueError('Unknown fraud-training parameters: ' + ', '.join(sorted(unknown)))
        settings = {**FRAUD_DEFAULTS, **{key: value for key, value in parameters.items() if key in FRAUD_DEFAULTS}}
        if settings['encoder_training'] not in ('frozen', 'finetune'):
            raise ValueError('encoder_training must be frozen or finetune.')
        head = settings['prediction_head']
        if isinstance(head, str):
            head = {'id': head}
        if not isinstance(head, dict) or set(head) - {'id', 'parameters'} or 'id' not in head:
            raise ValueError('prediction_head must specify a registered id and optional parameters.')
        settings['prediction_head'] = head
        base = {key: value for key, value in parameters.items() if key in DEFAULTS}
        if pretrained_model is not None:
            if pretrained_model.network is None or len(pretrained_model.network) != 2:
                raise ValueError('Initialization requires a fitted native link model.')
            for key in ARCHITECTURE:
                if key in base and base[key] != pretrained_model.parameters[key]:
                    raise ValueError('Pretrained encoder architecture differs for ' + key + '.')
            # Pretraining has its own epoch/optimizer schedule. Inherit only
            # architecture so a three-epoch initialization cannot silently
            # change the supervised stage's default ten-epoch schedule.
            base = {**{key: pretrained_model.parameters[key] for key in ARCHITECTURE}, **base}
            if pretrained_model.feature_schema != {'node': list(graph.node_feature_names), 'edge': list(graph.edge_feature_names)}:
                raise ValueError('Pretrained encoder feature schema is incompatible.')
        super()._initialize(graph, base)
        if pretrained_model is not None:
            self.network.load_state_dict(pretrained_model.network.state_dict(), strict=True)
        self.encoder_training = settings['encoder_training']
        self.parameters.update(settings)
        self.head_id = head['id']
        dim = self.network[1].fc1.out_features
        self.network.append(create_learned_head(self.head_id, 2 * dim + 1, head.get('parameters')).to(self.parameters['device']))
        self.head_input_schema = {
            'contract': 'tami-payment-representation/v1',
            'features': ['current_interaction_' + str(i) for i in range(dim)]
                        + ['previous_pair_memory_' + str(i) for i in range(dim)]
                        + ['normalized_candidate_log1p_amount'],
            'amount_unit': 'EUR', 'amount_feature': 'log1p_amount',
        }
        self.initialization = {'method': 'pretrained-link-model' if pretrained_model is not None else 'random'}
        # The link decoder's output layer is retained solely for compatibility
        # with link initialization; its scalar likelihood is not a fraud input.
        for parameter in self.network[1].fc2.parameters():
            parameter.requires_grad_(False)
        if self.encoder_training == 'frozen':
            for module in self.network[:2]:
                module.requires_grad_(False)
        self._amount_column(graph)

    @property
    def head(self):
        return self.network[2]

    def _amount_column(self, graph):
        if 'log1p_amount' not in graph.edge_feature_names:
            raise ValueError('Fraud head requires the canonical log1p_amount payment feature.')
        return list(graph.edge_feature_names).index('log1p_amount')

    def transaction_representation(self, dataset, group):
        graph = _graph(dataset)
        u, v, times = graph.sources[group], graph.destinations[group], graph.times[group]
        source, destination = self.network[0].compute_src_dst_node_temporal_embeddings(u, v, times)
        decoder = self.network[1]
        current = decoder.act(decoder.fc1(torch.cat([source, destination], dim=1)))
        previous = torch.stack(self.memory.get_memories(zip(u, v)))
        amounts = graph.edge_features[np.asarray(group) + 1, self._amount_column(graph)]
        normalized = (amounts - self.amount_normalization['mean']) / self.amount_normalization['scale']
        amount = torch.as_tensor(normalized, dtype=current.dtype, device=current.device).reshape(-1, 1)
        if not torch.isfinite(amount).all():
            raise ValueError('Candidate amount representation must be finite.')
        return TransactionRepresentation(torch.cat([current, previous, amount], dim=1),
                                         decoder.aggregate_hist_emb(current, previous))

    def score_group(self, dataset, group, negative_destinations=None):
        if negative_destinations is not None:
            raise ValueError('Fraud classification does not use sampled negative links.')
        representation = self.transaction_representation(dataset, group)
        return self.head(representation.inputs), None, representation.proposed_memory

    def fit_fraud(self, dataset, training, validation, parameters, *, pretrained_model=None):
        graph = dataset.graph
        training = _indices(graph, training, 'Training')
        validation = _indices(graph, validation, 'Validation')
        if graph.times[training].max() >= graph.times[validation].min():
            raise ValueError('Fraud fitting needs strictly separated train and validation intervals.')
        settings = {**FRAUD_DEFAULTS, **parameters}
        cutoffs = {}
        masks = {}
        for name, indices in (('training', training), ('validation', validation)):
            cutoff = settings.get(name + '_label_cutoff')
            if cutoff is None:
                cutoff = float(graph.times[indices].max())
            if isinstance(cutoff, bool) or not isinstance(cutoff, (int, float)) or not np.isfinite(cutoff):
                raise ValueError(name + '_label_cutoff must be finite seconds.')
            cutoffs[name] = float(cutoff)
            masks[name] = _known(dataset, indices, cutoff)
            known_labels = np.asarray(dataset.labels)[indices][masks[name]]
            counts = {str(value): int(np.count_nonzero(known_labels == value)) for value in (0, 1)}
            if not counts['0'] or not counts['1']:
                raise ValueError(f'{name.capitalize()} requires both confirmed classes after label availability cutoff; '
                                 f'legitimate={counts["0"]}, fraud={counts["1"]}. Provide a labeled history with both classes.')
        self._initialize_fraud(graph, parameters, pretrained_model)
        p = self.parameters
        self.label_cutoffs = cutoffs
        amounts = np.asarray(graph.edge_features[training + 1, self._amount_column(graph)], dtype=np.float64)
        if not np.isfinite(amounts).all():
            raise ValueError('Training payment amounts must be finite.')
        self.amount_normalization = {'feature': 'log1p_amount', 'mean': float(amounts.mean()),
                                     'scale': max(float(amounts.std()), 1e-6), 'fitted_rows': len(training)}
        known_labels = np.asarray(dataset.labels)[training][masks['training']]
        weight = p['class_weight']
        if weight == 'balanced':
            weight = float(np.count_nonzero(known_labels == 0) / np.count_nonzero(known_labels == 1))
        elif weight is None:
            weight = 1.0
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not np.isfinite(weight) or weight <= 0:
            raise ValueError('class_weight must be null, balanced or a positive fraud weight.')
        self.positive_class_weight = float(weight)
        self.label_counts = {'legitimate': int(np.count_nonzero(known_labels == 0)),
                             'fraud': int(np.count_nonzero(known_labels == 1)),
                             'unknown_or_unavailable': int(len(training) - len(known_labels))}
        trainable = [value for value in self.network.parameters() if value.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=p['learning_rate'], weight_decay=p['weight_decay'])
        loss_function = nn.BCEWithLogitsLoss(reduction='sum', pos_weight=torch.tensor(weight, device=p['device']))
        training_targets = {int(index): float(dataset.labels[index]) for index in training[masks['training']]}
        prefix = np.flatnonzero(graph.times <= graph.times[training[-1]])
        best_loss, best_weights, stale = float('inf'), None, 0
        self.training_history = []
        started = time.perf_counter()
        LOGGER.info('Fraud %s: %d labeled training payments (%d fraud, %d legitimate), '
                    '%d labeled validation payments, positive_weight=%.3f, up to %d epochs on %s.',
                    self.encoder_training, len(training_targets), self.label_counts['fraud'],
                    self.label_counts['legitimate'], int(masks['validation'].sum()),
                    self.positive_class_weight, p['epochs'], p['device'])
        for epoch in range(p['epochs']):
            epoch_started = time.perf_counter()
            self._bind(graph, prefix)
            self.network.train(self.encoder_training == 'finetune')
            self.head.train()
            optimizer.zero_grad()
            pending, count, total = 0, 0, 0.0

            def step():
                for value in trainable:
                    if value.grad is not None:
                        value.grad.div_(pending)
                optimizer.step()
                optimizer.zero_grad()

            for group in timestamp_groups(graph, prefix):
                offsets = [offset for offset, index in enumerate(group) if int(index) in training_targets]
                with torch.set_grad_enabled(bool(offsets)):
                    logits, _, proposed = self.score_group(graph, group)
                    if offsets:
                        targets = torch.tensor([training_targets[int(group[offset])] for offset in offsets],
                                               dtype=logits.dtype, device=logits.device)
                        loss = loss_function(logits[offsets], targets)
                        if not torch.isfinite(loss):
                            raise ValueError('Fraud training produced a nonfinite loss.')
                        loss.backward()
                        total += float(loss.detach())
                        pending += len(offsets)
                        count += len(offsets)
                self._commit(graph, group, proposed)
                if pending >= p['batch_size']:
                    step()
                    pending = 0
            if pending:
                step()
            prediction = self.predict_fraud(graph, validation)['fraud_logits'][masks['validation']]
            labels = np.asarray(dataset.labels)[validation][masks['validation']]
            val_loss = float((np.logaddexp(0, prediction) - labels * prediction).mean())
            if not np.isfinite(val_loss):
                raise ValueError('Fraud validation produced a nonfinite loss.')
            self.training_history.append({'epoch': epoch + 1, 'train_loss': total / count,
                                          'validation_loss': val_loss, 'labeled_training_rows': count})
            improved = val_loss < best_loss
            if improved:
                best_loss, stale = val_loss, 0
                best_weights = {key: value.detach().cpu().clone() for key, value in self.network.state_dict().items()}
                self.best_epoch = epoch + 1
            else:
                stale += 1
            LOGGER.info('Fraud epoch %d/%d: train_loss=%.6f validation_loss=%.6f%s epoch=%.1fs total=%.1fs',
                        epoch + 1, p['epochs'], total / count, val_loss,
                        ' new_best' if improved else f' patience={stale}/{p["patience"]}',
                        time.perf_counter() - epoch_started, time.perf_counter() - started)
            if stale >= p['patience']:
                LOGGER.info('Fraud training stopped early after epoch %d; best epoch was %d.',
                            epoch + 1, self.best_epoch)
                break
        self.network.load_state_dict(best_weights, strict=True)
        self.network.eval()
        self.memory.reset_memory()
        LOGGER.info('Fraud training complete: best_epoch=%d best_validation_loss=%.6f elapsed=%.1fs',
                    self.best_epoch, best_loss, time.perf_counter() - started)

    def predict_fraud(self, dataset, indices):
        graph = _graph(dataset)
        indices = _indices(graph, indices, 'Evaluation')
        self._bind(graph)
        self.network.eval()
        lookup = {int(index): offset for offset, index in enumerate(indices)}
        logits = np.empty(len(indices), dtype=np.float64)
        prefix = np.flatnonzero(graph.times <= graph.times[indices[-1]])
        with torch.no_grad():
            for group in timestamp_groups(graph, prefix):
                values, _, proposed = self.score_group(graph, group)
                values = values.cpu().numpy()
                for offset, index in enumerate(group):
                    if int(index) in lookup:
                        logits[lookup[int(index)]] = values[offset]
                self._commit(graph, group, proposed)
        scores = FraudLogitScorer(self.head_id).score(logits)
        return {'fraud_logits': logits, 'fraud_probabilities': scores['fraud_probability'], 'scores': scores['score']}

    def save(self, path):
        experiment_metadata = getattr(self, 'experiment_metadata', None)
        if experiment_metadata is not None and not isinstance(experiment_metadata, dict):
            raise ValueError('Fraud experiment metadata must be an object or null.')
        metadata = {'version': 1, 'task': 'temporal-fraud-classification', 'parameters': self.parameters,
                    'features': self.feature_schema, 'head': self.head.descriptor(),
                    'head_input_schema': self.head_input_schema, 'amount_normalization': self.amount_normalization,
                    'initialization': self.initialization, 'label_cutoffs': self.label_cutoffs,
                    'label_counts': self.label_counts, 'positive_class_weight': self.positive_class_weight,
                    'history': self.training_history, 'best_epoch': self.best_epoch,
                    'experiment_metadata': experiment_metadata}
        weights = {key: value.detach().cpu().numpy() for key, value in self.network.state_dict().items()}
        weights['metadata'] = np.asarray(json.dumps(metadata, allow_nan=False))
        with open(path, 'wb') as stream:
            np.savez_compressed(stream, **weights)

    def load_fraud(self, path, dataset):
        graph = _graph(dataset)
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved['metadata']))
            required = {'version', 'task', 'parameters', 'features', 'head', 'head_input_schema',
                        'amount_normalization', 'initialization', 'label_cutoffs', 'label_counts',
                        'positive_class_weight', 'history', 'best_epoch', 'experiment_metadata'}
            if set(metadata) != required or type(metadata['version']) is not int or metadata['version'] != 1 or metadata['task'] != 'temporal-fraud-classification':
                raise ValueError('Unsupported or incomplete fraud checkpoint metadata.')
            schema = {'node': list(graph.node_feature_names), 'edge': list(graph.edge_feature_names)}
            if schema != metadata['features']:
                raise ValueError('Graph feature names or order differ from the fitted fraud model.')
            self._initialize_fraud(graph, {**metadata['parameters'], 'device': 'cpu'})
            experiment_metadata = metadata['experiment_metadata']
            if experiment_metadata is not None and not isinstance(experiment_metadata, dict):
                raise ValueError('Fraud experiment metadata must be an object or null.')
            # JSON parsing accepts nonfinite extensions by default. Artifacts
            # must preserve ordinary JSON provenance so framework comparisons
            # are meaningful and the manifest can be safely serialized again.
            try:
                json.dumps(experiment_metadata, allow_nan=False)
            except (TypeError, ValueError) as error:
                raise ValueError('Fraud experiment metadata is not finite JSON.') from error
            if self.head.descriptor() != metadata['head'] or self.head_input_schema != metadata['head_input_schema']:
                raise ValueError('Fraud checkpoint has an incompatible head input contract.')
            expected = set(self.network.state_dict())
            if set(saved.files) != expected | {'metadata'}:
                raise ValueError('Fraud checkpoint has missing or unexpected tensor weights.')
            weights = {key: torch.from_numpy(saved[key].copy()) for key in expected}
            if any(not torch.isfinite(value).all() for value in weights.values()):
                raise ValueError('Fraud checkpoint contains nonfinite tensor weights.')
            try:
                self.network.load_state_dict(weights, strict=True)
            except RuntimeError as error:
                raise ValueError('Fraud checkpoint tensor dimensions are incompatible.') from error
            normalization = metadata['amount_normalization']
            if (not isinstance(normalization, dict) or set(normalization) != {'feature', 'mean', 'scale', 'fitted_rows'}
                    or normalization['feature'] != 'log1p_amount'
                    or any(type(normalization[key]) not in (int, float) or not np.isfinite(normalization[key]) for key in ('mean', 'scale'))
                    or normalization['scale'] <= 0
                    or type(normalization['fitted_rows']) is not int or normalization['fitted_rows'] < 1):
                raise ValueError('Fraud checkpoint amount normalization is invalid.')
            self.amount_normalization = normalization
            self.initialization = metadata['initialization']
            self.label_cutoffs = metadata['label_cutoffs']
            self.label_counts = metadata['label_counts']
            self.positive_class_weight = metadata['positive_class_weight']
            self.training_history = metadata['history']
            self.best_epoch = metadata['best_epoch']
            self.experiment_metadata = experiment_metadata
            self.network.eval()
            self.memory.reset_memory()

    def load_graph(self, path, dataset):
        """Common native-service loading interface; this still requires fraud weights."""
        self.load_fraud(path, dataset)


def create():
    return Model()
