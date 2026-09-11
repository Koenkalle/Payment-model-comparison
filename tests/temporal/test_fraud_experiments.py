"""Fraud-task boundaries, honest target handling, and complete saved experiments."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from framework.contracts import EventDataset
from framework.temporal_fraud_data import labeled_payments, load_labeled_dataset, available_labels, fingerprint
from framework.temporal_fraud_experiments import chronological_splits, label_cutoffs


def document():
    events, truth = [], {}
    for i in range(80):
        identifier = f'payment-{i}'
        events.append({'id': identifier, 'kind': 'payment', 'u': i % 2,
                       'v': 2 + (i % 2 if i % 7 else 2), 't': 3 + i // 2,
                       'amount': 100 if i % 4 == 1 else 10 + i % 5})
        if i % 11:
            truth[identifier] = i % 4 == 1
    return {'accounts': [{'id': i} for i in range(5)], 'events': events, 'truth': truth}


class FraudDataTests(unittest.TestCase):
    def test_targets_and_availability_stay_outside_graph_features(self):
        raw = document()
        raw['label_available_at'] = {'payment-1': 20}
        first = labeled_payments(EventDataset(raw, {}), {'node_feature_dim': 4})
        changed = copy.deepcopy(raw)
        changed['truth'] = {key: not value for key, value in raw['truth'].items()}
        second = labeled_payments(EventDataset(changed, {}), {'node_feature_dim': 4})
        self.assertEqual(fingerprint(first, False), fingerprint(second, False))
        self.assertNotEqual(fingerprint(first), fingerprint(second))
        self.assertEqual(first.labels[0], -1)
        self.assertEqual(first.label_available_at[1], 17 * 60)
        self.assertEqual(available_labels(first, 10 * 60)[1], -1)
        self.assertEqual(available_labels(first, 18 * 60)[1], 1)
        self.assertEqual(first.provenance['label_availability'], 'mixed')

    def test_fraction_and_explicit_splits_keep_timestamp_groups_together(self):
        data = labeled_payments(EventDataset(document(), {}))
        splits = chronological_splits(data)
        self.assertEqual([len(v) for v in splits.values()], [48, 12, 8, 12])
        boundaries = [data.times[splits[key][0]] for key in ('model_validation', 'policy_validation', 'test')]
        explicit = chronological_splits(data, {'boundaries': boundaries})
        for name in splits:
            np.testing.assert_array_equal(splits[name], explicit[name])
        cutoffs = label_cutoffs(data, splits)
        self.assertLess(cutoffs['train'], boundaries[0])
        with self.assertRaisesRegex(ValueError, 'precede'):
            label_cutoffs(data, splits, {'train': boundaries[-1]})
        with self.assertRaisesRegex(ValueError, 'distinct timestamps'):
            chronological_splits(data, {'train': .001})

    def test_mapped_csv_outcome_timestamps_and_unknown_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            (base / 'payments.csv').write_text('id,t,u,v,amount,fraud,confirmed\na,2020-01-01T00:00:00Z,a,b,10,1,2020-01-02T00:00:00Z\nb,2020-01-01T00:01:00Z,b,a,20,unknown,\nc,2020-01-01T00:02:00Z,a,b,5,0,\n')
            config = {'loader': 'payment_csv', 'path': 'payments.csv', 'time_unit': 'iso8601',
                      'columns': {'id': 'id', 'time': 't', 'sender': 'u', 'recipient': 'v', 'amount': 'amount', 'label': 'fraud', 'label_available_at': 'confirmed'}}
            data = load_labeled_dataset(config, base)
            np.testing.assert_array_equal(data.labels, [1, -1, 0])
            self.assertEqual(data.label_available_at[0], 86400)
            self.assertEqual(available_labels(data, 120).tolist(), [-1, -1, 0])


@unittest.skipUnless(importlib.util.find_spec('torch'), 'optional PyTorch dependency')
class FraudExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.raw = document()
        (self.base / 'payments.json').write_text(json.dumps(self.raw))
        self.config = {'model': 'dyg_tami_native', 'task': 'temporal-fraud-classification',
                       'dataset': {'loader': 'payment_json', 'path': 'payments.json'},
                       'graph': {'node_feature_dim': 4}, 'initialization': {'method': 'random'},
                       'parameters': {'epochs': 1, 'patience': 1, 'batch_size': 8, 'learning_rate': .003,
                                      'time_feat_dim': 4, 'channel_embedding_dim': 4, 'num_layers': 1,
                                      'patch_size': 2, 'max_input_sequence_length': 8, 'dropout': 0., 'seed': 31}}

    def tearDown(self):
        self.temporary.cleanup()

    def train(self, name='artifact'):
        from framework.experiments import train_experiment
        return train_experiment(self.config, self.base / name, self.base)

    def test_real_training_roundtrip_relocation_and_artifact_checks(self):
        from framework.experiments import evaluate_artifact
        from framework.temporal_fraud_experiments import load_artifact
        metadata, report = self.train()
        evaluated = evaluate_artifact(self.base / 'artifact', self.config['dataset'], self.base, 'test')
        self.assertEqual(evaluated['rows'], report['rows'])
        self.assertEqual(evaluated['metrics'], report['metrics'])
        self.assertEqual(metadata['threshold_source'], 'policy_validation_f1')
        self.assertGreater(metadata['label_counts']['train']['fraud'], 0)
        self.assertGreater(metadata['label_counts']['train']['unknown'], 0)
        self.assertEqual(metadata['amount_normalization']['fitted_rows'], 48)
        relocated = self.base / 'relocated'
        shutil.copytree(self.base / 'artifact', relocated / 'artifact')
        shutil.copy(self.base / 'payments.json', relocated / 'payments.json')
        _, data, model = load_artifact(relocated / 'artifact')
        self.assertEqual(len(data.ids), 80)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.train()
        saved = relocated / 'artifact' / 'manifest.json'
        altered = json.loads(saved.read_text())
        altered['encoder_training'] = 'frozen'
        saved.write_text(json.dumps(altered))
        with self.assertRaisesRegex(ValueError, 'provenance'):
            load_artifact(relocated / 'artifact')
        altered = copy.deepcopy(metadata)
        moved = altered['split']['model_validation'][:2]
        altered['split']['train'].extend(moved)
        altered['split']['model_validation'] = altered['split']['model_validation'][2:]
        saved.write_text(json.dumps(altered))
        with self.assertRaisesRegex(ValueError, 'provenance'):
            load_artifact(relocated / 'artifact')

    def test_test_label_changes_leave_fitted_tensors_and_selection_unchanged(self):
        before, first = self.train()
        for identifier in before['split']['test']:
            if identifier in self.raw['truth']:
                self.raw['truth'][identifier] = not self.raw['truth'][identifier]
        (self.base / 'payments.json').write_text(json.dumps(self.raw))
        after, second = self.train('changed-labels')
        self.assertEqual(before['training_history'], after['training_history'])
        self.assertEqual(before['threshold'], after['threshold'])
        with np.load(self.base / 'artifact' / 'model.npz', allow_pickle=False) as original, np.load(self.base / 'changed-labels' / 'model.npz', allow_pickle=False) as changed:
            for key in original.files:
                if key == 'metadata':
                    first_metadata = json.loads(str(original[key]))
                    second_metadata = json.loads(str(changed[key]))
                    # The audit records the changed source/held-out labels;
                    # all learned state and selection must remain identical.
                    first_metadata.pop('experiment_metadata')
                    second_metadata.pop('experiment_metadata')
                    self.assertEqual(first_metadata, second_metadata)
                else:
                    np.testing.assert_array_equal(original[key], changed[key])
        self.assertEqual([r['score'] for r in first['rows']], [r['score'] for r in second['rows']])
        self.assertNotEqual(first['metrics'], second['metrics'])

    def test_missing_class_fails_early_and_missing_policy_classes_has_fixed_default(self):
        for identifier in list(self.raw['truth']):
            if int(identifier.split('-')[-1]) < 48:
                self.raw['truth'][identifier] = False
        (self.base / 'payments.json').write_text(json.dumps(self.raw))
        with self.assertRaisesRegex(ValueError, 'train requires both'):
            self.train()
        self.assertFalse((self.base / 'artifact').exists())
        self.raw = document()
        for i in range(60, 68):
            self.raw['truth'].pop(f'payment-{i}', None)
        (self.base / 'payments.json').write_text(json.dumps(self.raw))
        metadata, result = self.train()
        self.assertEqual(metadata['threshold'], 1)
        self.assertEqual(metadata['label_counts']['policy_validation']['unknown'], 8)
        self.assertIn('missing_policy_classes', metadata['threshold_source'])

    def test_composed_file_episodes_rebase_paths_for_saved_artifacts(self):
        from framework.temporal_fraud_experiments import load_artifact
        (self.base / 'second.json').write_text(json.dumps(document()))
        self.config['dataset'] = {'loader': 'payment_episodes', 'episodes': [
            {'loader': 'payment_json', 'path': 'payments.json'},
            {'loader': 'payment_json', 'path': 'second.json'}]}
        metadata, _ = self.train()
        restored, data, _ = load_artifact(self.base / 'artifact')
        self.assertEqual(restored, metadata)
        self.assertEqual(len(data.ids), 160)
        self.assertEqual(len(data.document['accounts']), 10)
        self.assertIn('episode-1:payment-1', data.document['truth'])
        self.assertIn('episode-2:payment-1', data.document['truth'])

    def test_initialization_rejects_mismatched_graph_conversion(self):
        from framework.experiments import train_experiment
        link_config = {'model': 'dyg_tami_native', 'dataset': {
            'loader': 'payment_graph', 'node_feature_dim': 4, 'dataset': self.config['dataset']},
            'parameters': self.config['parameters'], 'split': {'train': .4, 'validation': .1}}
        train_experiment(link_config, self.base / 'link', self.base)
        self.config['initialization'] = {'method': 'checkpoint', 'artifact': 'link'}
        self.config['graph']['node_feature_dim'] = 8
        with self.assertRaisesRegex(ValueError, 'graph or conversion'):
            self.train()
        self.config['graph']['node_feature_dim'] = 4
        metadata, _ = self.train()
        self.assertEqual(metadata['initialization']['method'], 'checkpoint')


if __name__ == '__main__':
    unittest.main()
