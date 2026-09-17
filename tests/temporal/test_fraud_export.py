"""Native fraud export: real scores, frozen calibration and held-out selection."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from framework.experiments import train_experiment
from framework.fraud_export import export_fraud
from framework.registry import create_model, load_dataset
from prediction_heads.registry import load_head

TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None
PARAMETERS = dict(epochs=1, patience=1, batch_size=8, learning_rate=0.003,
                  time_feat_dim=4, channel_embedding_dim=4, num_layers=1,
                  patch_size=2, max_input_sequence_length=8, dropout=0.0, seed=31)


@unittest.skipUnless(TORCH_AVAILABLE, 'optional PyTorch dependency')
class FraudExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temp.name)
        cls.document = {
            'accounts': [{'id': i, 'external_id': 'account-' + str(i)} for i in range(5)],
            'events': [{'id': 'deposit', 'kind': 'deposit', 'u': -1, 'v': 0, 't': 0, 'amount': 1000}],
            'truth': {},
        }
        for i in range(80):
            cls.document['events'].append({
                'id': 'payment-' + str(i), 'kind': 'payment', 'u': i % 2,
                'v': 2 + (i % 2 if i % 7 else 2), 't': i // 2,
                'amount': 10 + i % 5,
            })
            if i % 5:
                cls.document['truth']['payment-' + str(i)] = bool(i % 2)
        cls.document['events'].append({'id': 'report', 'kind': 'report', 'u': -1, 'v': 2, 't': 40, 'amount': 0})
        (cls.base / 'payments.json').write_text(json.dumps(cls.document))
        cls.event_config = {'loader': 'payment_json', 'path': 'payments.json'}
        cls.graph_config = {'loader': 'payment_graph', 'dataset': cls.event_config, 'node_feature_dim': 4}
        cls.metadata, _ = train_experiment(
            {'model': 'dyg_tami_native', 'dataset': cls.graph_config, 'parameters': PARAMETERS},
            cls.base / 'artifact', cls.base)
        cls.config = {'artifact': 'artifact', 'dataset': cls.event_config,
                      'calibration': {'dataset': cls.graph_config, 'max_rows': 3},
                      'head': 'empirical_tail', 'alpha': 0.2}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def variant(self, name, document):
        (self.base / name).write_text(json.dumps(document))
        return {**self.config, 'dataset': {'loader': 'payment_json', 'path': name}}

    def test_registered_benchmark_payment_view_exports(self):
        (self.base / 'handbook.csv').write_text(
            'TRANSACTION_ID,TX_TIME_SECONDS,CUSTOMER_ID,TERMINAL_ID,TX_AMOUNT,TX_FRAUD\n'
            'source-a,60,1,1,12,0\nsource-b,120,2,1,25,1\nsource-c,180,1,2,40,\n')
        target = {'loader': 'fraud_dataset', 'dataset': 'handbook', 'path': 'handbook.csv',
                  'view': 'payments', 'amount_to_eur': 1.}
        result = export_fraud({**self.config, 'dataset': target}, self.base)
        self.assertEqual([row['id'] for row in result['predictions']], ['source-a', 'source-b', 'source-c'])
        self.assertEqual(result['dataset']['truth'], {'source-a': False, 'source-b': True})
        self.assertEqual(result['evaluation_ids'], ['source-a', 'source-b', 'source-c'])

    def test_native_scores_heads_and_complete_time_group_calibration(self):
        result = export_fraud(self.config, self.base)
        self.assertEqual(result['schema'], 'native-fraud-comparison/v1')
        self.assertEqual(len(result['predictions']), 80)
        self.assertEqual(len(result['dataset']['events']), 82)
        self.assertEqual(result['calibration']['count'], 4)
        self.assertEqual(result['calibration']['ids'], ['payment-' + str(i) for i in range(64, 68)])
        self.assertEqual(result['evaluation_ids'], ['payment-' + str(i) for i in range(68, 80)])
        self.assertEqual(result['history'], {'mode': 'shadow', 'payments': 'observed-attempts',
                                            'timestamps': 'strictly-before', 'deposits': False, 'reports': False})
        graph = load_dataset(self.graph_config, self.base)
        native, _ = create_model('dyg_tami_native', graph.schema)
        native.load_graph(self.base / 'artifact' / 'model.npz', graph)
        expected = native.predict_graph(graph, np.arange(80))['positive_logits']
        actual = np.asarray([row['logit'] for row in result['predictions']])
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(result['heads']['empirical_tail']['reference_logits'], np.sort(expected[64:68]))
        for identifier, state in result['heads'].items():
            head = load_head(state)
            self.assertEqual(head.to_dict(), state)
            self.assertEqual(head.reference_count, 4)
            self.assertEqual(result['metrics'][identifier]['rows'], 12)
            self.assertEqual(result['metrics'][identifier]['known'] + result['metrics'][identifier]['unknown'], 12)
            flags = head.score(actual[68:])['score'] > head.get_threshold(result['alpha'])
            self.assertEqual(result['metrics'][identifier]['flagged'], int(flags.sum()))
        self.assertEqual(result['calibration']['minimum_tail_probability'], 0.2)
        self.assertFalse(result['calibration']['alpha_can_flag'])  # Strict p < alpha.
        json.dumps(result, allow_nan=False)

    def test_outcomes_do_not_change_model_or_head_and_copies_keep_test_membership(self):
        original = export_fraud(self.config, self.base)
        relabeled = copy.deepcopy(self.document)
        relabeled['truth'] = {key: not value for key, value in relabeled['truth'].items()}
        result = export_fraud(self.variant('relabeled.json', relabeled), self.base)
        self.assertEqual(result['predictions'], original['predictions'])
        self.assertEqual(result['heads'], original['heads'])
        self.assertEqual(result['evaluation_ids'], original['evaluation_ids'])
        self.assertTrue(result['provenance']['same_dataset'])
        shifted = copy.deepcopy(self.document)
        for event in shifted['events']:
            event['t'] += 1000
        result = export_fraud(self.variant('shifted.json', shifted), self.base)
        self.assertTrue(result['provenance']['same_dataset'])
        self.assertEqual(result['evaluation_ids'], original['evaluation_ids'])

    def test_calibration_original_identity_and_nonempty_evaluation(self):
        changed = copy.deepcopy(self.config)
        changed['calibration']['dataset']['node_feature_dim'] = 8
        with self.assertRaisesRegex(ValueError, 'original graph'):
            export_fraud(changed, self.base)
        changed = copy.deepcopy(self.config)
        changed['calibration']['max_rows'] = 100
        with self.assertRaisesRegex(ValueError, 'entire test partition'):
            export_fraud(changed, self.base)
        changed['calibration']['max_rows'] = 0
        with self.assertRaisesRegex(ValueError, 'positive integer'):
            export_fraud(changed, self.base)
        with self.assertRaisesRegex(ValueError, 'alpha'):
            export_fraud({**self.config, 'alpha': float('nan')}, self.base)

    def test_separate_target_and_overlap_detection(self):
        target = copy.deepcopy(self.document)
        for account in target['accounts']:
            account['external_id'] = 'new-' + account['external_id']
        result = export_fraud(self.variant('independent.json', target), self.base)
        self.assertFalse(result['provenance']['same_dataset'])
        self.assertIsNone(result['provenance']['training_rows_in_evaluation'])
        self.assertEqual(len(result['evaluation_ids']), 80)
        copied = copy.deepcopy(self.document)
        copied['truth'] = {}
        for event in copied['events']:
            event['id'] = 'copy-' + event['id']
        with self.assertRaisesRegex(ValueError, 'overlaps training'):
            export_fraud(self.variant('renamed-copy.json', copied), self.base)
        changed = copy.deepcopy(self.document)
        changed['events'][-2]['amount'] = 999
        with self.assertRaisesRegex(ValueError, 'overlaps training'):
            export_fraud(self.variant('changed-history.json', changed), self.base)

    def test_artifact_checksum_and_source_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / 'artifact'
            shutil.copytree(self.base / 'artifact', artifact)
            metadata = copy.deepcopy(self.metadata)
            metadata['implementation_sha256'] = {}
            (artifact / 'manifest.json').write_text(json.dumps(metadata))
            config = {**self.config, 'artifact': str(artifact)}
            with self.assertRaisesRegex(ValueError, 'implementation changed'):
                export_fraud(config, self.base)
            with (artifact / 'model.npz').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                export_fraud(config, self.base)

    def test_cli_writes_finite_bundle_without_overwriting(self):
        config_path = self.base / 'export-config.json'
        output = self.base / 'cli-bundle.json'
        config_path.write_text(json.dumps(self.config))
        command = [sys.executable, str(ROOT / 'experiment.py'), 'export-fraud',
                   '--config', str(config_path), '--output', str(output)]
        first = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        before = output.read_bytes()
        result = json.loads(before)
        self.assertEqual(result['head'], 'empirical_tail')
        second = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(second.returncode, 2)
        self.assertIn('already exists', second.stderr)
        self.assertEqual(output.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
