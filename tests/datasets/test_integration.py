"""Real runner and comparison bridges over tiny invented source-schema fixtures."""
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from framework.registry import load_dataset
from framework.experiments import train_experiment, evaluate_artifact
from framework.temporal_fraud_data import load_labeled_dataset


class DatasetIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.config = {'loader': 'fraud_dataset', 'dataset': 'handbook',
                       'path': 'handbook.csv', 'view': 'numeric'}
        with (self.base / 'handbook.csv').open('w', newline='') as output:
            writer = csv.writer(output)
            writer.writerow(['TRANSACTION_ID', 'TX_TIME_SECONDS', 'CUSTOMER_ID',
                             'TERMINAL_ID', 'TX_AMOUNT', 'TX_FRAUD', 'TX_FRAUD_SCENARIO'])
            # Reverse source order exercises preparation's actual sort, including ties.
            for index in reversed(range(80)):
                writer.writerow([str(index), 100 + index // 2, index % 3, index % 3,
                                 150 if index % 2 else 10, '' if index == 0 else index % 2,
                                 3 if index % 2 else 0])

    def tearDown(self):
        self.temporary.cleanup()

    def test_views_keep_inputs_units_identity_and_unknown_labels(self):
        numeric = load_dataset(self.config, self.base)
        self.assertEqual(numeric.feature_names, ('amount',))
        self.assertEqual(numeric.ids[:2], ('1', '0'))
        self.assertEqual(numeric.labels[:2].tolist(), [1, -1])
        self.assertFalse(numeric.features.flags.writeable)
        with self.assertRaisesRegex(ValueError, 'observable'):
            load_dataset({**self.config, 'features': ['TX_FRAUD_SCENARIO']}, self.base)
        with self.assertRaisesRegex(ValueError, 'conversion'):
            load_dataset({**self.config, 'view': 'payments'}, self.base)
        graph = load_dataset({**self.config, 'view': 'graph', 'node_feature_dim': 4}, self.base)
        self.assertEqual(graph.edge_feature_names, ('log1p_amount',))
        self.assertTrue(np.all(graph.sources != graph.destinations))
        self.assertEqual(graph.times[0], 0)
        self.assertIsNone(graph.provenance['amount_currency'])
        payment_config = {**self.config, 'view': 'payments', 'amount_to_eur': 2.}
        payments = load_dataset(payment_config, self.base)
        self.assertEqual(payments.document['events'][0]['amount'], 300.)
        self.assertNotIn('0', payments.document['truth'])
        self.assertEqual(payments.provenance['time_origin_unix_or_numeric_seconds'], 0)
        self.assertEqual(payments.document['events'][0]['t'], 100 / 60)
        self.assertFalse(any('TX_FRAUD_SCENARIO' in event for event in payments.document['events']))
        labeled = load_labeled_dataset({**payment_config, 'view': 'labeled_graph', 'node_feature_dim': 4}, self.base)
        self.assertEqual(labeled.ids, numeric.ids)
        np.testing.assert_array_equal(labeled.labels, numeric.labels)
        self.assertEqual(labeled.graph.node_features.shape[1], 4)
        wrapped = load_dataset({'loader': 'payment_graph', 'dataset': payment_config, 'node_feature_dim': 4}, self.base)
        np.testing.assert_array_equal(wrapped.edge_features, labeled.graph.edge_features)
        episodes = load_dataset({'loader': 'payment_episodes', 'episodes': [payment_config, payment_config]}, self.base)
        self.assertEqual(len(episodes.document['events']), 160)
        self.assertEqual(len({event['id'] for event in episodes.document['events']}), 160)
        with self.assertRaisesRegex(ValueError, 'fixed canonical amount'):
            load_dataset({**payment_config, 'features': ['amount']}, self.base)
        selected = load_dataset({**self.config, 'selection': {'start': 103, 'stop': 106}}, self.base)
        self.assertEqual(len(selected.ids), 6)
        self.assertEqual(selected.times.tolist(), [103, 103, 104, 104, 105, 105])

    def test_ulb_is_explicitly_tabular(self):
        with (self.base / 'ulb.csv').open('w', newline='') as output:
            writer = csv.writer(output)
            writer.writerow(['Time', 'Amount', *[f'V{i}' for i in range(1, 29)], 'Class'])
            writer.writerow([0, 10, *range(28), 0])
        config = {'loader': 'fraud_dataset', 'dataset': 'ulb', 'path': 'ulb.csv'}
        self.assertEqual(load_dataset({**config, 'view': 'numeric'}, self.base).features.shape, (1, 29))
        for view in ('graph', 'payments', 'labeled_graph'):
            with self.assertRaisesRegex(ValueError, 'no stable relational identities'):
                load_dataset({**config, 'view': view}, self.base)

    def test_prepare_reopen_inspect_and_export_cli(self):
        config_path = self.base / 'source.json'
        config_path.write_text(json.dumps({**self.config, 'view': 'stream'}))
        store = self.base / 'ordered.sqlite'
        def cli(*arguments):
            return subprocess.run([sys.executable, str(ROOT / 'experiment.py'), *arguments],
                                  cwd=self.base, text=True, capture_output=True)
        prepared = cli('prepare', '--config', str(config_path), '--output', str(store))
        self.assertEqual(prepared.returncode, 0, prepared.stderr)
        self.assertNotEqual(cli('prepare', '--config', str(config_path), '--output', str(store)).returncode, 0)
        native = load_dataset(self.config, self.base)
        reopened = load_dataset({'loader': 'prepared_fraud', 'path': str(store), 'view': 'numeric'}, self.base)
        self.assertEqual(native.ids, reopened.ids)
        np.testing.assert_array_equal(native.features, reopened.features)
        self.assertEqual(native.provenance['source_sha256'], reopened.provenance['source_sha256'])
        inspection = cli('inspect-data', '--config', str(config_path))
        self.assertEqual(inspection.returncode, 0, inspection.stderr)
        description = json.loads(inspection.stdout)
        self.assertEqual(description['label_counts']['unknown'], 1)
        self.assertIn('logistic_regression', description['models']['numeric'])
        self.assertEqual(description['descriptor']['currency'], None)
        config_path.write_text(json.dumps({**self.config, 'view': 'payments', 'amount_to_eur': 1.}))
        export = self.base / 'payments.json'
        result = cli('prepare', '--config', str(config_path), '--output', str(export))
        self.assertEqual(result.returncode, 0, result.stderr)
        from framework.comparison_service import validate_request
        document, _ = validate_request({'model_id': 'dyg_tami_native', 'dataset': json.loads(export.read_text())})
        self.assertEqual(len(document['events']), 80)
        self.assertEqual(load_dataset({'loader': 'payment_json', 'path': str(export)}).document['truth'], document['truth'])

    @unittest.skipUnless(importlib.util.find_spec('sklearn'), 'optional scikit-learn')
    def test_numeric_training_roundtrip_and_partition_guards(self):
        metadata, initial = train_experiment({'model': 'logistic_regression', 'dataset': self.config},
                                             self.base / 'logistic', self.base)
        restored = evaluate_artifact(self.base / 'logistic', self.config, self.base, 'test')
        self.assertEqual(initial['rows'], restored['rows'])
        self.assertEqual(restored['training_rows_in_evaluation'], 0)
        self.assertEqual(metadata['dataset']['descriptor']['dataset_id'], 'handbook')
        other_base = self.base / 'configurations'
        other_base.mkdir()
        equivalent = {**self.config, 'path': '../handbook.csv'}
        self.assertEqual(evaluate_artifact(self.base / 'logistic', equivalent, other_base, 'test')['rows'], initial['rows'])
        with self.assertRaisesRegex(ValueError, 'original dataset'):
            evaluate_artifact(self.base / 'logistic', {**self.config, 'selection': {'start': 101}}, self.base, 'test')

    @unittest.skipUnless(importlib.util.find_spec('sklearn'), 'optional scikit-learn')
    def test_prepared_artifact_rejects_modified_store(self):
        from datasets.stream import prepare_source
        with prepare_source({**self.config, 'view': 'stream'}, self.base, self.base / 'data.sqlite'):
            pass
        config = {'loader': 'prepared_fraud', 'path': 'data.sqlite', 'view': 'numeric'}
        train_experiment({'model': 'logistic_regression', 'dataset': config}, self.base / 'prepared-model', self.base)
        evaluate_artifact(self.base / 'prepared-model', config, self.base, 'test')
        with sqlite3.connect(self.base / 'data.sqlite') as database:
            database.execute('UPDATE events SET features=? WHERE event_id=?', (json.dumps({'amount': 999}), '79'))
        with self.assertRaisesRegex(ValueError, 'original dataset'):
            evaluate_artifact(self.base / 'prepared-model', config, self.base, 'test')

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'optional PyTorch')
    def test_fraud_training_reload_and_live_comparison(self):
        config = {'model': 'dyg_tami_native', 'task': 'temporal-fraud-classification',
                  'dataset': {**{key: value for key, value in self.config.items() if key != 'path'},
                              'paths': ['handbook.csv'], 'view': 'labeled_graph', 'amount_to_eur': 1.},
                  'graph': {'node_feature_dim': 4},
                  'initialization': {'method': 'random'},
                  'parameters': {'epochs': 1, 'patience': 1, 'batch_size': 8,
                                 'time_feat_dim': 4, 'channel_embedding_dim': 4, 'num_layers': 1,
                                 'patch_size': 2, 'max_input_sequence_length': 8, 'dropout': 0.,
                                 'seed': 31, 'device': 'cpu', 'num_threads': 1}}
        artifact = self.base / 'fraud'
        metadata, initial = train_experiment(config, artifact, self.base)
        restored = evaluate_artifact(artifact, config['dataset'], self.base, 'test')
        self.assertEqual(initial['rows'], restored['rows'])
        from framework.native_comparison import NativeComparison
        comparison = NativeComparison(artifact)
        dataset = load_labeled_dataset(config['dataset'], self.base, config['graph'])
        result = comparison.compare(dataset.document, {'trainingMode': 'supervised', 'mode': 'shadow',
                                                       'decisionPolicy': 'manual', 'predictionHead': 'fraud_linear',
                                                       'manualTau': 1.})
        self.assertEqual(result['schema'], 'native-fraud-comparison/v2')
        self.assertEqual(comparison.original.ids, tuple(event['id'] for event in dataset.document['events']))
        # Source labels/diagnostics and interval selection cannot rename historical
        # observations into apparently independent evaluation data.
        source_path = self.base / 'handbook.csv'
        original_bytes = source_path.read_bytes()
        text = source_path.read_text()
        source_path.write_text(text.replace(',150,1,3', ',150,0,99'))
        changed = load_labeled_dataset(config['dataset'], self.base, config['graph'])
        self.assertEqual(changed.graph.node_ids, dataset.graph.node_ids)
        np.testing.assert_array_equal(changed.graph.edge_features, dataset.graph.edge_features)
        selected = load_dataset({**self.config, 'view': 'payments', 'amount_to_eur': 1.,
                                 'selection': {'start': 103, 'stop': 106}}, self.base)
        with self.assertRaisesRegex(ValueError, 'overlaps historical training'):
            comparison.compare(selected.document, {'trainingMode': 'supervised', 'mode': 'shadow',
                                                   'decisionPolicy': 'manual', 'predictionHead': 'fraud_linear', 'manualTau': 1.})
        source_path.write_bytes(original_bytes)
        # Saved configs support relocation of an explicitly listed partition set.
        from framework.temporal_fraud_experiments import load_artifact
        relocated = self.base / 'relocated'
        shutil.copytree(artifact, relocated / 'fraud')
        shutil.copy(self.base / 'handbook.csv', relocated / 'handbook.csv')
        _, moved, _ = load_artifact(relocated / 'fraud')
        self.assertEqual(moved.ids, dataset.ids)


if __name__ == '__main__':
    unittest.main()
