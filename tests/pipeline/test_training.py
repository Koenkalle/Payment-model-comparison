"""Real fitting plus persistence, shared-population and overlap regression tests."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from framework.pipeline_training import TrainingService


class Store:
    """Small numeric-only stored datasets; the service uses the public store API."""
    def __init__(self, root):
        self.root = root
        self.data = {}
        self.add('original')

    def add(self, identifier, offset=0, rows=None, rename=False, unknown=False, feature='amount'):
        path = self.root / (identifier + '.csv')
        rows = list(range(100)) if rows is None else rows
        content = ['id,time,' + feature + ',label']
        for index in rows:
            outcome = '' if unknown and index >= 80 else str(index % 2)
            value = f'{"copy" if rename else "p"}-{index}'
            content.append(f'{value},{index // 2 + offset},{20 + index % 2 * 80 + index % 3},{outcome}')
        path.write_text('\n'.join(content) + '\n')
        self.data[identifier] = {'id': identifier, 'name': identifier, 'views': ['numeric'],
                                 'config': {'loader': 'numeric_csv', 'path': str(path), 'features': [feature],
                                            'id_column': 'id', 'time_column': 'time', 'label_column': 'label'}}

    def get(self, identifier):
        if identifier not in self.data:
            raise ValueError('Unknown stored dataset.')
        return copy.deepcopy(self.data[identifier])

    def dataset_config(self, identifier, view):
        if view not in self.data[identifier]['views']:
            raise ValueError('Unsupported view')
        return copy.deepcopy(self.data[identifier]['config'])


class PaymentsStore(Store):
    def add(self, identifier, offset=0, **kwargs):
        super().add(identifier, offset, **kwargs)
        raw = {'accounts': [{'id': i} for i in range(5)], 'events': [], 'truth': {}}
        for index in range(100):
            row_id = f'p-{index}'
            raw['events'].append({'id': row_id, 'kind': 'payment', 'u': index % 2,
                                  'v': 2 + index % 3, 't': index // 2 + offset,
                                  'amount': 20 + index % 2 * 80 + index % 3})
            raw['truth'][row_id] = bool(index % 2)
        path = self.root / (identifier + '.json')
        path.write_text(json.dumps(raw))
        self.data[identifier]['payment_config'] = {'loader': 'payment_json', 'path': str(path)}
        self.data[identifier]['views'] = ['numeric', 'payments', 'labeled_graph']

    def dataset_config(self, identifier, view):
        if view in ('payments', 'labeled_graph'):
            return copy.deepcopy(self.data[identifier]['payment_config'])
        return super().dataset_config(identifier, view)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.store = Store(self.base)
        self.service = TrainingService(self.store, self.base / 'pipeline')

    def tearDown(self):
        self.service.close()
        self.temporary.cleanup()

    def wait(self, job):
        self.service._futures[job['id']].result(timeout=40)
        return self.service.job(job['id'])

    def train(self, **overrides):
        job = self.wait(self.service.start_training({'dataset_id': 'original', 'model_id': 'logistic_regression', **overrides}))
        self.assertEqual(job['status'], 'succeeded', job.get('error'))
        return job['result']['run']

    def compare(self, runs, dataset_id='original', partition='test'):
        return self.wait(self.service.start_comparison({'dataset_id': dataset_id,
                         'run_ids': [run['id'] for run in runs], 'partition': partition}))

    def test_real_training_persists_ready_artifact_splits_and_configuration(self):
        run = self.train(parameters={'C': .3}, name='My tuned baseline')
        self.assertEqual(run['partition_counts'], {'train': 60, 'validation': 20, 'test': 20})
        self.assertEqual(run['parameters']['C'], .3)
        self.assertEqual(run['name'], 'My tuned baseline')
        folder = self.base / 'pipeline' / 'runs' / run['id']
        config = json.loads((folder / 'configuration.json').read_text())
        self.assertEqual(config['parameters']['C'], .3)
        self.assertEqual(self.service.runs(), [run])
        self.service.close()
        self.service = TrainingService(self.store, self.base / 'pipeline')
        self.assertEqual(self.service.runs(), [run])
        self.assertEqual(self.service.job(run['job_id'])['status'], 'succeeded')

    def test_shared_heldout_population_intersects_different_splits(self):
        first = self.train(split={'train': .5, 'validation': .2})
        second = self.train(split={'train': .6, 'validation': .2})
        job = self.compare([first, second])
        self.assertEqual(job['status'], 'succeeded', job.get('error'))
        result = job['result']
        self.assertEqual(result['row_count'], 20)
        self.assertEqual(set(result['evaluation_ids']), {f'p-{i}' for i in range(80, 100)})
        self.assertEqual(result['population'], 'common-held-out-test')
        self.assertEqual(result['training_rows_in_evaluation'], 0)
        self.assertEqual(result['validation_rows_in_evaluation'], 0)
        self.assertTrue(all(set(row['predictions']) == {first['id'], second['id']} for row in result['rows']))
        self.assertTrue(all(model['metrics']['rows'] == 20 for model in result['models']))
        self.assertNotIn('rows', next(row for row in self.service.jobs() if row['id'] == job['id'])['result'])
        self.assertTrue((self.base / 'pipeline' / 'comparisons' / (job['id'] + '.json')).exists())

    def test_all_rows_on_original_or_renamed_original_rejected(self):
        run = self.train()
        self.store.add('duplicate')
        for identifier in ('original', 'duplicate'):
            job = self.compare([run], identifier, 'all')
            self.assertEqual(job['status'], 'failed')
            self.assertIn('held-out test', job['error'])
        job = self.compare([run], 'duplicate', 'test')
        self.assertEqual(job['status'], 'succeeded', job.get('error'))
        self.assertEqual(job['result']['row_count'], 20)

    def test_partial_copy_with_renamed_row_ids_cannot_be_new_test_data(self):
        run = self.train()
        self.store.add('overlap', rows=range(30, 95), rename=True)
        job = self.compare([run], 'overlap', 'all')
        self.assertEqual(job['status'], 'failed')
        self.assertIn('overlap training or validation', job['error'])

    def test_source_namespace_prevents_reused_rows_hidden_by_a_different_feature_view(self):
        self.store.data['original']['source_namespace'] = ['source', 'release-1']
        run = self.train()
        self.store.add('reprojected', feature='balance')
        self.store.data['reprojected']['source_namespace'] = ['source', 'release-1']
        job = self.compare([run], 'reprojected', 'all')
        self.assertEqual(job['status'], 'failed')
        self.assertIn('overlap training or validation', job['error'])

    def test_new_dataset_requires_explicit_all_and_excludes_unknown_labels(self):
        run = self.train()
        self.store.add('fresh', offset=1000, unknown=True)
        failure = self.compare([run], 'fresh')
        self.assertEqual(failure['status'], 'failed')
        self.assertIn('Explicitly choose all', failure['error'])
        job = self.compare([run], 'fresh', 'all')
        self.assertEqual(job['status'], 'succeeded', job.get('error'))
        result = job['result']
        self.assertEqual(result['population'], 'new-dataset')
        self.assertEqual((result['row_count'], result['known'], result['unknown']), (100, 80, 20))
        self.assertEqual(result['models'][0]['metrics']['known'], 80)
        self.assertEqual(result['models'][0]['metrics']['unknown'], 20)

    def test_incompatible_feature_views_fail_without_fabricated_predictions(self):
        run = self.train()
        self.store.add('different', offset=1000, feature='balance')
        job = self.compare([run], 'different', 'all')
        self.assertEqual(job['status'], 'failed')
        self.assertIn('feature names', job['error'])
        self.assertNotIn('result', job)

    def test_invalid_parameters_never_queue_arbitrary_paths_or_code(self):
        cases = [{'parameters': {'device': 'cuda'}}, {'parameters': {'C': float('nan')}},
                 {'parameters': {'max_iter': True}}, {'parameters': {'class_weight': {'1': 2}}},
                 {'split': {'train': .9, 'validation': .2}}, {'parameters': {'max_iter': 5001}},
                 {'dataset_id': '../../private'}, {'name': ''}, {'output': '/tmp/untrusted'},
                 {'model_id': []},
                 {'parameters': {'decision_threshold': 1}}, {'parameters': {'decision_threshold': True}}]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                self.service.start_training({'dataset_id': 'original', 'model_id': 'logistic_regression', **case})
        self.assertEqual(self.service.jobs(), [])

    def test_dependency_and_dataset_incompatibility_are_explicit(self):
        self.service._dependencies['xgboost'] = 'missing for test'
        models = {row['id']: row for row in self.service.models('original')}
        self.assertTrue(models['logistic_regression']['available'])
        self.assertFalse(models['dyg_tami_native']['available'])
        self.assertIn('labeled_graph', models['dyg_tami_native']['reason'])
        self.assertFalse(models['xgboost_native']['available'])
        self.assertIn('xgboost', models['xgboost_native']['reason'])
        with self.assertRaisesRegex(ValueError, 'xgboost'):
            self.service.start_training({'dataset_id': 'original', 'model_id': 'xgboost_native'})
        self.store.data['original'].update(fraud=0, legitimate=100)
        model = next(row for row in self.service.models('original') if row['id'] == 'logistic_regression')
        self.assertFalse(model['available'])
        self.assertIn('both known fraud', model['reason'])

    def test_changed_weights_are_no_longer_ready(self):
        run = self.train()
        path = self.base / 'pipeline' / 'runs' / run['id'] / 'artifact' / 'model.json'
        path.write_text('{}')
        self.assertEqual(self.service.runs(), [])
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.service.start_comparison({'dataset_id': 'original', 'run_ids': [run['id']]})

    def test_second_live_owner_rejected_and_dead_worker_jobs_fail_on_restart(self):
        with self.assertRaisesRegex(ValueError, 'Another live training server'):
            TrainingService(self.store, self.base / 'pipeline')
        self.service.close()
        path = self.base / 'pipeline' / 'jobs' / 'stopped.json'
        path.write_text(json.dumps({'id': 'stopped', 'kind': 'training', 'status': 'running',
                                    'created_at': '2026-01-01', 'worker_pid': 1234567}))
        self.service = TrainingService(self.store, self.base / 'pipeline')
        job = self.service.job('stopped')
        self.assertEqual(job['status'], 'failed')
        self.assertEqual(job['stage'], 'interrupted')
        self.assertIn('Server stopped', job['error'])

    def test_failed_fitting_persists_error_and_never_exposes_ready_run(self):
        with patch('framework.pipeline_training.train_experiment', side_effect=ValueError('both classes are required')):
            job = self.wait(self.service.start_training({'dataset_id': 'original', 'model_id': 'logistic_regression'}))
        self.assertEqual(job['status'], 'failed')
        self.assertIn('both classes', job['error'])
        self.assertEqual(self.service.runs(), [])

    def test_numeric_delayed_labels_are_masked_for_fit_and_selection_only(self):
        from framework.registry import load_dataset
        self.service.close()
        self.store = PaymentsStore(self.base)
        path = self.base / 'original.json'
        raw = json.loads(path.read_text())
        raw['label_available_at'] = {'p-1': 1000, 'p-61': 1000, 'p-81': 1000}
        path.write_text(json.dumps(raw))
        self.service = TrainingService(self.store, self.base / 'pipeline')
        run = self.train()
        self.assertEqual(run['label_policy']['masked'], {'train': 1, 'validation': 1})
        config = json.loads((self.base / 'pipeline' / 'runs' / run['id'] / 'configuration.json').read_text())
        training = load_dataset(config['dataset'])
        self.assertEqual(training.labels[1], -1)
        self.assertEqual(training.labels[61], -1)
        self.assertEqual(training.labels[81], 1)
        self.assertEqual(len(self.service.runs()), 1)
        job = self.compare([run])
        self.assertEqual(job['status'], 'succeeded', job.get('error'))
        self.assertEqual(job['result']['known'], 20)
        self.assertEqual(next(row for row in job['result']['rows'] if row['id'] == 'p-81')['label'], 1)

    def test_synthetic_single_class_validation_requires_an_explicit_fixed_cutoff(self):
        from framework.pipeline_data import DatasetStore
        self.service.close()
        self.store = DatasetStore(self.base / 'stored')
        dataset = self.store.generate({'generator': 'synthetic_payments',
                                      'parameters': {'name': 'mixed', 'size': 'medium', 'seed': 42}})
        self.service = TrainingService(self.store, self.base / 'pipeline')
        failed = self.wait(self.service.start_training({'dataset_id': dataset['id'],
                           'model_id': 'logistic_regression', 'parameters': {'decision_threshold': None}}))
        self.assertEqual(failed['status'], 'failed')
        self.assertIn('both known classes in validation', failed['error'])
        run = self.train(dataset_id=dataset['id'], parameters={'decision_threshold': .5})
        self.assertEqual(run['threshold'], .5)
        self.assertEqual(run['threshold_source'], 'configured')
        self.assertEqual(run['parameters']['decision_threshold'], .5)
        comparison = self.compare([run], dataset['id'])
        self.assertEqual(comparison['status'], 'succeeded', comparison.get('error'))
        self.assertEqual(comparison['result']['models'][0]['probability_threshold'], .5)

    def test_selected_numeric_only_registered_source_can_train_and_compare(self):
        from framework.pipeline_data import DatasetStore
        path = self.base / 'selected.csv'
        rows = ['TRANSACTION_ID,TX_TIME_SECONDS,CUSTOMER_ID,TERMINAL_ID,TX_AMOUNT,TX_FRAUD']
        rows += [f'row-{i},{i},1,2,{10 + 90 * (i % 2)},{i % 2}' for i in range(100)]
        path.write_text('\n'.join(rows) + '\n')
        config = self.base / 'selected.config.json'
        config.write_text(json.dumps({'loader': 'fraud_dataset', 'dataset': 'handbook',
                                     'path': str(path), 'name': 'Selected numeric', 'selection': {'start': 20}}))
        self.service.close()
        self.store = DatasetStore(self.base / 'stored', [config])
        dataset = next(row for row in self.store.catalog()['datasets'] if row['name'] == 'Selected numeric')
        self.assertNotIn('payments', dataset['views'])
        self.service = TrainingService(self.store, self.base / 'pipeline')
        run = self.train(dataset_id=dataset['id'])
        job = self.compare([run], dataset['id'])
        self.assertEqual(job['status'], 'succeeded', job.get('error'))
        self.assertEqual(job['result']['row_count'], 16)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'optional PyTorch dependency')
    def test_real_temporal_fraud_and_numeric_models_share_one_probability_comparison(self):
        self.service.close()
        self.store = PaymentsStore(self.base)
        self.service = TrainingService(self.store, self.base / 'pipeline')
        logistic = self.train()
        temporal = self.train(model_id='dyg_tami_native', parameters={
            'epochs': 1, 'patience': 1, 'batch_size': 16, 'time_feat_dim': 4,
            'channel_embedding_dim': 4, 'num_heads': 1, 'dropout': 0.,
            'max_input_sequence_length': 8, 'decision_threshold': .25})
        self.assertAlmostEqual(temporal['threshold'], -math.log2(.75))
        self.assertEqual(temporal['threshold_source'], 'configured')
        self.assertEqual(temporal['partition_counts'], {'train': 60, 'model_validation': 10,
                                                       'policy_validation': 10, 'test': 20})
        self.assertEqual(len(self.service.runs()), 2)
        job = self.compare([logistic, temporal])
        self.assertEqual(job['status'], 'succeeded', job.get('error'))
        result = job['result']
        self.assertEqual(result['row_count'], 20)
        self.assertEqual(result['evaluation_bounds'], [40 * 60, 49 * 60])
        for row in result['rows']:
            self.assertEqual(set(row['predictions']), {logistic['id'], temporal['id']})
            for prediction in row['predictions'].values():
                self.assertGreaterEqual(prediction['probability'], 0)
                self.assertLessEqual(prediction['probability'], 1)
            self.assertEqual(row['predictions'][temporal['id']]['score_units'], 'fraud-surprise-bits')
        self.assertTrue(self.service.job(temporal['job_id'])['logs'])
        self.store.add('fresh', offset=1000)
        fresh = self.compare([logistic, temporal], 'fresh', 'all')
        self.assertEqual(fresh['status'], 'succeeded', fresh.get('error'))
        self.assertEqual(fresh['result']['row_count'], 100)


if __name__ == '__main__':
    unittest.main()
