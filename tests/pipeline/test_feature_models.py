"""Shared stored features reach model fits, context, and held-out comparisons."""
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from datasets.stream import prepare_source
from framework.pipeline_service import PipelineService
from framework.registry import load_dataset
from models.implementations import tabfm


class FeatureModelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.pipeline = PipelineService(self.base / 'workspace')
        self.addCleanup(lambda: self.pipeline.close())

    def wait(self, job):
        self.pipeline.training._futures[job['id']].result(timeout=90)
        result = self.pipeline.training.job(job['id'])
        self.assertEqual(result['status'], 'succeeded', result.get('error'))
        return result['result']

    def variant(self, identifier, features):
        status, result = self.pipeline.post('/api/pipeline/datasets/' + identifier + '/features',
                                            {'features': features, 'name': 'Shared features'})
        self.assertEqual(status, 201)
        return result['dataset']

    def test_saved_columns_reach_real_models_and_tabfm_context(self):
        source = self.pipeline.store.generate({'generator': 'handbook_generator',
                    'parameters': {'transactions': 160, 'fraud_rate': .3}})
        selected = ['log_amount', 'graph_sender_pagerank',
                    'graph_ppr_sender_to_recipient', 'graph_same_component']
        variant = self.variant(source['id'], selected)
        data = load_dataset(self.pipeline.store.dataset_config(variant['id'], 'numeric'))
        self.assertEqual(list(data.feature_names), selected)
        # Graph encoders consume exactly the same columns, with one padding row.
        graph = load_dataset(self.pipeline.store.dataset_config(variant['id'], 'labeled_graph')).graph
        self.assertEqual(list(graph.edge_feature_names), selected)
        np.testing.assert_allclose(graph.edge_features[1:], data.features, rtol=1e-6)

        versions = dict.fromkeys(('tabfm', 'torch', 'numpy', 'scikit-learn', 'scipy'), 'test')

        class Classifier:
            classes_ = np.array([0, 1])

            def predict_proba(self, x):
                probability = np.where(x[:, 0] > .5, .8, .2)
                return np.column_stack((1 - probability, probability))

        def prepare(model, context_x, context_y, parameters, expected_versions=None):
            self.assertEqual(context_x.shape[1], len(selected))
            return Classifier(), versions

        models = [('logistic_regression', {})]
        if importlib.util.find_spec('xgboost'):
            models.append(('xgboost_native', {'num_boost_round': 3}))
        models.append(('tabfm', {'max_context_rows': 20}))
        if importlib.util.find_spec('torch'):
            models.append(('dyg_tami_native', {'epochs': 1, 'patience': 1,
                'batch_size': 16, 'time_feat_dim': 4, 'channel_embedding_dim': 4,
                'num_heads': 1, 'max_input_sequence_length': 8}))
        runs = []
        with patch.object(tabfm, 'availability_error', return_value=None), \
                patch.object(tabfm.Model, '_prepare', prepare):
            for model, parameters in models:
                with self.subTest(model=model):
                    result = self.wait(self.pipeline.training.start_training({
                        'dataset_id': variant['id'], 'model_id': model,
                        'parameters': {**parameters, 'decision_threshold': .5}}))
                    run = result['run']
                    self.assertEqual(run['feature_names'], selected)
                    runs.append(run)
                    if model == 'tabfm':
                        folder = self.pipeline.root / 'runs' / run['id'] / 'artifact'
                        state = json.loads((folder / 'model.json').read_text())
                        manifest = json.loads((folder / 'manifest.json').read_text())
                        ids = manifest['split']['train']
                        context_ids = [ids[index] for index in state['context_indices']]
                        expected = data.features[[data.ids.index(identifier) for identifier in context_ids]]
                        np.testing.assert_array_equal(state['context_x'], expected)
            self.pipeline.close()
            self.pipeline = PipelineService(self.base / 'workspace')
            report = self.wait(self.pipeline.training.start_comparison({
                'dataset_id': variant['id'], 'run_ids': [run['id'] for run in runs],
                'partition': 'test'}))
            self.assertEqual(report['row_count'], 32)
            self.assertEqual(len(report['models']), len(models))
            self.assertEqual(report['training_rows_in_evaluation'], 0)

    def test_numeric_only_variant_retains_source_identity_and_label_cutoffs(self):
        csv = self.base / 'source.csv'
        csv.write_text('TRANSACTION_ID,TX_TIME_SECONDS,CUSTOMER_ID,TERMINAL_ID,TX_AMOUNT,TX_FRAUD\n' +
            '\n'.join(f'p-{i},{i},1,2,{10 + 90 * (i % 2)},{i % 2}' for i in range(100)) + '\n')
        prepared = self.base / 'source.sqlite'
        with prepare_source({'dataset': 'handbook', 'path': str(csv)}, self.base, prepared):
            pass
        with sqlite3.connect(prepared) as database:
            database.execute("UPDATE outcomes SET available_at=1000 WHERE event_id IN ('p-1','p-61')")
        config = self.base / 'source.config.json'
        config.write_text(json.dumps({'loader': 'prepared_fraud', 'path': str(prepared),
                                      'name': 'Numeric history'}))
        self.pipeline.close()
        self.pipeline = PipelineService(self.base / 'workspace', [config])
        source = next(row for row in self.pipeline.store.catalog()['datasets'] if row['name'] == 'Numeric history')
        self.assertNotIn('payments', source['views'])
        variant = self.variant(source['id'], ['log_amount', 'sender_out_count'])
        left = self.pipeline.training._snapshot(source['id'])
        right = self.pipeline.training._snapshot(variant['id'])
        self.assertEqual(left['observations'], right['observations'])
        self.assertEqual(right['label_available_at'], {'p-1': 1000., 'p-61': 1000.})
        run = self.wait(self.pipeline.training.start_training({'dataset_id': variant['id'],
                        'model_id': 'logistic_regression', 'parameters': {'decision_threshold': .5}}))['run']
        self.assertEqual(run['label_policy']['masked'], {'train': 1, 'validation': 1})
        second = self.variant(variant['id'], ['sender_out_count'])
        snapshot = self.pipeline.training._snapshot(second['id'])
        record = self.pipeline.training._ready(run['id'])
        with self.assertRaisesRegex(ValueError, 'held-out test'):
            self.pipeline.training._comparison_population({'partition': 'all'}, snapshot, [record])


if __name__ == '__main__':
    unittest.main()
