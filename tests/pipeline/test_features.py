"""Stored feature recipes are immutable, inspectable and shared across views."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from datasets import features as recipes
from framework.pipeline_data import DatasetStore
from framework.registry import load_dataset
from framework.temporal_fraud_data import load_labeled_dataset


CSV = ('TRANSACTION_ID,TX_TIME_SECONDS,CUSTOMER_ID,TERMINAL_ID,TX_AMOUNT,TX_FRAUD,TX_FRAUD_SCENARIO\n'
       'late,120,1,1,12,1,99\nearly,60,1,2,10,0,0\nunknown,120,2,2,8,,\n')


class StoredFeatureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.store = DatasetStore(self.base / 'store')
        self.parent = self.store.import_source({'source': 'handbook', 'csv': CSV, 'amount_to_eur': 2.})

    def tearDown(self):
        self.temp.cleanup()

    def save(self, features=None, **kwargs):
        return self.store.save_features(self.parent['id'], {
            'features': features or ['amount', 'log_amount', 'sender_out_count'], **kwargs})

    def test_catalog_describes_columns_and_exact_population_statistics(self):
        catalog = self.store.features(self.parent['id'])
        self.assertEqual(catalog['dataset_id'], self.parent['id'])
        self.assertEqual(catalog['row_count'], 3)
        self.assertEqual(catalog['enabled_features'], ['amount'])
        definitions = {definition['id']: definition for definition in catalog['features']}
        self.assertTrue(definitions['amount']['enabled'])
        self.assertFalse(definitions['sender_out_count']['enabled'])
        self.assertNotIn('TX_FRAUD', definitions)
        stats = definitions['amount']['statistics']
        self.assertEqual((stats['count'], stats['missing'], stats['unique']), (3, 0, 3))
        self.assertEqual((stats['min'], stats['max'], stats['mean']), (8, 12, 10))
        self.assertAlmostEqual(stats['std'], np.std([10, 12, 8]))
        self.assertEqual(stats['quantiles'], {'p25': 9, 'p50': 10, 'p75': 11})
        self.assertEqual(sum(bucket['count'] for bucket in stats['histogram']), 3)
        self.assertIn('strictly earlier', definitions['sender_out_count']['description'])
        self.assertNotIn(str(self.base), json.dumps(catalog, allow_nan=False))

    def test_immutable_variant_stores_ordered_columns_and_keeps_source_unchanged(self):
        before = load_dataset(self.store.dataset_config(self.parent['id']))
        variant = self.save(['sender_out_count', 'log_amount'], name='Shared inputs')
        self.assertNotEqual(variant['id'], self.parent['id'])
        self.assertEqual(variant['kind'], 'features')
        self.assertEqual(variant['name'], 'Shared inputs')
        self.assertEqual(variant['parent_dataset_id'], self.parent['id'])
        self.assertEqual(variant['source_dataset_id'], self.parent['id'])
        self.assertEqual(variant['source_namespace'], self.parent['source_namespace'])
        self.assertEqual(self.store.get(self.parent['id']), self.parent)
        numeric = load_dataset(self.store.dataset_config(variant['id']))
        self.assertEqual(numeric.feature_names, ('sender_out_count', 'log_amount'))
        self.assertEqual(numeric.ids, before.ids)
        np.testing.assert_array_equal(numeric.times, before.times)
        np.testing.assert_array_equal(numeric.labels, before.labels)
        np.testing.assert_allclose(numeric.features[:, 0], [0, np.log1p(1) / 6, 0])
        np.testing.assert_allclose(numeric.features[:, 1], np.log1p([10, 12, 8]) / 8)
        self.assertFalse(numeric.features.flags.writeable)
        self.assertEqual(self.store.sample(variant['id'])['columns'], list(numeric.feature_names))
        np.testing.assert_array_equal(load_dataset(self.store.dataset_config(self.parent['id'])).features,
                                      before.features)
        same = self.save(['sender_out_count', 'log_amount'], name='Another name')
        changed = self.save(['log_amount', 'sender_out_count'])
        self.assertEqual(same['fingerprint'], variant['fingerprint'])
        self.assertNotEqual(changed['fingerprint'], variant['fingerprint'])

    def test_graph_and_labeled_graph_reuse_stored_columns_and_keep_clocks(self):
        variant = self.save()
        numeric = load_dataset(self.store.dataset_config(variant['id']))
        with patch.object(recipes, 'prepare_features', side_effect=AssertionError('Must use stored columns')):
            for view in ('graph', 'labeled_graph'):
                original = load_dataset(self.store.dataset_config(self.parent['id'], view))
                config = self.store.dataset_config(variant['id'], view)
                selected = load_dataset(config)
                graph = selected.graph if view == 'labeled_graph' else selected
                original_graph = original.graph if view == 'labeled_graph' else original
                self.assertEqual(graph.edge_feature_names, numeric.feature_names)
                np.testing.assert_array_equal(graph.edge_features[1:], numeric.features.astype(np.float32))
                np.testing.assert_array_equal(graph.edge_features[0], np.zeros(3))
                self.assertFalse(graph.edge_features.flags.writeable)
                self.assertEqual(graph.ids, original_graph.ids)
                self.assertEqual(graph.node_ids, original_graph.node_ids)
                np.testing.assert_array_equal(graph.times, original_graph.times)
                np.testing.assert_array_equal(graph.sources, original_graph.sources)
                np.testing.assert_array_equal(graph.destinations, original_graph.destinations)
                self.assertEqual(graph.provenance['time_origin_seconds'], 60.)
                if view == 'labeled_graph':
                    np.testing.assert_array_equal(selected.labels, original.labels)
                    np.testing.assert_array_equal(selected.label_available_at, original.label_available_at)
                    compact = load_labeled_dataset(config, graph_config={'node_feature_dim': 8})
                    self.assertEqual(compact.graph.node_features.shape[1], 8)
        before = self.store.document(self.parent['id'])['dataset']
        after = self.store.document(variant['id'])['dataset']
        for key in ('accounts', 'events', 'truth', 'label_available_at', 'units'):
            self.assertEqual(after.get(key), before.get(key))
        self.assertIn('Payment replay', variant['feature_replay_note'])

    def test_cache_restart_and_variants_reenable_columns_without_recomputation(self):
        with patch.object(recipes, 'prepare_features', wraps=recipes.prepare_features) as calculate:
            self.store.features(self.parent['id'])
            variant = self.save(['sender_out_count'])
            self.store.features(self.parent['id'])
            self.assertEqual(calculate.call_count, 1)
        again = DatasetStore(self.base / 'store')
        with patch.object(recipes, 'prepare_features', side_effect=AssertionError('Unexpected recalculation')):
            self.assertEqual(again.features(variant['id'])['enabled_features'], ['sender_out_count'])
            child = again.save_features(variant['id'], {'features': ['amount', 'recipient_in_count']})
            selected = load_dataset(again.dataset_config(child['id']))
        self.assertEqual(child['parent_dataset_id'], variant['id'])
        self.assertEqual(child['source_dataset_id'], self.parent['id'])
        self.assertEqual(selected.feature_names, ('amount', 'recipient_in_count'))
        self.assertEqual(load_dataset(again.source_config(child['id'])).feature_names, ('amount',))

    def test_recipe_updates_do_not_change_saved_catalog_or_values(self):
        variant = self.save()
        catalog = self.store.features(variant['id'])
        before = load_dataset(self.store.dataset_config(variant['id']))
        with patch.object(recipes, 'RECIPE_VERSION', 'future-recipes/v2'), \
                patch.object(recipes, 'prepare_features', side_effect=AssertionError('Historical recipe must remain pinned')):
            self.assertEqual(self.store.features(variant['id']), {**catalog, 'latest_recipe_version': 'future-recipes/v2'})
            child = self.store.save_features(variant['id'], {'features': variant['feature_names']})
            self.assertEqual(child['fingerprint'], variant['fingerprint'])
            self.assertEqual(child['feature_recipe_version'], variant['feature_recipe_version'])
            np.testing.assert_array_equal(load_dataset(self.store.dataset_config(child['id'])).features, before.features)

    def test_variant_contains_its_source_and_all_original_recipe_columns(self):
        variant = self.save(['log_amount'])
        shutil.rmtree(self.store._directory(self.parent['id']))
        with patch.object(recipes, 'prepare_features', side_effect=AssertionError('Self-contained saved recipe')):
            self.assertEqual(self.store.features(variant['id'])['row_count'], 3)
            child = self.store.save_features(variant['id'], {'features': ['amount', 'sender_in_count']})
            for view in ('numeric', 'graph', 'labeled_graph', 'payments'):
                load_dataset(self.store.dataset_config(child['id'], view))

    def test_unavailable_relational_features_are_explained_and_rejected(self):
        columns = ['Time', 'Amount', *[f'V{index}' for index in range(1, 29)], 'Class']
        csv = ','.join(columns) + '\n' + ','.join(['0', '20', *['.2'] * 28, '1']) + '\n'
        source = self.store.import_source({'source': 'ulb', 'csv': csv})
        catalog = self.store.features(source['id'])
        definitions = {definition['id']: definition for definition in catalog['features']}
        unavailable = definitions['sender_out_count']
        self.assertFalse(unavailable['available'])
        self.assertIn('identities', unavailable['unavailable_reason'])
        self.assertIsNone(unavailable['statistics'])
        self.assertTrue(definitions['log_amount']['available'])
        with self.assertRaisesRegex(ValueError, 'available observable'):
            self.store.save_features(source['id'], {'features': ['sender_out_count']})
        variant = self.store.save_features(source['id'], {'features': ['V3', 'log_amount']})
        self.assertEqual(variant['views'], ['numeric'])
        self.assertEqual(load_dataset(self.store.dataset_config(variant['id'])).features.shape, (1, 2))

    def test_invalid_selections_and_failed_save_publish_nothing(self):
        before = self.store.catalog()['datasets']
        for payload in ({}, {'features': []}, {'features': ['amount', 'amount']}, {'features': [True]},
                        {'features': 'amount'}, {'features': ['TX_FRAUD']}, {'features': ['../metadata.json']},
                        {'features': ['amount'], 'path': '/etc/passwd'}, {'features': ['amount'], 'name': ''}):
            with self.assertRaises(ValueError):
                self.store.save_features(self.parent['id'], payload)
        self.store.features(self.parent['id'])
        with patch('framework.pipeline_data.write_numeric', side_effect=ValueError('Failed write')):
            with self.assertRaisesRegex(ValueError, 'Failed write'):
                self.save()
        self.assertEqual(self.store.catalog()['datasets'], before)
        self.assertFalse(list(self.store.directory.glob('.creating-*')))

    def test_saved_matrix_and_feature_cache_checksums_are_verified(self):
        variant = self.save()
        path = self.store._directory(variant['id']) / 'features.npz'
        original = path.read_bytes()
        path.write_bytes(original + b'changed')
        with self.assertRaisesRegex(ValueError, 'feature payload checksum'):
            self.store.dataset_config(variant['id'])
        with self.assertRaisesRegex(ValueError, 'feature payload checksum'):
            self.store.features(variant['id'])
        path.write_bytes(original)
        cache = next(self.store._directory(variant['id']).glob('.features-*'))
        (cache / 'columns.npz').write_bytes((cache / 'columns.npz').read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'feature payload checksum'):
            self.store.features(variant['id'])
        # The already saved numeric matrix remains independent of the catalog cache.
        self.assertEqual(load_dataset(self.store.dataset_config(variant['id'])).features.shape, (3, 3))

    def test_payment_scenarios_support_same_selected_feature_contract(self):
        source = self.store.generate({'generator': 'synthetic_payments',
                                      'parameters': {'name': 'benign', 'size': 'small'}})
        variant = self.store.save_features(source['id'], {'features': ['amount', 'pair_total_count']})
        numeric = load_dataset(self.store.dataset_config(variant['id']))
        labeled = load_dataset(self.store.dataset_config(variant['id'], 'labeled_graph'))
        self.assertEqual(numeric.ids, labeled.ids)
        self.assertEqual(labeled.graph.edge_feature_names, numeric.feature_names)
        np.testing.assert_array_equal(labeled.graph.edge_features[1:], numeric.features.astype(np.float32))
        self.assertEqual(len(numeric.ids), source['payment_rows'])


if __name__ == '__main__':
    unittest.main()
