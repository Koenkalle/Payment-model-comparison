"""Graph features use only observable prefix topology in every dataset view."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from datasets import features
from datasets.features import prepare_features
from datasets.stream import prepare_source
from framework.pipeline_data import DatasetStore
from framework.registry import load_dataset


def payment(identifier, time, source, destination):
    return {'id': identifier, 'kind': 'payment', 't': time,
            'u': source, 'v': destination, 'amount': 10.}


class GraphFeatureIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)

    def document(self, events, truth=None):
        return {'accounts': [{'id': index} for index in range(8)],
                'events': events, 'truth': truth or {}}

    def prepare(self, document):
        path = self.base / 'payments.json'
        path.write_text(json.dumps(document))
        return prepare_features({'loader': 'payment_json', 'path': str(path)}, self.base)

    def graph_values(self, data):
        indices = [index for index, name in enumerate(data.feature_names) if name.startswith('graph_')]
        self.assertTrue(indices)
        return data.features[:, indices]

    def value(self, data, identifier, name):
        return data.features[data.ids.index(identifier), data.feature_names.index(name)]

    def test_candidates_ties_future_vertices_and_outcomes_cannot_enter_history(self):
        events = [payment('first', 0, 0, 1), payment('second', 1, 1, 2),
                  payment('third', 2, 2, 0), payment('tie-a', 3, 0, 3),
                  payment('tie-b', 3, 3, 4), payment('after', 4, 4, 0)]
        document = self.document(events, {'first': False, 'tie-a': True})
        baseline, definitions = self.prepare(document)
        self.assertEqual(self.value(baseline, 'first', 'graph_sender_pagerank'), 0)
        self.assertEqual(self.value(baseline, 'tie-a', 'graph_recipient_degree'), 0)
        self.assertEqual(self.value(baseline, 'tie-b', 'graph_sender_degree'), 0)
        self.assertEqual(self.value(baseline, 'tie-b', 'graph_same_component'), 0)
        self.assertEqual(self.value(baseline, 'after', 'graph_same_component'), 1)
        changed = copy.deepcopy(document)
        changed['truth'] = {'first': True, 'tie-a': False}
        for event in changed['events']:
            event['amount'] *= 1000
            event['settled'] = False
        changed['events'].insert(1, {'id': 'deposit', 'kind': 'deposit', 't': .1,
                                     'u': -1, 'v': 5, 'amount': 800})
        changed['events'].insert(2, {'id': 'report', 'kind': 'report', 't': .2,
                                     'u': -1, 'v': 5, 'amount': 0})
        changed['events'].append(payment('future', 10, 6, 7))
        changed['accounts'].extend([{'id': 8}, {'id': 9}])
        updated, _ = self.prepare(changed)
        np.testing.assert_array_equal(self.graph_values(baseline), self.graph_values(updated)[:len(baseline.ids)])
        permuted = copy.deepcopy(document)
        permuted['events'][3:5] = reversed(permuted['events'][3:5])
        reordered, _ = self.prepare(permuted)
        for index, identifier in enumerate(baseline.ids):
            np.testing.assert_array_equal(self.graph_values(baseline)[index],
                                          self.graph_values(reordered)[reordered.ids.index(identifier)])
        self.assertTrue(all(definition['available'] for definition in definitions if definition['group'] == 'graph'))
        self.assertFalse(baseline.features.flags.writeable)

    def test_bipartite_ppr_finds_three_hop_relation_before_first_direct_payment(self):
        # C0--T0--C1--T1 creates a path without C0 ever paying T1.
        source = self.base / 'source.csv'
        source.write_text('TRANSACTION_ID,TX_TIME_SECONDS,CUSTOMER_ID,TERMINAL_ID,TX_AMOUNT,TX_FRAUD\n'
                          'e0,0,0,0,10,0\ne1,1,1,0,20,0\ne2,2,1,1,30,1\nquery,3,0,1,40,0\n')
        prepared = self.base / 'prepared.sqlite'
        with prepare_source({'dataset': 'handbook', 'path': str(source)}, self.base, prepared):
            pass
        data, definitions = prepare_features({'loader': 'prepared_fraud', 'path': str(prepared),
                                              'selection': {'start': 3}}, self.base)
        self.assertEqual(data.ids, ('query',))
        self.assertEqual(self.value(data, 'query', 'prior_contact'), 0)
        self.assertGreater(self.value(data, 'query', 'graph_ppr_sender_to_recipient'), 0)
        self.assertGreater(self.value(data, 'query', 'graph_ppr_recipient_to_sender'), 0)
        self.assertEqual(self.value(data, 'query', 'graph_same_component'), 1)
        self.assertAlmostEqual(self.value(data, 'query', 'graph_sender_component_size'), np.log1p(4))
        for name in ('graph_ppr_sender_to_recipient', 'graph_ppr_recipient_to_sender'):
            definition = next(item for item in definitions if item['id'] == name)
            self.assertIn('undirected', definition['orientation'].lower())

    def test_saved_graph_columns_stats_and_model_views_are_reused_without_recalculation(self):
        store = DatasetStore(self.base / 'store')
        source = store.generate({'generator': 'handbook_generator', 'parameters': {
            'transactions': 80, 'customer_count': 4, 'terminal_count': 3}})
        names = ['graph_ppr_sender_to_recipient', 'graph_recipient_pagerank',
                 'graph_ppr_forward_error_bound', 'amount']
        catalog = store.features(source['id'])
        selected = store.save_features(source['id'], {'features': names})
        numeric = load_dataset(store.dataset_config(selected['id']))
        self.assertEqual(list(numeric.feature_names), names)
        self.assertEqual(selected['feature_recipe_version'], 'payment-features/v2')
        for name in names:
            definition = next(item for item in catalog['features'] if item['id'] == name)
            values = numeric.features[:, numeric.feature_names.index(name)]
            self.assertAlmostEqual(definition['statistics']['mean'], float(values.mean()))
        with patch.object(features, 'prepare_features', side_effect=AssertionError('Saved feature values only')):
            reopened = DatasetStore(self.base / 'store')
            self.assertEqual(reopened.features(selected['id'])['enabled_features'], names)
            graph = load_dataset(reopened.dataset_config(selected['id'], 'labeled_graph')).graph
            self.assertEqual(list(graph.edge_feature_names), names)
            np.testing.assert_array_equal(graph.edge_features[1:], numeric.features.astype(np.float32))
            child = reopened.save_features(selected['id'], {'features': [names[0]]})
            self.assertEqual(load_dataset(reopened.dataset_config(child['id'])).feature_names, (names[0],))

    def test_numeric_only_sources_skip_graph_computation_and_explain_unavailability(self):
        header = ['Time', 'Amount', *[f'V{index}' for index in range(1, 29)], 'Class']
        store = DatasetStore(self.base / 'store')
        source = store.import_source({'source': 'ulb', 'csv': ','.join(header) + '\n' +
                                      ','.join(['0', '10', *['0'] * 28, '1']) + '\n'})
        with patch.object(features, 'GraphFeatureState', side_effect=AssertionError('No graph work without identities')):
            catalog = store.features(source['id'])
        graph = [entry for entry in catalog['features'] if entry['group'] == 'graph']
        self.assertTrue(graph)
        for entry in graph:
            self.assertFalse(entry['available'])
            self.assertIn('identities', entry['unavailable_reason'])
            self.assertIsNone(entry['statistics'])
        with self.assertRaisesRegex(ValueError, 'available observable'):
            store.save_features(source['id'], {'features': ['graph_sender_pagerank']})

    def test_existing_v1_variants_stay_pinned_and_original_source_discovers_graph_recipes(self):
        store = DatasetStore(self.base / 'store')
        source = store.catalog()['datasets'][0]
        old_recipes = [recipe for recipe in features.FEATURE_RECIPES if recipe.definition['group'] != 'graph']
        with patch.object(features, 'RECIPE_VERSION', 'payment-features/v1'), \
                patch.object(features, 'FEATURE_RECIPES', old_recipes):
            old = store.save_features(source['id'], {'features': ['amount', 'sender_out_count']})
        before = load_dataset(store.dataset_config(old['id'])).features.copy()
        pinned = store.features(old['id'])
        self.assertEqual(pinned['recipe_version'], 'payment-features/v1')
        self.assertEqual(pinned['latest_recipe_version'], 'payment-features/v2')
        self.assertFalse(any(item['group'] == 'graph' for item in pinned['features']))
        current = store.features(source['id'])
        self.assertEqual(current['recipe_version'], 'payment-features/v2')
        self.assertTrue(any(item['id'] == 'graph_recipient_pagerank' for item in current['features']))
        np.testing.assert_array_equal(load_dataset(store.dataset_config(old['id'])).features, before)


if __name__ == '__main__':
    unittest.main()
