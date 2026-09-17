"""Causality, source preservation, extension and checkpoint regressions."""
import csv
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from datasets import features
from datasets.features import FeatureEvent, FeatureRecipe, PaymentHistory, prepare_features
from datasets.stream import prepare_source


ROOT = Path(__file__).resolve().parents[2]


def payment(identifier, time, source=0, destination=1, amount=100, **extra):
    return {'id': identifier, 'kind': 'payment', 't': time, 'u': source,
            'v': destination, 'amount': amount, **extra}


class FeatureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)

    def payments(self, events, truth=None):
        path = self.base / 'payments.json'
        path.write_text(json.dumps({'accounts': [{'id': 0}, {'id': 1}, {'id': 2}],
                                    'events': events, 'truth': truth or {}}))
        return {'loader': 'payment_json', 'path': str(path)}

    def column(self, dataset, name):
        return dataset.features[:, dataset.feature_names.index(name)]

    def prepared(self, dataset, fields, rows, **config):
        source = self.base / 'source.csv'
        with source.open('w', newline='') as output:
            writer = csv.writer(output)
            writer.writerow(fields)
            writer.writerows(rows)
        path = self.base / 'prepared.sqlite'
        with prepare_source({'dataset': dataset, 'path': str(source)}, self.base, path):
            pass
        return {'loader': 'prepared_fraud', 'path': str(path), **config}

    def test_timestamp_ties_are_simultaneous_and_window_boundary_is_inclusive(self):
        config = self.payments([payment('first', 0, settled=False),
                                payment('tie', 0, amount=200),
                                payment('boundary', 60),
                                payment('after', 60.001)])
        dataset, definitions = prepare_features(config, self.base)
        self.assertEqual(len([item for item in definitions
                              if item['kind'] == 'derived' and item['group'] != 'graph']), 27)
        self.assertEqual(dataset.ids, ('first', 'tie', 'boundary', 'after'))
        np.testing.assert_array_equal(dataset.times, [0, 0, 3600, 3600.06])
        np.testing.assert_allclose(self.column(dataset, 'sender_out_count'),
                                   [0, 0, math.log1p(2) / 6, math.log1p(3) / 6])
        np.testing.assert_allclose(self.column(dataset, 'sender_recent_out_count'),
                                   [0, 0, math.log1p(2) / 4, math.log1p(1) / 4])
        np.testing.assert_allclose(self.column(dataset, 'sender_recent_out_value'),
                                   [0, 0, math.log1p(300) / 10, math.log1p(100) / 10])
        np.testing.assert_array_equal(self.column(dataset, 'prior_contact'), [0, 0, 1, 1])

    def test_outcomes_settlement_and_reports_do_not_change_features(self):
        events = [payment('first', 0, settled=False),
                  {'id': 'deposit', 'kind': 'deposit', 't': 10, 'u': -1, 'v': 0, 'amount': 800},
                  {'id': 'report', 'kind': 'report', 't': 15, 'u': -1, 'v': 0, 'amount': 0},
                  payment('candidate', 20)]
        original, _ = prepare_features(self.payments(events, {'first': False, 'candidate': True}), self.base)
        changed = [{**event, 'settled': True} if event['kind'] == 'payment' else event
                   for event in events if event['kind'] != 'report']
        updated, _ = prepare_features(self.payments(changed, {'first': True}), self.base)
        np.testing.assert_array_equal(original.features, updated.features)
        np.testing.assert_array_equal(original.labels, [0, 1])
        np.testing.assert_array_equal(updated.labels, [1, -1])
        self.assertAlmostEqual(self.column(original, 'sender_seen')[1], math.log1p(2) / 6)
        self.assertAlmostEqual(self.column(original, 'sender_gap')[1], math.log1p(10) / 6)
        self.assertAlmostEqual(self.column(original, 'sender_out_value')[1], math.log1p(100) / 10)
        self.assertFalse(original.features.flags.writeable)

    def test_appending_future_events_cannot_change_existing_rows(self):
        events = [payment('first', 1), payment('second', 2, amount=150)]
        before, _ = prepare_features(self.payments(events), self.base)
        after, _ = prepare_features(self.payments([*events, payment('future', 200, amount=1e8)]), self.base)
        np.testing.assert_array_equal(before.features, after.features[:2])

    def test_prepared_source_columns_and_units_survive_configured_subset(self):
        config = self.prepared('paysim',
                               ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'nameDest', 'oldbalanceDest'],
                               [[1, 'TRANSFER', 35, 'a', 100, 'b', 50],
                                [2, 'CASH_OUT', 50, 'a', 65, 'b', 85]],
                               features=['amount'], amount_to_eur=10)
        dataset, definitions = prepare_features(config, self.base)
        self.assertEqual(dataset.feature_names[:3], ('amount', 'oldbalanceOrg', 'oldbalanceDest'))
        np.testing.assert_array_equal(dataset.features[:, :3], [[35, 100, 50], [50, 65, 85]])
        amount_transform = next(item for item in definitions if item['id'] == 'log1p_amount')
        self.assertEqual(amount_transform['kind'], 'derived')
        self.assertNotIn('log1p_amount', [item['id'] for item in definitions if item['kind'] == 'source'])
        np.testing.assert_allclose(self.column(dataset, 'log1p_amount'), np.log1p([35, 50]))
        self.assertAlmostEqual(self.column(dataset, 'log_amount')[0], math.log1p(35) / 8)
        self.assertAlmostEqual(self.column(dataset, 'amount_bin')[0], 2 / 9)
        bucket = next(item for item in definitions if item['id'] == 'amount_bin')
        self.assertEqual(bucket['amount_unit_assumption'], 'source currency units (unspecified)')
        self.assertEqual(bucket['amount_bins'], [15, 35, 75, 150, 300, 650, 1500, 4000, 10000])
        self.assertAlmostEqual(self.column(dataset, 'sender_recent_out_count')[1], math.log1p(1) / 4)

    def test_ulb_has_amount_features_and_unavailable_relational_definitions(self):
        config = self.prepared('ulb', ['Time', 'Amount', *[f'V{i}' for i in range(1, 29)], 'Class'],
                               [[10, 8, *range(-14, 14), 1], [20, 12, *range(28), 0]],
                               features=['amount'])
        dataset, definitions = prepare_features(config, self.base)
        self.assertEqual(len(dataset.feature_names), 32)
        np.testing.assert_array_equal(dataset.features[0, 1:29], range(-14, 14))
        self.assertEqual(dataset.feature_names[-2:], ('log_amount', 'amount_bin'))
        unavailable = [item for item in definitions if not item['available']]
        self.assertEqual(len([item for item in unavailable if item['group'] != 'graph']), 25)
        self.assertTrue(any(item['group'] == 'graph' for item in unavailable))
        self.assertTrue(all('identities' in item['unavailable_reason'] for item in unavailable))
        self.assertFalse(set(item['id'] for item in unavailable) & set(dataset.feature_names))
        np.testing.assert_array_equal(dataset.labels, [1, 0])

    def test_selection_keeps_observed_history_before_its_start(self):
        config = self.prepared('handbook',
                               ['TRANSACTION_ID', 'TX_TIME_SECONDS', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT'],
                               [['before', 10, 'same', 'same', 100], ['selected', 20, 'same', 'same', 50],
                                ['future', 30, 'same', 'same', 1000]],
                               selection={'start': 20, 'stop': 30})
        dataset, _ = prepare_features(config, self.base)
        self.assertEqual(dataset.ids, ('selected',))
        self.assertAlmostEqual(self.column(dataset, 'sender_out_count')[0], math.log1p(1) / 6)
        # Equal identifiers from different entity types must remain separate.
        self.assertEqual(self.column(dataset, 'sender_in_count')[0], 0)
        self.assertEqual(self.column(dataset, 'recipient_out_count')[0], 0)

    def test_registry_extension_and_duplicate_ids(self):
        config = self.payments([payment('payment', 0, amount=40)])
        definition = {'id': 'amount_squared', 'label': 'Squared amount', 'description': 'Fixture extension.',
                      'group': 'payment', 'kind': 'derived', 'unit': 'EUR squared', 'transform': 'amount ** 2',
                      'window': 'current payment', 'requirements': ['amount'], 'version': '1'}
        recipe = FeatureRecipe(definition, lambda context: context.event.amount ** 2)
        with patch.object(features, 'FEATURE_RECIPES', [*features.FEATURE_RECIPES, recipe]):
            dataset, definitions = prepare_features(config, self.base)
        self.assertEqual(dataset.feature_names[-1], 'amount_squared')
        self.assertEqual(dataset.features[0, -1], 1600)
        self.assertTrue(definitions[-1]['available'])
        duplicate = FeatureRecipe({**definition, 'id': 'amount'}, recipe.calculate)
        with patch.object(features, 'FEATURE_RECIPES', [duplicate]), self.assertRaisesRegex(ValueError, 'unique IDs'):
            prepare_features(config, self.base)

    def test_history_memory_is_sparse_and_old_event_records_are_released(self):
        history = PaymentHistory()
        for index in range(10000):
            event = FeatureEvent(str(index), index * 60., 1., ('source', index), ('destination', index))
            history.advance(event.time)
            history.observe(event)
        self.assertEqual(len(history.accounts), 20000)
        self.assertEqual(len(history.pairs), 10000)
        self.assertEqual(len(history.recent), 61)
        history.advance(10000 * 60. + 3600)
        self.assertEqual(len(history.recent), 0)
        self.assertEqual(history.accounts[('source', 0)].out_count, 1)
        self.assertEqual(history.accounts[('source', 0)].recent_out_count, 0)

    def test_legacy_checkpoint_outputs_match_saved_parity(self):
        from models.implementations.xgboost_numpy import xgb_trace, xgb_unsupervised_trace
        model = json.loads((ROOT / 'models/xgboost.json').read_text())
        for folder, calculate in [('parity', xgb_unsupervised_trace),
                                   ('parity-supervised', lambda episode: xgb_trace(model, episode))]:
            with self.subTest(folder=folder):
                fixture = json.loads((ROOT / folder / 'xgboost.json').read_text())
                actual = calculate(fixture['episode'])
                np.testing.assert_array_equal([row['scores'] for row in actual],
                                              [row['scores'] for row in fixture['expected']])


if __name__ == '__main__':
    unittest.main()
