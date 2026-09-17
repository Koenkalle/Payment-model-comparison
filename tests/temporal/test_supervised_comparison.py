"""Real fraud-trained Torch artifacts use the ordinary comparison flow."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(TORCH_AVAILABLE, 'optional PyTorch dependency')
class SupervisedComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from framework.experiments import train_experiment
        from framework.native_comparison import NativeComparison
        cls.temporary = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temporary.name)
        cls.document = {
            'accounts': [{'id': i, 'external_id': 'training-' + str(i)} for i in range(5)],
            'events': [], 'truth': {},
        }
        for i in range(80):
            cls.document['events'].append({
                'id': 'payment-' + str(i), 'kind': 'payment', 'u': i % 2,
                'v': 2 + (i % 2 if i % 7 else 2), 't': 10 + i // 2,
                'amount': 100 if i % 4 == 1 else 10 + i % 5,
            })
            cls.document['truth']['payment-' + str(i)] = i % 4 == 1
        # This policy-window fraud is confirmed only after every saved window.
        # It stays graph context but cannot tune the historical policy cutoff.
        cls.document['label_available_at'] = {'payment-61': 1000}
        (cls.base / 'payments.json').write_text(json.dumps(cls.document))
        parameters = dict(epochs=1, patience=1, batch_size=8, learning_rate=.003,
                          time_feat_dim=4, channel_embedding_dim=4, num_layers=1,
                          patch_size=2, max_input_sequence_length=8, dropout=0., seed=31)
        cls.config = {
            'model': 'dyg_tami_native', 'task': 'temporal-fraud-classification',
            'dataset': {'loader': 'payment_json', 'path': 'payments.json'},
            'graph': {'node_feature_dim': 4}, 'parameters': parameters,
            'initialization': {'method': 'link_pretrain_train_partition', 'parameters': {'epochs': 1}},
            'prediction_head': {'id': 'fraud_linear'}, 'encoder_training': 'finetune',
        }
        train_experiment(cls.config, cls.base / 'artifact', cls.base)
        cls.scorer = NativeComparison(cls.base / 'artifact')
        cls.target = copy.deepcopy(cls.document)
        for account in cls.target['accounts']:
            account['external_id'] = 'target-' + account['external_id']

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def options(self, **extra):
        return {'trainingMode': 'supervised', 'mode': 'shadow', 'decisionPolicy': 'manual',
                'manualTau': 1, 'warmup': 4, **extra}

    def test_saved_fraud_logits_and_scores_use_native_model(self):
        from framework.registry import create_model
        actual = self.scorer.compare(self.target, self.options())
        graph = self.scorer._graph(actual['dataset'])
        model, _ = create_model('dyg_tami_native', 'labeled-temporal-graph/v1',
                                 task='temporal-fraud-classification')
        model.load_graph(self.base / 'artifact' / 'model.npz', graph)
        expected = model.predict_fraud(graph, np.arange(len(graph.ids)))
        np.testing.assert_array_equal([row['fraud_logit'] for row in actual['predictions']], expected['fraud_logits'])
        np.testing.assert_allclose([row['score'] for row in actual['predictions']], expected['scores'])
        self.assertEqual(actual['head'], 'fraud_linear')
        self.assertEqual(actual['model']['training']['encoder_training'], 'finetune')
        self.assertEqual(actual['calibration']['partition'], 'policy_validation')
        self.assertEqual(self.scorer.frontiers['fraud_linear']['unknown'], 1)
        self.assertEqual(self.scorer.frontiers['fraud_linear']['positives'], 1)
        for row in actual['predictions']:
            self.assertEqual(row['evidence']['kind'], 'native-fraud')
            self.assertEqual(row['evidence']['version'], 1)
            self.assertNotIn('tail_probability', row['evidence'])
            self.assertEqual(row['decision'] == 'BLOCK', row['evidence']['fraud_probability'] > .5)

    def test_identity_describes_loaded_checkpoint_even_if_disk_changes(self):
        path = self.base / 'artifact' / 'manifest.json'
        saved = path.read_text()
        metadata = json.loads(saved)
        metadata['model_sha256'] = 'replaced-on-disk'
        metadata['best_epoch'] = 999
        try:
            path.write_text(json.dumps(metadata))
            identity = self.scorer.describe()['artifact']
            self.assertEqual(identity['model_sha256'], json.loads(saved)['model_sha256'])
            self.assertEqual(identity['path'], str(self.base / 'artifact'))
            self.assertEqual(identity['training_mode'], 'supervised')
            self.assertEqual(identity['best_epoch'], 1)
            self.assertEqual(identity['epochs_completed'], 1)
            self.assertEqual(identity['training_dataset'], 'payments.json')
        finally:
            path.write_text(saved)

    def test_target_labels_do_not_change_weights_scores_or_any_policy(self):
        changed = copy.deepcopy(self.target)
        changed['truth'] = {identifier: not label for identifier, label in changed['truth'].items()}
        for mode in ('shadow', 'enforce'):
            for policy in ('shared', 'manual', 'tuned', 'auto'):
                with self.subTest(mode=mode, policy=policy):
                    options = self.options(mode=mode, decisionPolicy=policy)
                    actual = self.scorer.compare(self.target, options)
                    relabeled = self.scorer.compare(changed, options)
                    self.assertEqual(actual['predictions'], relabeled['predictions'])
                    self.assertEqual(actual['policy_state'], relabeled['policy_state'])
                    self.assertEqual(actual['heads'], relabeled['heads'])
        with self.assertRaisesRegex(ValueError, 'supervised scoring only'):
            self.scorer.compare(self.target, self.options(trainingMode='unsupervised'))

    def test_current_amount_affects_own_score_and_blocks_remove_future_context(self):
        changed = copy.deepcopy(self.target)
        changed['events'][0]['amount'] *= 100
        options = self.options(mode='enforce', manualTau=-1)
        first = self.scorer.compare(self.target, options)
        second = self.scorer.compare(changed, options)
        self.assertTrue(all(not row['settled'] for row in first['predictions']))
        self.assertNotEqual(first['predictions'][0]['logit'], second['predictions'][0]['logit'])
        self.assertEqual(first['predictions'][1:], second['predictions'][1:])
        shadow = self.scorer.compare(self.target, self.options(manualTau=100))
        allow_all = self.scorer.compare(self.target, self.options(mode='enforce', manualTau=100))
        self.assertEqual([row['logit'] for row in shadow['predictions']],
                         [row['logit'] for row in allow_all['predictions']])
        self.assertNotEqual([row['logit'] for row in first['predictions']],
                            [row['logit'] for row in shadow['predictions']])
        future = copy.deepcopy(self.target)
        future['events'][-1]['amount'] *= 100
        future_result = self.scorer.compare(future, self.options(manualTau=100))
        self.assertEqual(shadow['predictions'][:-1], future_result['predictions'][:-1])

    def test_original_history_selects_only_test_and_reserved_subsets_are_rejected(self):
        saved_test = self.scorer.metadata['split']['test']
        for policy in ('manual', 'tuned', 'auto'):
            actual = self.scorer.compare(self.document, self.options(decisionPolicy=policy))
            self.assertEqual(actual['evaluation_ids'], saved_test)
            self.assertEqual([row['id'] for row in actual['predictions'] if row['decision'] != 'CONTEXT'], saved_test)
        reserved_id = self.scorer.metadata['split']['policy_validation'][0]
        subset = copy.deepcopy(self.document)
        subset['events'] = [row for row in subset['events'] if row['id'] == reserved_id]
        subset['events'][0]['id'] = 'copied-payment'
        subset['truth'] = {}
        subset['label_available_at'] = {}
        with self.assertRaisesRegex(ValueError, 'overlaps historical'):
            self.scorer.compare(subset, self.options())


if __name__ == '__main__':
    unittest.main()
