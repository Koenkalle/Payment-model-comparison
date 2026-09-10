"""Native model tests: real training, chronological state, data isolation and artifacts."""
import copy
import csv
import dataclasses
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from framework.registry import load_dataset, create_model
from framework.experiments import chronological_split, train_experiment, evaluate_artifact
from framework.temporal_sampling import DestinationSampler, timestamp_groups

TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None
PARAMETERS = dict(epochs=2, patience=2, batch_size=8, learning_rate=0.003,
                  time_feat_dim=4, channel_embedding_dim=4,
                  num_layers=1, patch_size=2, max_input_sequence_length=8,
                  dropout=0.0, seed=31)


@unittest.skipUnless(TORCH_AVAILABLE, 'optional PyTorch dependency')
class TemporalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        with (self.base / 'links.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['id', 'time', 'source', 'destination', 'amount', 'fraud'])
            # Two simultaneous independent interactions, repeated pairs, and a
            # recipient first appearing in the held-out test interval.
            for i in range(80):
                writer.writerow([str(i), i // 2, 's' + str(i % 2),
                                 'd' + str((i % 2 if i % 7 else (i + 1) % 3) if i < 70 else 3),
                                 1 + i % 5, i % 2])
        self.config = dict(loader='temporal_csv', path='links.csv',
                           columns=dict(id='id', time='time', source='source', destination='destination', label='fraud'),
                           features=['amount'], node_feature_dim=4)
        self.dataset = load_dataset(self.config, self.base)
        self.parts = chronological_split(self.dataset, {})

    def tearDown(self):
        self.temp.cleanup()

    def model(self):
        model, _ = create_model('dyg_tami_native', self.dataset.schema)
        model._initialize(self.dataset, PARAMETERS)
        return model

    def test_native_encoder_and_decoder_train_and_save_load(self):
        import torch
        model = self.model()
        initial = {key: value.detach().clone() for key, value in model.network.state_dict().items()}
        model.fit_graph(self.dataset, self.parts['train'], self.parts['validation'], PARAMETERS)
        final = model.network.state_dict()
        for prefix in ('0.time_encoder.', '0.neighbor_co_occurrence_encoder.', '0.projection_layer.', '0.transformers.', '0.output_layer.', '1.fc1.', '1.fc2.'):
            self.assertTrue(any(not torch.equal(initial[key], final[key]) for key in final if key.startswith(prefix)), prefix)
        expected = model.predict_graph(self.dataset, self.parts['test'])
        memory = {key: value.clone() for key, value in model.memory.most_recent_hist_emb.items()}
        model.save(self.base / 'model.npz')
        restored = self.model()
        restored.load_graph(self.base / 'model.npz', self.dataset)
        self.assertEqual(set(memory), set(restored.memory.most_recent_hist_emb))
        for key in memory:
            torch.testing.assert_close(memory[key], restored.memory.most_recent_hist_emb[key])
        actual = restored.predict_graph(self.dataset, self.parts['test'])
        np.testing.assert_allclose(expected['positive_logits'], actual['positive_logits'], rtol=0, atol=0)
        # Evaluating a later partition cannot contaminate an earlier one.
        first = restored.predict_graph(self.dataset, self.parts['validation'])
        restored.predict_graph(self.dataset, self.parts['test'])
        again = restored.predict_graph(self.dataset, self.parts['validation'])
        np.testing.assert_array_equal(first['positive_logits'], again['positive_logits'])

    def test_causality_unknown_nodes_outcome_and_future_isolation(self):
        from models.implementations.temporal_history import NeighborSampler
        model = self.model()
        past = np.arange(16, 32)
        original = model.predict_graph(self.dataset, past)
        # Future edge attributes and endpoint changes cannot affect earlier predictions.
        features = self.dataset.edge_features.copy()
        features[50:] *= 1000
        changed_sources = self.dataset.sources.copy()
        changed_sources[50:] = self.dataset.destinations[50:]
        changed = dataclasses.replace(self.dataset, edge_features=features, sources=changed_sources)
        altered = model.predict_graph(changed, past)
        np.testing.assert_array_equal(original['positive_logits'], altered['positive_logits'])
        # No incident at the query timestamp is returned, including tied events.
        sampler = NeighborSampler(self.dataset)
        for node in range(1, len(self.dataset.node_ids)):
            _, _, times = sampler.get_all_first_hop_neighbors([node], [8.0])
            self.assertTrue(np.all(times[0] < 8.0))
        self.assertTrue(np.isfinite(model.predict_graph(self.dataset, self.parts['test'])['positive_logits']).all())
        # Changing outcome columns has no effect on temporal-link data or scores.
        path = self.base / 'links.csv'
        with path.open() as stream:
            rows = list(csv.reader(stream))
        for row in rows[1:]:
            row[-1] = str(1 - int(row[-1]))
        with path.open('w', newline='') as stream:
            csv.writer(stream).writerows(rows)
        relabeled = load_dataset(self.config, self.base)
        np.testing.assert_array_equal(original['positive_logits'], model.predict_graph(relabeled, past)['positive_logits'])

    def test_negative_sampling_and_equal_time_pair_memory(self):
        import torch
        from models.implementations.tami import HistoricalDecoder
        sampler = DestinationSampler(10)
        for group in timestamp_groups(self.dataset, np.arange(len(self.dataset.ids))):
            sampled = sampler.sample(self.dataset, group)
            positive = set(zip(self.dataset.sources[group], self.dataset.destinations[group]))
            for i, v in zip(group, sampled):
                if v >= 0:
                    self.assertNotIn((self.dataset.sources[i], v), positive)
                    self.assertNotEqual(self.dataset.sources[i], v)
        decoder = HistoricalDecoder(4, 4, 4, gamma=0.7)
        u, v = np.array([1, 1]), np.array([2, 2])
        source, destination = torch.randn(2, 4), torch.randn(2, 4)
        before, proposed = decoder.score(u, v, source, destination)
        self.assertFalse(decoder.historical_interaction_memory.most_recent_hist_emb)
        decoder.historical_interaction_memory.update_memories(zip(u, v), proposed)
        stored = decoder.historical_interaction_memory.most_recent_hist_emb[(1, 2)]
        torch.testing.assert_close(stored, proposed.detach().mean(0))
        self.assertFalse(stored.requires_grad)
        self.assertEqual(decoder.historical_interaction_memory.get_memories([(2, 1)])[0].abs().sum(), 0)
        decoder.historical_interaction_memory.reset_memory()
        decoder.historical_interaction_memory.update_memories(zip(u[::-1], v[::-1]), proposed.flip(0))
        torch.testing.assert_close(stored, decoder.historical_interaction_memory.most_recent_hist_emb[(1, 2)])

    def test_experiment_artifact_and_schema_validation(self):
        config = dict(model='dyg_tami_native', dataset=self.config, parameters=PARAMETERS)
        artifact = self.base / 'run'
        metadata, expected = train_experiment(config, artifact, self.base)
        self.assertEqual(metadata['task'], 'dynamic-link-prediction')
        self.assertEqual(expected['schema'], 'temporal-link-evaluation/v1')
        actual = evaluate_artifact(artifact, self.config, self.base, 'test')
        self.assertEqual(expected['rows'], actual['rows'])
        self.assertEqual(expected['metrics'], actual['metrics'])
        self.assertNotIn('fraud', json.dumps(actual['metrics']))
        with self.assertRaisesRegex(ValueError, 'feature'):
            evaluate_artifact(artifact, {**self.config, 'node_feature_dim': 8}, self.base)
        with self.assertRaisesRegex(ValueError, 'original graph'):
            evaluate_artifact(artifact, {**self.config, 'time_unit': 'hours'}, self.base, 'test')
        with self.assertRaisesRegex(ValueError, 'already exists'):
            train_experiment(config, artifact, self.base)
        with (artifact / 'model.npz').open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            evaluate_artifact(artifact, self.config, self.base)

    def test_dyglib_indexed_features_and_invalid_data(self):
        path = self.base / 'native.csv'
        path.write_text('idx,u,i,ts\n2,3,5,20\n1,1,3,10\n')
        nodes = np.arange(24).reshape(6, 4).astype(float)
        edges = np.arange(6).reshape(3, 2).astype(float)
        np.save(self.base / 'nodes.npy', nodes)
        np.save(self.base / 'edges.npy', edges)
        config = dict(loader='temporal_csv', path='native.csv', node_features_path='nodes.npy', edge_features_path='edges.npy')
        data = load_dataset(config, self.base)
        self.assertEqual(data.ids, ('1', '2'))
        np.testing.assert_array_equal(data.edge_features[1:], edges[[1, 2]])
        np.testing.assert_array_equal(data.node_features[data.sources[0]], nodes[1])
        self.assertEqual(data.node_features[0].sum(), 0)
        nodes[1, 0] = np.nan
        np.save(self.base / 'nodes.npy', nodes)
        with self.assertRaisesRegex(ValueError, 'finite'):
            load_dataset(config, self.base)

    def test_test_interval_never_changes_fitted_weights(self):
        import torch
        parameters = {**PARAMETERS, 'epochs': 1}
        original = self.model()
        original.fit_graph(self.dataset, self.parts['train'], self.parts['validation'], parameters)
        features = self.dataset.edge_features.copy()
        features[self.parts['test'] + 1] = 12345
        altered = dataclasses.replace(self.dataset, edge_features=features)
        second = self.model()
        second.fit_graph(altered, self.parts['train'], self.parts['validation'], parameters)
        self.assertEqual(original.training_history, second.training_history)
        for name, parameter in original.network.state_dict().items():
            torch.testing.assert_close(parameter, second.network.state_dict()[name], rtol=0, atol=0)

    def test_payment_graph_relative_configs_identify_same_dataset(self):
        from framework.temporal_experiments import fingerprint
        document = {'accounts': [{'id': 0}, {'id': 1}],
                    'events': [{'id': 'p', 'kind': 'payment', 'u': 0, 'v': 1, 't': 0, 'amount': 1}],
                    'truth': {'p': True}}
        (self.base / 'payments.json').write_text(json.dumps(document))
        relative = dict(loader='payment_graph', dataset=dict(loader='payment_json', path='payments.json'))
        absolute = dict(loader='payment_graph', dataset=dict(loader='payment_json', path=str(self.base / 'payments.json')))
        self.assertEqual(fingerprint(load_dataset(relative, self.base)), fingerprint(load_dataset(absolute, ROOT)))


if __name__ == '__main__':
    unittest.main()
