"""Stored dataset columns are native candidate inputs to the fraud head."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from datasets.implementations.temporal_csv import graph_dataset


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional PyTorch dependency")
class StoredFeatureHeadTests(unittest.TestCase):
    def setUp(self):
        from models.implementations.dyg_tami_fraud import Model

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.graph = graph_dataset(
            [str(i) for i in range(32)],
            [i // 2 for i in range(32)],
            ["source" + str(i % 2) for i in range(32)],
            ["destination" + str(i % 3) for i in range(32)],
            [[i % 4, i / 10] for i in range(32)],
            ["sender_out_count", "pair_total_count"],
            {"features_stored": True},
            4,
        )
        self.training, self.validation, self.test = (
            np.arange(20),
            np.arange(20, 26),
            np.arange(26, 32),
        )
        self.dataset = SimpleNamespace(
            graph=self.graph,
            labels=np.asarray([int(i % 4 == 0) for i in range(32)]),
            label_available_at=np.full(32, np.nan),
        )
        self.model = Model()
        self.model.fit_fraud(
            self.dataset,
            self.training,
            self.validation,
            dict(
                epochs=1,
                patience=1,
                batch_size=8,
                learning_rate=0.01,
                time_feat_dim=4,
                channel_embedding_dim=4,
                num_layers=1,
                patch_size=2,
                max_input_sequence_length=8,
                dropout=0.0,
                seed=31,
                num_threads=1,
                encoder_training="frozen",
            ),
        )

    def test_all_candidate_columns_use_training_only_normalization(self):
        self.assertNotIn("log1p_amount", self.graph.edge_feature_names)
        self.assertEqual(self.graph.edge_features.dtype, np.float32)
        normalization = self.model.feature_normalization
        values = self.graph.edge_features[self.training + 1].astype(np.float64)
        np.testing.assert_array_equal(normalization["mean"], values.mean(axis=0))
        np.testing.assert_array_equal(normalization["scale"], values.std(axis=0))
        self.assertEqual(normalization["fitted_rows"], len(self.training))
        self.assertEqual(
            self.model.head_input_schema["candidate_features"],
            list(self.graph.edge_feature_names),
        )
        self.assertEqual(
            self.model.head.descriptor()["input_contract"],
            "tami-payment-representation/v2",
        )
        # Neither padding nor the later validation/test distribution enters fit.
        self.assertNotEqual(
            normalization["mean"][1], float(self.graph.edge_features[:, 1].mean())
        )

    def test_each_selected_column_directly_changes_its_candidate_only(self):
        import torch

        group = np.array([0, 1])
        self.model._bind(self.graph)
        with torch.no_grad():
            for column in range(2):
                self.model.head.linear.weight.zero_()
                self.model.head.linear.weight[0, -2 + column] = 1.0
                baseline, _, before = self.model.score_group(self.graph, group)
                changed_values = self.graph.edge_features.copy()
                changed_values[1, column] += 5
                changed = replace(self.graph, edge_features=changed_values)
                result, _, after = self.model.score_group(changed, group)
                self.assertNotEqual(float(baseline[0]), float(result[0]))
                self.assertEqual(float(baseline[1]), float(result[1]))
                torch.testing.assert_close(before, after, rtol=0, atol=0)
                self.assertFalse(self.model.memory.most_recent_hist_emb)

    def test_v2_roundtrip_uses_saved_contract_and_rejects_corrupt_normalization(self):
        from models.implementations.dyg_tami_fraud import Model

        path = self.base / "features.npz"
        self.model.save(path)
        expected = self.model.predict_fraud(self.graph, self.test)
        restored = Model()
        # Checkpoint declares the contract even when caller provenance is absent.
        restored.load_fraud(path, replace(self.graph, provenance={}))
        for name, values in expected.items():
            np.testing.assert_array_equal(
                values, restored.predict_fraud(self.graph, self.test)[name]
            )
        with np.load(path, allow_pickle=False) as saved:
            tensors = {key: saved[key].copy() for key in saved.files}
        metadata = json.loads(str(tensors["metadata"]))
        self.assertEqual(metadata["version"], 2)
        self.assertNotIn("amount_normalization", metadata)
        self.assertEqual(
            restored.feature_normalization, metadata["feature_normalization"]
        )
        metadata["feature_normalization"]["scale"][0] = 0
        tensors["metadata"] = np.asarray(json.dumps(metadata))
        corrupted = self.base / "invalid.npz"
        np.savez(corrupted, **tensors)
        with self.assertRaisesRegex(ValueError, "feature normalization"):
            Model().load_fraud(corrupted, self.graph)
        with self.assertRaisesRegex(ValueError, "feature names or order"):
            Model().load_fraud(
                path,
                replace(
                    self.graph,
                    edge_feature_names=tuple(reversed(self.graph.edge_feature_names)),
                ),
            )


if __name__ == "__main__":
    unittest.main()
