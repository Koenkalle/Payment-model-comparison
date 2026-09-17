"""Real supervised gradients, temporal causality, and safe learned-head artifacts."""
import copy
import dataclasses
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from datasets.implementations.temporal_csv import graph_dataset
from prediction_heads.registry import manifest, create_learned_head

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
PARAMETERS = dict(
    epochs=2,
    patience=2,
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
)


@unittest.skipUnless(TORCH_AVAILABLE, "optional PyTorch dependency")
class FraudModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.graph = graph_dataset(
            [str(i) for i in range(48)],
            [i // 2 for i in range(48)],
            ["s" + str(i % 2) for i in range(48)],
            ["d" + str(i % 3) for i in range(48)],
            [[np.log1p(1 + i % 7 + 20 * (i % 4 == 0))] for i in range(48)],
            ["log1p_amount"],
            {},
            4,
        )
        labels = np.array([int(i % 4 == 0) for i in range(48)])
        labels[3] = -1
        self.dataset = SimpleNamespace(
            graph=self.graph, labels=labels, label_available_at=np.full(48, np.nan)
        )
        self.training, self.validation, self.test = (
            np.arange(28),
            np.arange(28, 36),
            np.arange(36, 48),
        )

    def tearDown(self):
        self.temp.cleanup()

    def fit(self, mode="finetune", dataset=None, pretrained=None, **extra):
        from models.implementations.dyg_tami_fraud import Model

        model = Model()
        model.fit_fraud(
            dataset or self.dataset,
            self.training,
            self.validation,
            {**PARAMETERS, "encoder_training": mode, **extra},
            pretrained_model=pretrained,
        )
        return model

    def test_real_gradients_finetune_and_frozen_encoder(self):
        import torch
        from models.implementations.dyg_tami import Model as LinkModel

        initial = LinkModel()
        initial._initialize(self.graph, PARAMETERS)
        before = {
            key: value.detach().clone()
            for key, value in initial.network.state_dict().items()
        }
        frozen = self.fit("frozen", pretrained=initial)
        tuned = self.fit("finetune", pretrained=initial)
        for key, value in before.items():
            torch.testing.assert_close(
                value, frozen.network.state_dict()[key], rtol=0, atol=0
            )
            torch.testing.assert_close(
                value, initial.network.state_dict()[key], rtol=0, atol=0
            )
        for prefix in ("0.time_encoder.", "0.transformers.", "1.fc1."):
            self.assertTrue(
                any(
                    not torch.equal(value, tuned.network.state_dict()[key])
                    for key, value in before.items()
                    if key.startswith(prefix)
                ),
                prefix,
            )
        self.assertFalse(tuned.network[1].fc2.weight.requires_grad)
        self.assertEqual(
            frozen.label_counts,
            {"legitimate": 20, "fraud": 7, "unknown_or_unavailable": 1},
        )
        self.assertTrue(
            all(row["labeled_training_rows"] == 27 for row in frozen.training_history)
        )
        self.assertFalse(
            torch.equal(frozen.head.linear.weight, tuned.head.linear.weight)
        )
        self.assertFalse(frozen.network[0].training)
        from models.implementations.dyg_tami_fraud import Model

        independent_schedule = Model()
        independent_schedule._initialize_fraud(
            self.graph, {"encoder_training": "frozen"}, initial
        )
        self.assertEqual(independent_schedule.parameters["epochs"], 10)
        self.assertEqual(independent_schedule.parameters["learning_rate"], 0.0001)
        self.assertEqual(
            Model.pretraining_parameters({"epochs": 9, "encoder_training": "frozen"}),
            {"epochs": 3},
        )
        self.assertEqual(
            Model.pretraining_parameters({"epochs": 9}, {"epochs": 5}), {"epochs": 5}
        )

    def test_training_labels_change_weights_queries_and_test_labels_do_not(self):
        import torch

        first = self.fit("frozen")
        labels = self.dataset.labels.copy()
        known = labels[self.training] >= 0
        labels[self.training[known]] = 1 - labels[self.training[known]]
        relabeled = copy.copy(self.dataset)
        relabeled.labels = labels
        second = self.fit("frozen", relabeled)
        self.assertFalse(
            torch.equal(first.head.linear.weight, second.head.linear.weight)
        )
        expected = first.predict_fraud(self.dataset, self.validation)["fraud_logits"]
        np.testing.assert_array_equal(
            expected, first.predict_fraud(relabeled, self.validation)["fraud_logits"]
        )
        heldout = copy.copy(self.dataset)
        heldout.labels = self.dataset.labels.copy()
        heldout.labels[self.test] = 1 - heldout.labels[self.test]
        heldout.graph = dataclasses.replace(
            self.graph, edge_features=self.graph.edge_features.copy()
        )
        heldout.graph.edge_features[self.test + 1] *= 1000
        third = self.fit("frozen", heldout)
        self.assertEqual(first.training_history, third.training_history)
        for key, value in first.network.state_dict().items():
            torch.testing.assert_close(
                value, third.network.state_dict()[key], rtol=0, atol=0
            )
        self.assertEqual(first.amount_normalization, third.amount_normalization)

    def test_pure_timestamp_scoring_candidate_amount_and_future_isolation(self):
        import torch

        model = self.fit("frozen")
        earlier = np.arange(12, 20)
        expected = model.predict_fraud(self.graph, earlier)["fraud_logits"]
        features = self.graph.edge_features.copy()
        features[30:] *= 1000
        altered = dataclasses.replace(self.graph, edge_features=features)
        np.testing.assert_array_equal(
            expected, model.predict_fraud(altered, earlier)["fraud_logits"]
        )
        model._bind(self.graph)
        group = np.array([0, 1])
        with torch.no_grad():
            first, _, proposed = model.score_group(self.graph, group)
            self.assertFalse(model.memory.most_recent_hist_emb)
            features = self.graph.edge_features.copy()
            features[1] += 3
            changed = dataclasses.replace(self.graph, edge_features=features)
            # Deterministically give the candidate feature a nonzero coefficient.
            model.head.linear.weight[0, -1] = 1
            baseline, _, before = model.score_group(self.graph, group)
            amounts, _, after = model.score_group(changed, group)
            self.assertNotEqual(float(baseline[0]), float(amounts[0]))
            self.assertEqual(float(baseline[1]), float(amounts[1]))
            torch.testing.assert_close(before, after, rtol=0, atol=0)
            self.assertFalse(model.memory.most_recent_hist_emb)
            model._commit(self.graph, group, proposed)
            self.assertTrue(model.memory.most_recent_hist_emb)
            self.assertTrue(
                all(
                    not value.requires_grad
                    for value in model.memory.most_recent_hist_emb.values()
                )
            )

    def test_availability_and_class_coverage(self):
        delayed = copy.copy(self.dataset)
        delayed.label_available_at = self.dataset.label_available_at.copy()
        delayed.label_available_at[0] = 10000
        model = self.fit("frozen", delayed)
        self.assertEqual(model.label_counts["fraud"], 6)
        self.assertEqual(model.label_counts["unknown_or_unavailable"], 2)
        for indices, stage in (
            (self.training, "Training"),
            (self.validation, "Validation"),
        ):
            missing = copy.copy(self.dataset)
            missing.labels = self.dataset.labels.copy()
            missing.labels[indices] = 0
            with self.assertRaisesRegex(
                ValueError, stage + " requires both confirmed classes"
            ):
                self.fit(dataset=missing)
        with self.assertRaisesRegex(ValueError, "class_weight"):
            self.fit(class_weight=-1)

    def test_safe_checkpoint_roundtrip_and_missing_head_rejection(self):
        from models.implementations.dyg_tami_fraud import Model

        model = self.fit()
        expected = model.predict_fraud(self.graph, self.test)
        path = self.base / "fraud.npz"
        model.experiment_metadata = {
            "task": "temporal-fraud-classification",
            "split": {"train": [self.graph.ids[i] for i in self.training]},
        }
        model.save(path)
        restored = Model()
        restored.load_fraud(path, self.graph)
        self.assertEqual(restored.experiment_metadata, model.experiment_metadata)
        for key, values in expected.items():
            np.testing.assert_array_equal(
                values, restored.predict_fraud(self.dataset, self.test)[key]
            )
        with np.load(path, allow_pickle=False) as saved:
            tensors = {key: saved[key].copy() for key in saved.files}
        missing = self.base / "missing.npz"
        np.savez(
            missing,
            **{key: value for key, value in tensors.items() if key != "2.linear.weight"}
        )
        with self.assertRaisesRegex(ValueError, "missing or unexpected tensor"):
            Model().load_fraud(missing, self.graph)
        metadata = json.loads(str(tensors["metadata"]))
        metadata["amount_normalization"]["scale"] = 0
        tensors["metadata"] = np.asarray(json.dumps(metadata))
        invalid = self.base / "invalid.npz"
        np.savez(invalid, **tensors)
        with self.assertRaisesRegex(ValueError, "normalization"):
            Model().load_fraud(invalid, self.graph)
        metadata["experiment_metadata"] = ["invalid provenance object"]
        tensors["metadata"] = np.asarray(json.dumps(metadata))
        np.savez(invalid, **tensors)
        with self.assertRaisesRegex(ValueError, "experiment metadata"):
            Model().load_fraud(invalid, self.graph)

    def test_head_contract_and_stable_fraud_score(self):
        import torch
        from prediction_heads.implementations.fraud_linear import FraudLogitScorer

        head = create_learned_head("fraud_linear", 9)
        self.assertEqual(head(torch.zeros(3, 9)).shape, (3,))
        with self.assertRaisesRegex(ValueError, "incompatible"):
            head(torch.zeros(3, 8))
        with self.assertRaisesRegex(ValueError, "Unknown learned"):
            create_learned_head("os.system", 9)
        self.assertEqual(
            [
                entry["id"]
                for entry in manifest(kind="transaction-representation")[
                    "prediction_heads"
                ]
            ],
            ["fraud_linear"],
        )
        result = FraudLogitScorer().score([-1e300, -10, 0, 10, 1e300])
        self.assertTrue(np.isfinite(result["score"]).all())
        self.assertTrue((np.diff(result["score"]) > 0).all())
        self.assertEqual(result["score"][2], 1)
        self.assertEqual(result["fraud_probability"][2], 0.5)


if __name__ == "__main__":
    unittest.main()
