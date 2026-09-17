"""TabFM adapter contracts; upstream inference is replaced, never downloaded."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.implementations import tabfm


REAL_DEPENDENCIES = tabfm._dependencies
VERSIONS = {"tabfm": "1.0.1", "torch": "test", "numpy": np.__version__,
            "scikit-learn": "test", "scipy": "test"}


class FakeClassifier:
    """Keeps the true class order intentionally reversed to catch column bugs."""
    instances = []

    def __init__(self, **parameters):
        self.parameters = parameters
        self.classes_ = np.array([1, 0])
        self.prediction_batches = []
        self.instances.append(self)

    def fit(self, x, y):
        self.x = x.copy()
        self.y = y.copy()
        return self

    def predict_proba(self, x):
        self.prediction_batches.append(x.copy())
        p = 1 / (1 + np.exp(-np.clip(x[:, 0] - self.x[:, 0].mean(), -30, 30)))
        return np.column_stack((p, 1 - p))


class TabFMTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        checkpoint = self.base / "classification"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text("{}")
        (checkpoint / "model.safetensors").write_bytes(b"fake; the fake loader never opens this")
        self.load_weights = Mock(return_value=object())
        self.snapshot = Mock(return_value=str(self.base))
        self.dependencies = patch.object(tabfm, "_dependencies", return_value=(
            FakeClassifier, self.load_weights, self.snapshot, dict(VERSIONS)))
        self.dependencies.start()
        FakeClassifier.instances.clear()
        self.x = np.column_stack((np.arange(100) / 10, np.arange(100) % 7))
        self.y = np.array([0] * 90 + [1] * 10)
        self.names = ("amount", "history")

    def tearDown(self):
        self.dependencies.stop()
        self.temporary.cleanup()

    def fit_model(self, parameters=None):
        model = tabfm.create()
        model.fit(self.x, self.y, self.names, parameters or {})
        return model

    def test_creation_does_not_import_or_download_optional_dependencies(self):
        with patch.object(tabfm, "_dependencies") as dependencies:
            model = tabfm.create()
            self.assertIsNone(model.classifier)
            dependencies.assert_not_called()
        self.snapshot.assert_not_called()

    def test_thread_limit_applies_on_preparation_only(self):
        torch = SimpleNamespace(__version__="test", set_num_threads=Mock())
        upstream = SimpleNamespace(__version__="test", TabFMClassifier=FakeClassifier,
                                   tabfm_v1_0_0_pytorch=SimpleNamespace(load=self.load_weights))
        modules = {"tabfm": upstream, "torch": torch,
                   "huggingface_hub": SimpleNamespace(snapshot_download=self.snapshot),
                   "safetensors.torch": object(),
                   "sklearn": SimpleNamespace(__version__="test"),
                   "scipy": SimpleNamespace(__version__="test")}
        arguments = {"model", "n_estimators", "norm_methods", "max_num_features", "max_num_rows",
                     "batch_size", "random_state", "use_amp", "cache_context", "maybe_quantize_kv_cache",
                     "keep_cache_on_device", "model_type", "checkpoint_path", "device", "dtype", "use_cache"}
        with patch.object(tabfm.sys, "version_info", (3, 12, 1)):
            with patch.object(tabfm.importlib, "import_module", side_effect=modules.__getitem__):
                with patch.object(tabfm.inspect, "signature", return_value=SimpleNamespace(parameters=arguments)):
                    _, load_weights, _, _ = REAL_DEPENDENCIES()
        torch.set_num_threads.assert_not_called()
        self.load_weights.assert_not_called()
        self.snapshot.assert_not_called()
        load_weights(device="cpu", dtype=None)
        torch.set_num_threads.assert_called_once_with(2)
        self.load_weights.assert_called_once_with(device="cpu", dtype=None)

    def test_missing_runtime_is_actionable_and_never_downloads_weights(self):
        with patch.object(tabfm.sys, "version_info", (3, 10, 14)):
            self.assertIn("Python 3.11+", tabfm.availability_error())
        with patch.object(tabfm.sys, "version_info", (3, 12, 1)):
            with patch.object(tabfm.importlib.util, "find_spec", return_value=None):
                self.assertIn("requirements-tabfm.txt", tabfm.availability_error())
            with patch.object(tabfm.importlib.util, "find_spec", return_value=object()):
                with patch.object(tabfm, "_dependencies", side_effect=RuntimeError("Unsupported TabFM API")):
                    self.assertIn("Unsupported TabFM API", tabfm.availability_error())
        self.snapshot.assert_not_called()

    def test_official_loader_gets_pinned_safe_snapshot_and_cpu(self):
        model = self.fit_model()
        self.snapshot.assert_called_once_with(
            repo_id=tabfm.WEIGHTS_REPOSITORY, revision=tabfm.WEIGHTS_REVISION,
            allow_patterns=["classification/config.json", "classification/model.safetensors"])
        self.load_weights.assert_called_once_with(
            model_type="classification", checkpoint_path=str(self.base / "classification"),
            device="cpu", dtype=None, use_cache=True)
        params = model.classifier.parameters
        self.assertEqual(params["batch_size"], 1)
        self.assertEqual(params["random_state"], 42)
        self.assertEqual(params["n_estimators"], 1)
        self.assertEqual(params["max_num_features"], 500)
        self.assertTrue(params["cache_context"])
        self.assertFalse(params["maybe_quantize_kv_cache"])
        self.assertNotIn("inference_batch_size", params)

    def test_no_pickle_fallback_when_safetensors_is_missing(self):
        (self.base / "classification" / "model.safetensors").unlink()
        with self.assertRaisesRegex(RuntimeError, "snapshot is incomplete"):
            self.fit_model()
        self.load_weights.assert_not_called()

    def test_context_is_deterministic_bounded_and_proportional(self):
        first = self.fit_model({"max_context_rows": 20, "seed": 7})
        second = self.fit_model({"max_context_rows": 20, "seed": 7})
        different = self.fit_model({"max_context_rows": 20, "seed": 8})
        self.assertEqual(first.state["context_indices"], second.state["context_indices"])
        self.assertNotEqual(first.state["context_indices"], different.state["context_indices"])
        indices = first.state["context_indices"]
        np.testing.assert_array_equal(first.classifier.x, self.x[indices])
        np.testing.assert_array_equal(first.classifier.y, self.y[indices])
        self.assertEqual(len(indices), 20)
        self.assertEqual(int(first.classifier.y.sum()), 2)
        self.assertEqual(first.provenance["training_rows"], 100)
        self.assertEqual(first.provenance["context_rows"], 20)
        self.assertEqual(first.state["parameters"], {**tabfm.DEFAULT_PARAMETERS, "max_context_rows": 20, "seed": 7})

    def test_rare_classes_survive_context_sampling(self):
        for minority in (0, 1):
            labels = np.full(100, 1 - minority)
            labels[-1] = minority
            model = tabfm.create()
            model.fit(self.x, labels, self.names, {"max_context_rows": 2})
            self.assertEqual(set(model.classifier.y), {0, 1})
            self.assertEqual(len(model.classifier.y), 2)

    def test_prediction_maps_fraud_class_and_bounds_query_batches(self):
        model = self.fit_model({"inference_batch_size": 3})
        query = self.x[:8] + 100
        result = model.predict(query, self.names)
        expected = model.classifier.predict_proba(query)[:, 0]
        np.testing.assert_allclose(result.probabilities, expected)
        self.assertEqual([len(x) for x in model.classifier.prediction_batches[:-1]], [3, 3, 2])
        self.assertTrue(np.isfinite(result.margins).all())
        self.assertIsNone(result.contributions)
        np.testing.assert_array_equal(model.classifier.x, self.x)
        self.assertEqual(len(model.state["context_y"]), 100)
        self.assertEqual(model.predict(np.empty((0, 2)), self.names).probabilities.shape, (0,))

    def test_json_roundtrip_rebuilds_same_context_and_predictions(self):
        model = self.fit_model({"max_context_rows": 20, "inference_batch_size": 3, "seed": 6})
        expected = model.predict(self.x[:10], self.names)
        artifact = self.base / "model.json"
        model.save(artifact)
        saved = json.loads(artifact.read_text())
        self.assertEqual(saved["provenance"]["revision"], tabfm.WEIGHTS_REVISION)
        self.assertEqual(saved["library_versions"], VERSIONS)
        restored = tabfm.create()
        restored.load(artifact, self.names)
        self.assertIsNot(restored.classifier, model.classifier)
        np.testing.assert_array_equal(restored.classifier.x, model.classifier.x)
        np.testing.assert_array_equal(restored.classifier.y, model.classifier.y)
        np.testing.assert_allclose(restored.predict(self.x[:10], self.names).probabilities, expected.probabilities)

    def test_input_and_parameter_validation_happens_before_weights(self):
        bad_parameters = [{"max_context_rows": 1}, {"max_context_rows": 2049},
                          {"n_estimators": True}, {"n_estimators": 9},
                          {"inference_batch_size": 0}, {"inference_batch_size": 257},
                          {"seed": -1}, {"seed": 2147483648}, {"seed": "42"},
                          {"learning_rate": .1}]
        for parameters in bad_parameters:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                self.fit_model(parameters)
        for labels in (np.zeros(100), np.full(100, -1), self.y[:-1], self.y.reshape(-1, 1)):
            with self.assertRaises(ValueError):
                tabfm.create().fit(self.x, labels, self.names, {})
        for x, names in ((self.x, ("same", "same")), (self.x, ("one",)),
                         (np.full((100, 2), np.nan), self.names),
                         (np.zeros((100, 501)), tuple(f"f{i}" for i in range(501)))):
            with self.assertRaises(ValueError):
                tabfm.create().fit(x, self.y, names, {})
        self.snapshot.assert_not_called()

    def test_invalid_outputs_explanations_and_feature_order_fail(self):
        model = self.fit_model()
        with self.assertRaisesRegex(ValueError, "order"):
            model.predict(self.x, self.names[::-1])
        with self.assertRaisesRegex(ValueError, "explanations"):
            model.predict(self.x, self.names, explain=True)
        for probabilities in ([[np.nan, .5]], [[1.1, -.1]], [[.5]], [[.2, .2]]):
            with patch.object(model.classifier, "predict_proba", return_value=np.array(probabilities)):
                with self.assertRaisesRegex(ValueError, "probabilities"):
                    model.predict(self.x[:1], self.names)

    def test_malformed_artifacts_are_rejected_before_any_weight_loading(self):
        model = self.fit_model({"max_context_rows": 10})
        original = model.state
        self.snapshot.reset_mock()
        mutations = [
            lambda x: x.update(version=2),
            lambda x: x.update(feature_names=self.names[::-1]),
            lambda x: x["provenance"].update(revision="main"),
            lambda x: x["parameters"].update(max_context_rows=2),
            lambda x: x["parameters"].pop("seed"),
            lambda x: x.update(context_y=[0] * 10),
            lambda x: x.update(context_x=[[0, float("nan")]] * 10),
            lambda x: x.update(context_indices=[0] * 10),
            lambda x: x.update(context_indices=list(range(1, 11)), training_rows=10),
            lambda x: x.update(training_rows=True),
            lambda x: x.update(library_versions={}),
        ]
        artifact = self.base / "bad.json"
        for mutate in mutations:
            state = copy.deepcopy(original)
            mutate(state)
            artifact.write_text(json.dumps(state))
            with self.subTest(state=state), self.assertRaises(ValueError):
                tabfm.create().load(artifact, self.names)
        self.snapshot.assert_not_called()

    def test_reload_rejects_changed_library_versions_before_download(self):
        model = self.fit_model()
        artifact = self.base / "model.json"
        model.save(artifact)
        self.snapshot.reset_mock()
        with patch.object(tabfm, "_dependencies", return_value=(FakeClassifier, self.load_weights, self.snapshot, {**VERSIONS, "tabfm": "99.0"})):
            with self.assertRaisesRegex(ValueError, "library versions"):
                tabfm.create().load(artifact, self.names)
        self.snapshot.assert_not_called()


if __name__ == "__main__":
    unittest.main()
