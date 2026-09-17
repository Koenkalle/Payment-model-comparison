"""TabFM through the real job/artifact/comparison pipeline, with inference stubbed."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from framework.pipeline_training import TrainingService
from framework.experiments import evaluate_artifact
from framework.registry import create_model, load_dataset
from models.implementations import tabfm
from test_training import PaymentsStore


class Classifier:
    classes_ = np.array([1, 0])  # Deliberately reverse the usual class order.
    query_sizes = []

    def predict_proba(self, x):
        self.query_sizes.append(len(x))
        p = np.where(x[:, 0] > 60, 0.8, 0.2)
        return np.column_stack((p, 1 - p))


VERSIONS = dict.fromkeys(("tabfm", "torch", "numpy", "scikit-learn", "scipy"), "test")


def prepare(_model, context_x, context_y, parameters, expected_versions=None):
    if expected_versions is not None:
        assert expected_versions == VERSIONS
    return Classifier(), dict(VERSIONS)


class TabFMPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.store = PaymentsStore(self.base)
        self.service = TrainingService(self.store, self.base / "pipeline")
        self.addCleanup(lambda: self.service.close())

    def wait(self, job):
        self.service._futures[job["id"]].result(timeout=40)
        return self.service.job(job["id"])

    def test_catalog_explains_optional_setup_and_context_mode(self):
        with patch.object(
            tabfm, "availability_error", return_value=tabfm.INSTALL_MESSAGE
        ):
            model = next(
                row for row in self.service.models("original") if row["id"] == "tabfm"
            )
            self.assertFalse(model["available"])
            self.assertIn("requirements-tabfm.txt", model["reason"])
            self.assertEqual(model["training_mode"], "in_context")
            self.assertIn("non-production", model["usage_note"])
            with self.assertRaisesRegex(ValueError, "Python 3.11"):
                self.service.start_training(
                    {"dataset_id": "original", "model_id": "tabfm"}
                )
            self.assertEqual(self.service.jobs(), [])
        model, descriptor = create_model(
            "tabfm", "numeric-table/v1", "fraud-classification"
        )
        self.assertIsInstance(model, tabfm.Model)
        self.assertEqual(descriptor["execution"], ["python"])
        with self.assertRaisesRegex(ValueError, "does not support"):
            create_model("tabfm", "payment-events/v1")

    def test_tabfm_parameters_and_feature_limits_are_checked_before_queueing(self):
        with patch.object(tabfm, "availability_error", return_value=None):
            for parameters in (
                {"max_context_rows": 1},
                {"max_context_rows": 2049},
                {"inference_batch_size": True},
                {"n_estimators": 9},
                {"seed": -1},
                {"checkpoint_path": "/tmp/untrusted"},
                {"device": "cuda"},
                {"inference_batch_size": float("nan")},
            ):
                with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                    self.service.start_training(
                        {
                            "dataset_id": "original",
                            "model_id": "tabfm",
                            "parameters": parameters,
                        }
                    )
            self.assertEqual(self.service.jobs(), [])
            self.store.data["original"]["feature_names"] = [str(i) for i in range(501)]
            model = next(
                row for row in self.service.models("original") if row["id"] == "tabfm"
            )
            self.assertFalse(model["available"])
            self.assertIn("500", model["reason"])

    def test_context_excludes_late_and_heldout_labels_then_reloads_for_comparison(self):
        path = self.base / "original.json"
        document = json.loads(path.read_text())
        document["label_available_at"] = {"p-1": 1000, "p-61": 1000, "p-81": 1000}
        path.write_text(json.dumps(document))
        with patch.object(tabfm, "availability_error", return_value=None), patch.object(
            tabfm.Model, "_prepare", prepare
        ):
            job = self.wait(
                self.service.start_training(
                    {
                        "dataset_id": "original",
                        "model_id": "tabfm",
                        "parameters": {
                            "max_context_rows": 10,
                            "inference_batch_size": 3,
                        },
                    }
                )
            )
            self.assertEqual(job["status"], "succeeded", job.get("error"))
            run = job["result"]["run"]
            self.assertEqual(
                run["partition_counts"], {"train": 60, "validation": 20, "test": 20}
            )
            self.assertEqual(
                run["label_policy"]["masked"], {"train": 1, "validation": 1}
            )
            self.assertEqual(run["model_provenance"]["context_rows"], 10)
            folder = self.base / "pipeline" / "runs" / run["id"]
            state = json.loads((folder / "artifact" / "model.json").read_text())
            metadata = json.loads((folder / "artifact" / "manifest.json").read_text())
            self.assertEqual(state["training_rows"], 59)
            self.assertEqual(set(state["context_y"]), {0, 1})
            self.assertEqual(
                metadata["model_provenance"]["revision"], tabfm.WEIGHTS_REVISION
            )
            config = json.loads((folder / "configuration.json").read_text())
            data = load_dataset(config["dataset"])
            eligible = [i for i in range(60) if data.labels[i] >= 0]
            selected = np.asarray(eligible)[state["context_indices"]]
            np.testing.assert_array_equal(state["context_x"], data.features[selected])
            np.testing.assert_array_equal(state["context_y"], data.labels[selected])
            with self.assertRaisesRegex(ValueError, "Selected evaluation IDs"):
                evaluate_artifact(
                    folder / "artifact",
                    config["dataset"],
                    partition="test",
                    evaluation_ids=["p-0"],
                )
            baseline = self.wait(
                self.service.start_training(
                    {"dataset_id": "original", "model_id": "logistic_regression"}
                )
            )
            self.assertEqual(baseline["status"], "succeeded", baseline.get("error"))
            self.service.close()
            self.service = TrainingService(self.store, self.base / "pipeline")
            self.assertEqual(len(self.service.runs()), 2)
            Classifier.query_sizes.clear()
            comparison = self.wait(
                self.service.start_comparison(
                    {
                        "dataset_id": "original",
                        "partition": "test",
                        "run_ids": [run["id"], baseline["result"]["run_id"]],
                    }
                )
            )
            self.assertEqual(comparison["status"], "succeeded", comparison.get("error"))
            result = comparison["result"]
            self.assertEqual(result["row_count"], 20)
            self.assertEqual(result["training_rows_in_evaluation"], 0)
            self.assertEqual(result["validation_rows_in_evaluation"], 0)
            self.assertEqual(sum(Classifier.query_sizes), 20)
            self.assertLessEqual(max(Classifier.query_sizes), 3)
            initial = json.loads((folder / "artifact" / "test-report.json").read_text())
            expected = {row["id"]: row["probability"] for row in initial["rows"]}
            for row in result["rows"]:
                self.assertEqual(
                    row["predictions"][run["id"]]["probability"], expected[row["id"]]
                )


if __name__ == "__main__":
    unittest.main()
