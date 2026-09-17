"""End-to-end contracts using local CSV fixtures, never a claimed production benchmark."""
import copy, csv, importlib.util, json, sys, tempfile, unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from framework.registry import load_dataset, create_model, manifest
from framework.experiments import (
    chronological_split,
    train_experiment,
    evaluate_artifact,
)
from datasets.implementations.payment_json import normalize


class FrameworkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        with (self.base / "features.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["id", "timestamp", "amount", "prior_count", "outcome"])
            for i in range(120):
                writer.writerow(
                    [
                        f"R{i:03d}",
                        i // 2,
                        5 + 100 * (i % 4 == 0) + i % 5,
                        i % 7,
                        "" if i % 19 == 0 else int(i % 4 == 0),
                    ]
                )
        self.dataset = {
            "loader": "numeric_csv",
            "path": "features.csv",
            "id_column": "id",
            "time_column": "timestamp",
            "time_unit": "minutes",
            "label_column": "outcome",
            "features": ["amount", "prior_count"],
        }

    def tearDown(self):
        self.temp.cleanup()

    def test_loader_and_timestamp_safe_splits(self):
        data = load_dataset(self.dataset, self.base)
        self.assertEqual(data.feature_names, ("amount", "prior_count"))
        self.assertFalse(data.features.flags.writeable)
        self.assertIn(-1, data.labels)
        parts = chronological_split(data, {})
        self.assertLess(
            data.times[parts["train"]].max(), data.times[parts["validation"]].min()
        )
        self.assertLess(
            data.times[parts["validation"]].max(), data.times[parts["test"]].min()
        )
        self.assertEqual(sum(map(len, parts.values())), 120)
        bad = {**self.dataset, "features": ["amount", "outcome"]}
        with self.assertRaisesRegex(ValueError, "outcomes"):
            load_dataset(bad, self.base)
        for fractions in ({"train": 0.9, "validation": 0.2}, {"train": 0}):
            with self.assertRaises(ValueError):
                chronological_split(data, fractions)
        with self.assertRaisesRegex(ValueError, "does not support"):
            create_model("dygformer", "numeric-table/v1")
        with self.assertRaisesRegex(ValueError, "does not support"):
            create_model("xgboost_native", "payment-events/v1")

    def test_invalid_csv(self):
        text = (self.base / "features.csv").read_text()
        for replacement in ("R000,0,nan,0,", "R000,0,not-a-number,0,"):
            (self.base / "features.csv").write_text(
                text.replace("R000,0,105,0,", replacement)
            )
            with self.assertRaises(ValueError):
                load_dataset(self.dataset, self.base)
        (self.base / "features.csv").write_text(text.replace("R001,", "R000,"))
        with self.assertRaisesRegex(ValueError, "unique"):
            load_dataset(self.dataset, self.base)

    def test_payment_csv_identity_and_outcome_isolation(self):
        (self.base / "payments.csv").write_text(
            "id,time,from,to,value,label\na,2026-01-01T00:00:00Z,external-a,external-b,12,1\nb,2026-01-01T00:01:00Z,external-b,external-a,4,\n"
        )
        config = {
            "loader": "payment_csv",
            "path": "payments.csv",
            "time_unit": "iso8601",
            "columns": {
                "id": "id",
                "time": "time",
                "sender": "from",
                "recipient": "to",
                "amount": "value",
                "label": "label",
            },
        }
        data = load_dataset(config, self.base)
        doc = data.document
        self.assertEqual(doc["events"][1]["t"], 1)
        self.assertEqual(doc["accounts"][0]["external_id"], "external-a")
        self.assertEqual(doc["truth"], {"a": True})
        self.assertTrue(all("label" not in event for event in doc["events"]))
        (self.base / "normalized.json").write_text(json.dumps(doc))
        again = load_dataset(
            {"loader": "payment_json", "path": "normalized.json"}, self.base
        )
        self.assertEqual(again.document["events"], doc["events"])
        for mutation in ("duplicate", "time", "truth"):
            bad = copy.deepcopy(doc)
            if mutation == "duplicate":
                bad["events"][1]["id"] = "a"
            elif mutation == "time":
                bad["events"][1]["t"] = -1
            else:
                bad["truth"]["absent"] = False
            with self.assertRaises(ValueError):
                normalize(bad)

    def run_model(self, identifier):
        config = {
            "model": identifier,
            "dataset": self.dataset,
            "parameters": {"num_boost_round": 10}
            if identifier == "xgboost_native"
            else {"C": 0.8},
        }
        artifact = self.base / identifier
        metadata, initial = train_experiment(config, artifact, self.base)
        self.assertEqual(metadata["implementation"]["status"], "library-model")
        self.assertEqual(metadata["threshold_source"], "validation_f1")
        restored = evaluate_artifact(artifact, self.dataset, self.base, "test")
        self.assertEqual(
            [r["id"] for r in initial["rows"]], [r["id"] for r in restored["rows"]]
        )
        np.testing.assert_allclose(
            [r["probability"] for r in initial["rows"]],
            [r["probability"] for r in restored["rows"]],
            rtol=1e-7,
        )
        self.assertEqual(initial["metrics"], restored["metrics"])
        self.assertEqual(restored["partition"], "test")
        self.assertEqual(restored["training_rows_in_evaluation"], 0)
        self.assertGreater(
            evaluate_artifact(artifact, self.dataset, self.base)[
                "training_rows_in_evaluation"
            ],
            0,
        )
        with self.assertRaisesRegex(ValueError, "original dataset"):
            evaluate_artifact(
                artifact, {**self.dataset, "time_unit": "seconds"}, self.base, "test"
            )
        if identifier == "xgboost_native":
            for row in restored["rows"]:
                self.assertAlmostEqual(
                    sum(row["contributions"]) + row["baseline"], row["margin"], places=5
                )
        else:
            # The saved scaler proves preprocessing uses only labeled training rows.
            saved = json.loads((artifact / "model.json").read_text())
            data = load_dataset(self.dataset, self.base)
            indices = chronological_split(data, {})["train"]
            indices = indices[data.labels[indices] >= 0]
            np.testing.assert_allclose(
                saved["mean"], data.features[indices].mean(axis=0)
            )
        with self.assertRaisesRegex(ValueError, "order"):
            evaluate_artifact(
                artifact,
                {**self.dataset, "features": ["prior_count", "amount"]},
                self.base,
            )
        with self.assertRaisesRegex(ValueError, "already exists"):
            train_experiment(config, artifact, self.base)
        (artifact / "model.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "checksum"):
            evaluate_artifact(artifact, self.dataset, self.base)

    @unittest.skipUnless(
        importlib.util.find_spec("sklearn"), "optional scikit-learn dependency"
    )
    def test_logistic_save_load(self):
        self.run_model("logistic_regression")

    @unittest.skipUnless(
        importlib.util.find_spec("xgboost"), "optional official XGBoost dependency"
    )
    def test_native_xgboost_save_load(self):
        self.run_model("xgboost_native")

    def test_each_model_has_an_inspection_entry(self):
        for descriptor in manifest("models")["models"]:
            self.assertTrue(
                (
                    ROOT / (descriptor["python_module"].replace(".", "/") + ".py")
                ).is_file()
            )
            if descriptor.get("browser"):
                self.assertTrue((ROOT / descriptor["browser"]).is_file())


if __name__ == "__main__":
    unittest.main()
