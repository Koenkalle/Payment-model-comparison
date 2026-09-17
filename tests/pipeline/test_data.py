"""Durable data pipeline, projection contracts and causal numeric boundaries."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from datasets.implementations.payment_json import normalize
from datasets.implementations.pipeline_numeric import numeric_payments
from datasets.stream import prepare_source
from framework.contracts import EventDataset
from framework.pipeline_data import DatasetStore
from framework.registry import load_dataset


CSV = (
    "TRANSACTION_ID,TX_TIME_SECONDS,CUSTOMER_ID,TERMINAL_ID,TX_AMOUNT,TX_FRAUD,TX_FRAUD_SCENARIO\n"
    "late,120,1,1,12,1,99\nearly,60,1,2,10,0,0\nunknown,120,2,2,8,,\n"
)


class DatasetStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.store = DatasetStore(self.base / "store")

    def tearDown(self):
        self.temporary.cleanup()

    def upload(self, **values):
        return self.store.import_source(
            {"source": "handbook", "csv": CSV, "amount_to_eur": 2.0, **values}
        )

    def test_fixture_is_stored_and_restart_keeps_ids(self):
        catalog = self.store.catalog()
        self.assertEqual(catalog["generators"][0]["id"], "handbook_generator")
        self.assertEqual(len(catalog["datasets"]), 1)
        fixture = catalog["datasets"][0]
        self.assertEqual(fixture["rows"], 48)
        self.assertEqual(fixture["kind"], "fixed")
        config = self.store.dataset_config(fixture["id"], "numeric")
        self.assertTrue(str(config["path"]).startswith(str(self.base)))
        self.assertTrue(Path(config["path"]).is_file())
        self.assertNotIn(str(self.base), json.dumps(catalog))
        self.assertEqual(
            DatasetStore(self.base / "store").catalog()["datasets"][0]["id"],
            fixture["id"],
        )

    def test_uploaded_csv_has_durable_numeric_and_payment_views(self):
        metadata = self.upload()
        self.assertEqual(
            (
                metadata["rows"],
                metadata["known"],
                metadata["fraud"],
                metadata["unknown"],
            ),
            (3, 2, 1, 1),
        )
        numeric = load_dataset(self.store.dataset_config(metadata["id"], "numeric"))
        self.assertEqual(numeric.ids, ("early", "late", "unknown"))
        self.assertEqual(numeric.labels.tolist(), [0, 1, -1])
        self.assertEqual(numeric.features[:, 0].tolist(), [10.0, 12.0, 8.0])
        document = self.store.document(metadata["id"])["dataset"]
        self.assertEqual(
            [event["amount"] for event in document["events"]], [20.0, 24.0, 16.0]
        )
        self.assertEqual(document["truth"], {"early": False, "late": True})
        self.assertNotIn(str(self.base), json.dumps(document))
        again = DatasetStore(self.base / "store")
        self.assertEqual(again.get(metadata["id"]), metadata)
        self.assertEqual(again.document(metadata["id"])["dataset"], document)
        self.assertEqual(again.sample(metadata["id"], 1)["rows"][0]["id"], "early")

    def test_no_currency_assumption_and_ulb_numeric_is_retained(self):
        metadata = self.store.import_source({"source": "handbook", "csv": CSV})
        self.assertEqual(metadata["views"], ["numeric", "graph"])
        self.assertTrue(metadata["requires_conversion"])
        with self.assertRaisesRegex(ValueError, "conversion"):
            self.store.document(metadata["id"])
        columns = ["Time", "Amount", *[f"V{index}" for index in range(1, 29)], "Class"]
        csv = ",".join(columns) + "\n" + ",".join(["0", "20", *[".2"] * 28, "1"]) + "\n"
        metadata = self.store.import_source({"source": "ulb", "csv": csv})
        self.assertEqual(metadata["views"], ["numeric"])
        numeric = load_dataset(self.store.dataset_config(metadata["id"], "numeric"))
        self.assertEqual(numeric.features.shape, (1, 29))
        self.assertEqual(numeric.labels.tolist(), [1])
        self.assertEqual(len(self.store.sample(metadata["id"])["columns"]), 29)

    def test_registered_sources_are_snapshots_independent_of_original_files(self):
        directory = self.base / "original"
        directory.mkdir()
        (directory / "source.csv").write_text(CSV)
        config = {
            "loader": "fraud_dataset",
            "dataset": "handbook",
            "path": "source.csv",
            "name": "Registered local",
            "amount_to_eur": 1.0,
            "selection": {"start": 120},
        }
        (directory / "config.json").write_text(json.dumps(config))
        with prepare_source(config, directory, directory / "source.sqlite"):
            pass
        (directory / "prepared.json").write_text(
            json.dumps(
                {
                    "loader": "prepared_fraud",
                    "path": "source.sqlite",
                    "amount_to_eur": 1.0,
                }
            )
        )
        paths = [directory / "config.json", directory / "prepared.json"]
        store = DatasetStore(self.base / "configured", paths)
        self.assertEqual(len(store.catalog()["datasets"]), 3)
        metadata = next(
            item
            for item in store.catalog()["datasets"]
            if item["selection"].get("start") == 120
        )
        self.assertEqual(metadata["rows"], 2)
        (directory / "source.csv").unlink()
        (directory / "source.sqlite").unlink()
        store = DatasetStore(self.base / "configured", paths)
        self.assertEqual(len(store.catalog()["datasets"]), 3)
        self.assertEqual(
            len(load_dataset(store.dataset_config(metadata["id"], "numeric")).ids), 2
        )

    def test_registered_payment_selection_and_confirmations_are_snapshotted(self):
        document = {
            "accounts": [{"id": 0}, {"id": 1}],
            "events": [
                {
                    "id": f"p{index}",
                    "kind": "payment",
                    "t": index,
                    "u": 0,
                    "v": 1,
                    "amount": index + 2,
                }
                for index in range(4)
            ],
            "truth": {f"p{index}": bool(index % 2) for index in range(4)},
            "label_available_at": {"p0": 0, "p2": 3},
        }
        (self.base / "payments.json").write_text(json.dumps(document))
        config = {
            "loader": "payment_json",
            "path": "payments.json",
            "name": "Selected payments",
            "selection": {"start": 60, "stop": 180},
        }
        (self.base / "payments.config.json").write_text(json.dumps(config))
        store = DatasetStore(
            self.base / "selected", [self.base / "payments.config.json"]
        )
        metadata = next(
            entry
            for entry in store.catalog()["datasets"]
            if entry["name"] == "Selected payments"
        )
        self.assertEqual(metadata["rows"], 2)
        self.assertEqual(metadata["selection"], config["selection"])
        data = load_dataset(store.dataset_config(metadata["id"], "numeric"))
        self.assertEqual(data.ids, ("p1", "p2"))
        self.assertEqual(data.times.tolist(), [60.0, 120.0])
        saved = store.document(metadata["id"])["dataset"]
        self.assertEqual(saved["truth"], {"p1": True, "p2": False})
        self.assertEqual(saved["label_available_at"], {"p2": 3})
        (self.base / "payments.json").unlink()
        self.assertEqual(
            load_dataset(store.dataset_config(metadata["id"], "numeric")).ids, data.ids
        )

    def test_modified_snapshot_payload_or_configuration_is_rejected(self):
        metadata = self.upload()
        path = Path(self.store.dataset_config(metadata["id"], "numeric")["path"])
        original = path.read_bytes()
        path.write_bytes(original + b"changed")
        with self.assertRaisesRegex(ValueError, "payload checksum"):
            self.store.dataset_config(metadata["id"], "numeric")
        path.write_bytes(original)
        config_path = path.parent / "config.json"
        config = json.loads(config_path.read_text())
        config["selection"] = {"start": 120}
        config_path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "configuration checksum"):
            self.store.document(metadata["id"])

    def test_generated_handbook_settings_determinism_and_all_views(self):
        options = {
            "generator": "handbook_generator",
            "parameters": {"transactions": 120, "fraud_rate": 0.2, "seed": 9},
        }
        first = self.store.generate({**options, "name": "First title"})
        second = self.store.generate({**options, "name": "Second title"})
        self.assertNotEqual(first["id"], second["id"])
        self.assertEqual(first["fingerprint"], second["fingerprint"])
        self.assertEqual(first["fraud"], 24)
        self.assertEqual(first["parameters"]["transactions"], 120)
        self.assertEqual(first["provenance"]["origin"], "synthetic")
        self.assertEqual(first["currency"], "EUR")
        for view, schema in [
            ("numeric", "numeric-table/v1"),
            ("graph", "temporal-graph/v1"),
            ("labeled_graph", "labeled-temporal-graph/v1"),
            ("payments", "payment-events/v1"),
        ]:
            self.assertEqual(
                load_dataset(self.store.dataset_config(first["id"], view)).schema,
                schema,
            )
        changed = self.store.generate(
            {**options, "parameters": {**options["parameters"], "seed": 10}}
        )
        self.assertNotEqual(changed["fingerprint"], first["fingerprint"])

    def test_scenario_settings_delays_and_saved_numeric_bridge(self):
        first = self.store.generate(
            {
                "generator": "synthetic_payments",
                "parameters": {
                    "name": "relay",
                    "size": "small",
                    "reportDelay": 30,
                    "forwardDelay": 1,
                },
            }
        )
        second = self.store.generate(
            {
                "generator": "synthetic_payments",
                "parameters": {
                    "name": "relay",
                    "size": "small",
                    "reportDelay": 3000,
                    "forwardDelay": 10,
                },
            }
        )
        left = self.store.document(first["id"])["dataset"]
        right = self.store.document(second["id"])["dataset"]
        self.assertNotEqual(
            [event["t"] for event in left["events"] if event["kind"] == "report"],
            [event["t"] for event in right["events"] if event["kind"] == "report"],
        )
        self.assertTrue(left["label_available_at"])
        self.assertTrue(
            all(
                account["external_id"].startswith("synthetic-")
                for account in left["accounts"]
            )
        )
        self.assertEqual(
            [account["external_id"] for account in left["accounts"]],
            [account["external_id"] for account in right["accounts"]],
        )
        data = load_dataset(self.store.dataset_config(first["id"], "numeric"))
        self.assertEqual(len(data.ids), first["payment_rows"])
        self.assertEqual(
            data.feature_names,
            ("log1p_amount", "prior_source_count", "prior_destination_count"),
        )
        labeled = load_dataset(self.store.dataset_config(first["id"], "labeled_graph"))
        self.assertEqual(labeled.schema, "labeled-temporal-graph/v1")
        self.assertGreater(np.isfinite(labeled.label_available_at).sum(), 0)

    def test_fingerprint_ignores_names_and_labels_stay_outside_features(self):
        first = self.upload(name="First")
        second = self.upload(name="Renamed")
        self.assertEqual(first["fingerprint"], second["fingerprint"])
        changed = self.upload(csv=CSV.replace("12,1,99", "12,0,77"))
        self.assertNotEqual(changed["fingerprint"], first["fingerprint"])
        np.testing.assert_array_equal(
            load_dataset(self.store.dataset_config(first["id"], "numeric")).features,
            load_dataset(self.store.dataset_config(changed["id"], "numeric")).features,
        )

    def test_invalid_requests_and_failed_import_leave_no_partial_dataset(self):
        before = self.store.catalog()["datasets"]
        invalid = [
            {"generator": "missing"},
            {"generator": "handbook_generator", "path": "/etc/passwd"},
            {"generator": "handbook_generator", "parameters": {"seed": True}},
            {"generator": "handbook_generator", "parameters": {"transactions": 100001}},
            {
                "generator": "synthetic_payments",
                "parameters": {"reportDelay": float("nan")},
            },
            {"generator": "synthetic_payments", "parameters": {"name": "unknown"}},
        ]
        for payload in invalid:
            with self.assertRaises(ValueError):
                self.store.generate(payload)
        for payload in (
            {"source": "handbook", "csv": "bad"},
            {"source": "handbook", "csv": CSV, "path": "/etc/passwd"},
            {"source": "handbook", "csv": CSV, "amount_to_eur": 0},
        ):
            with self.assertRaises(ValueError):
                self.store.import_source(payload)
        for identifier in (
            "../README.md",
            "/etc/passwd",
            "ds-" + "a" * 31,
            "ds-" + "a" * 32 + "/..",
        ):
            with self.assertRaises(ValueError):
                self.store.get(identifier)
        self.assertEqual(self.store.catalog()["datasets"], before)
        self.assertFalse(list(self.store.directory.glob(".creating-*")))

    def test_large_payment_interval_fails_but_numeric_data_and_selected_view_remain(
        self,
    ):
        csv = (
            CSV.splitlines()[0]
            + "\n"
            + "\n".join(f"{index},{index},{index},0,1,0,0" for index in range(260))
        )
        metadata = self.upload(csv=csv)
        self.assertEqual(metadata["rows"], 260)
        self.assertEqual(
            len(load_dataset(self.store.dataset_config(metadata["id"], "numeric")).ids),
            260,
        )
        with self.assertRaisesRegex(ValueError, "256 accounts"):
            self.store.document(metadata["id"])
        document = self.store.document(metadata["id"], {"start": 100, "stop": 110})[
            "dataset"
        ]
        self.assertEqual(len(document["events"]), 10)
        self.assertAlmostEqual(document["events"][0]["t"], 100 / 60)
        with self.assertRaisesRegex(ValueError, "empty"):
            self.store.document(metadata["id"], {"start": 1000})


class CausalNumericTests(unittest.TestCase):
    def test_ties_and_future_truth_do_not_change_candidate_features(self):
        document = normalize(
            {
                "accounts": [{"id": 0}, {"id": 1}, {"id": 2}],
                "events": [
                    {
                        "id": "a",
                        "kind": "payment",
                        "t": 1,
                        "u": 0,
                        "v": 1,
                        "amount": 10,
                    },
                    {
                        "id": "b",
                        "kind": "payment",
                        "t": 1,
                        "u": 0,
                        "v": 1,
                        "amount": 20,
                    },
                    {
                        "id": "deposit",
                        "kind": "deposit",
                        "t": 2,
                        "u": -1,
                        "v": 0,
                        "amount": 10,
                    },
                    {
                        "id": "c",
                        "kind": "payment",
                        "t": 3,
                        "u": 0,
                        "v": 1,
                        "amount": 30,
                    },
                    {
                        "id": "future",
                        "kind": "payment",
                        "t": 4,
                        "u": 1,
                        "v": 2,
                        "amount": 40,
                    },
                ],
                "truth": {"a": True, "b": False},
            }
        )
        first = numeric_payments(EventDataset(document, {}))
        self.assertEqual(first.ids, ("a", "b", "c", "future"))
        self.assertEqual(
            first.features[:, 1:].tolist(), [[0, 0], [0, 0], [2, 2], [0, 0]]
        )
        self.assertEqual(first.times.tolist(), [60.0, 60.0, 180.0, 240.0])
        self.assertEqual(first.labels.tolist(), [1, 0, -1, -1])
        self.assertFalse(first.features.flags.writeable)
        changed = copy.deepcopy(document)
        changed["truth"] = {"a": False, "b": True, "future": True}
        changed["events"][-1]["amount"] = 9000
        second = numeric_payments(EventDataset(changed, {}))
        np.testing.assert_array_equal(first.features[:3], second.features[:3])
        changed["events"][:2] = list(reversed(changed["events"][:2]))
        third = numeric_payments(EventDataset(changed, {}))
        for index, identifier in enumerate(first.ids[:3]):
            np.testing.assert_array_equal(
                first.features[index], third.features[third.ids.index(identifier)]
            )


if __name__ == "__main__":
    unittest.main()
