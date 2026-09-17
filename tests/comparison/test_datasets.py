"""Dataset web boundary: real adapters, bounded selections and no client paths."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from framework.dataset_service import DatasetService, MAX_UPLOAD_BYTES
from datasets.stream import prepare_source

CSV = (
    "TRANSACTION_ID,TX_TIME_SECONDS,CUSTOMER_ID,TERMINAL_ID,TX_AMOUNT,TX_FRAUD,TX_FRAUD_SCENARIO\n"
    "late,120,1,1,12,1,99\nearly,60,1,2,10,0,0\nunknown,120,2,2,8,,\n"
)


class DatasetWebTests(unittest.TestCase):
    def setUp(self):
        self.service = DatasetService()

    def upload(self, **values):
        return self.service.load(
            {"id": "csv:handbook", "csv": CSV, "amount_to_eur": 2.0, **values}
        )

    def test_demo_is_available_without_model_loading(self):
        entries = {entry["id"]: entry for entry in self.service.catalog()["datasets"]}
        self.assertFalse(entries["csv:ulb"]["supported"])
        self.assertTrue(entries["csv:paysim"]["upload"])
        data = self.service.load({"id": "handbook-demo"})
        self.assertEqual(data["summary"]["events"], 48)
        self.assertEqual(data["summary"]["known"], 48)
        self.assertIn("invented", data["summary"]["release"])

    def test_csv_projection_currency_labels_ties_and_interval(self):
        data = self.upload(selection={"start": 120, "stop": 180})
        document = data["dataset"]
        self.assertEqual(
            [event["id"] for event in document["events"]], ["late", "unknown"]
        )
        self.assertEqual(document["events"][0]["amount"], 24)
        self.assertEqual(document["events"][0]["t"], 2)
        self.assertEqual(document["truth"], {"late": True})
        self.assertEqual(data["summary"]["unknown"], 1)
        self.assertTrue(
            all("TX_FRAUD_SCENARIO" not in event for event in document["events"])
        )
        self.assertNotEqual(document["events"][0]["u"], document["events"][0]["v"])
        with self.assertRaisesRegex(ValueError, "conversion"):
            self.service.load({"id": "csv:handbook", "csv": CSV})
        with self.assertRaisesRegex(ValueError, "empty"):
            self.upload(selection={"start": 200})

    def test_paysim_uses_same_adapters_and_excludes_post_event_fields(self):
        csv = (
            "step,type,amount,nameOrig,nameDest,oldbalanceOrg,oldbalanceDest,newbalanceOrig,newbalanceDest,isFraud,isFlaggedFraud\n"
            "1,TRANSFER,10,C1,C2,100,0,90,10,1,1\n"
        )
        data = self.service.load(
            {
                "id": "csv:paysim",
                "csv": csv,
                "amount_to_eur": 1.0,
                "release": "local-test",
            }
        )
        self.assertEqual(data["dataset"]["events"][0]["t"], 60)
        self.assertEqual(data["summary"]["known"], 1)
        self.assertNotIn("newbalanceOrig", data["dataset"]["events"][0])

    def test_browser_limits_fail_without_truncation_and_allow_recovery(self):
        csv = (
            CSV.splitlines()[0]
            + "\n"
            + "\n".join(f"{i},{i},{i},0,1,0,0" for i in range(260))
        )
        with self.assertRaisesRegex(ValueError, "256 accounts"):
            self.upload(csv=csv)
        data = self.upload(csv=csv, selection={"stop": 10})
        self.assertEqual(data["summary"]["events"], 10)
        with self.assertRaisesRegex(ValueError, "6 MiB"):
            self.upload(csv="x" * (MAX_UPLOAD_BYTES + 1))

    def test_untrusted_request_options_cannot_select_local_files(self):
        for payload in (
            {"id": "../README.md"},
            {"id": "handbook-demo", "path": "/etc/passwd"},
            {"id": "csv:handbook", "python_module": "os"},
            {"id": "handbook-demo", "csv": CSV},
        ):
            with self.assertRaises(ValueError):
                self.service.load(payload)
        with self.assertRaisesRegex(ValueError, "no account or merchant"):
            self.service.load({"id": "csv:ulb"})
        for selection in (
            {"start": 3, "stop": 2},
            {"limit": 10},
            {"start": float("nan")},
        ):
            with self.assertRaises(ValueError):
                self.upload(selection=selection)

    def test_registered_csv_and_prepared_files_are_available_by_opaque_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            (base / "data.csv").write_text(CSV)
            config = {
                "loader": "fraud_dataset",
                "dataset": "handbook",
                "path": "data.csv",
                "view": "numeric",
                "features": ["amount"],
                "amount_to_eur": 1.0,
                "name": "My local data",
                "selection": {"start": 120},
            }
            (base / "source.json").write_text(json.dumps(config))
            with prepare_source(config, base, base / "data.sqlite"):
                pass
            (base / "prepared.json").write_text(
                json.dumps(
                    {
                        "loader": "prepared_fraud",
                        "path": "data.sqlite",
                        "currency": "EUR",
                        "amount_to_eur": 1.0,
                    }
                )
            )
            service = DatasetService([base / "source.json", base / "prepared.json"])
            catalog = service.catalog()
            self.assertNotIn(str(base), json.dumps(catalog))
            self.assertEqual(service.load({"id": "local:1"})["summary"]["events"], 2)
            self.assertEqual(
                service.load({"id": "local:1", "selection": {}})["summary"]["events"], 3
            )
            self.assertEqual(service.load({"id": "local:2"})["summary"]["events"], 3)


if __name__ == "__main__":
    unittest.main()
