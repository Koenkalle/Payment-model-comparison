"""Feature help follows stored definitions without doing model preprocessing."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from datasets.feature_info import describe_feature, source_definition
from datasets.payment_features import FEATURE_SPECS
from framework.pipeline_data import DatasetStore
from framework.pipeline_service import PipelineService


CSV = (
    "TRANSACTION_ID,TX_TIME_SECONDS,CUSTOMER_ID,TERMINAL_ID,TX_AMOUNT,TX_FRAUD,TX_FRAUD_SCENARIO\n"
    "a,0,1,1,10,0,0\nb,60,2,1,20,1,1\nc,120,1,1,30,0,0\n"
)


class FeatureInfoTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.store = DatasetStore(Path(temporary.name) / "store")
        self.source = self.store.import_source(
            {"source": "handbook", "csv": CSV, "amount_to_eur": 1}
        )

    def test_all_features_have_help_including_sources_and_unavailable_recipes(self):
        catalog = self.store.features(self.source["id"])
        for feature in catalog["features"]:
            with self.subTest(feature=feature["id"]):
                self.assertTrue(feature["info"]["description"])
                self.assertTrue(feature["info"]["sections"])
                self.assertGreater(len(feature["info"]["sections"][0]["text"]), 70)
        for name in [
            "amount",
            "log1p_amount",
            "prior_source_count",
            "prior_destination_count",
            "oldbalanceOrg",
            "oldbalanceDest",
            *[f"V{i}" for i in range(1, 29)],
        ]:
            definition = source_definition(name, "EUR")
            self.assertNotEqual(
                definition["info"]["description"],
                "An original numeric column preserved from the dataset.",
            )
        self.assertIn(
            "business meaning is not available",
            source_definition("V1")["info"]["sections"][0]["text"],
        )

    def test_payment_introductions_identify_each_endpoint_direction_and_measure(self):
        catalog = {
            feature["id"]: feature
            for feature in self.store.features(self.source["id"])["features"]
        }
        descriptions = [
            catalog[spec["id"]]["info"]["description"] for spec in FEATURE_SPECS
        ]
        self.assertEqual(
            len(descriptions),
            len(set(descriptions)),
            "Each payment feature needs its own introduction.",
        )
        for endpoint in ("sender", "recipient"):
            for direction, action in (("out", "sent"), ("in", "received")):
                for recent in ("", "recent_"):
                    count = catalog[f"{endpoint}_{recent}{direction}_count"]["info"][
                        "description"
                    ]
                    value = catalog[f"{endpoint}_{recent}{direction}_value"]["info"][
                        "description"
                    ]
                    for text in (count, value):
                        self.assertIn(endpoint, text)
                        self.assertIn(action, text)
                        self.assertIn(
                            "60 minutes" if recent else "earlier timestamps", text
                        )
                    self.assertIn("How many", count)
                    self.assertIn("Total requested amount", value)
        for name in ("V1", "V28", "custom_risk_measure"):
            self.assertIn(name, source_definition(name)["info"]["description"])

    def test_stale_generic_popup_text_cannot_hide_a_saved_feature_definition(self):
        spec = next(
            spec for spec in FEATURE_SPECS if spec["id"] == "recipient_in_count"
        )
        definition = {
            **spec,
            "kind": "derived",
            "version": "pinned-v1",
            "info": {
                "description": "Numeric input supplied to the model from this dataset."
            },
        }
        before = copy.deepcopy(definition)
        enriched = describe_feature(definition)
        self.assertIn("recipient received", enriched["info"]["description"])
        for field in ("description", "transform", "window", "unit", "version"):
            self.assertEqual(
                enriched[field],
                before[field],
                "Presentation enrichment preserves the saved " + field,
            )
        self.assertEqual(definition, before)
        # Some older renderers cached a copy of the short definition in info.
        definition["info"]["description"] = definition["description"]
        self.assertEqual(
            describe_feature(definition)["info"]["description"],
            enriched["info"]["description"],
        )
        source = {
            "id": "amount",
            "kind": "source",
            "description": "Original numeric dataset column, preserved without modification.",
            "info": {"description": "A stored numeric model input."},
        }
        self.assertIn("payment amount", describe_feature(source)["info"]["description"])
        custom = {
            **definition,
            "description": "Payments received by this recipient in a provider-defined six-hour window.",
            "transform": "sqrt(value)",
            "window": "six hours",
            "info": {"description": "Numeric model input."},
        }
        self.assertEqual(
            describe_feature(custom)["info"]["description"], custom["description"]
        )
        custom["info"][
            "description"
        ] = "Provider explanation for this exact saved feature."
        self.assertEqual(
            describe_feature(custom)["info"]["description"],
            custom["info"]["description"],
        )

    def test_cached_saved_feature_help_is_repaired_without_changing_its_values_or_manifest(
        self,
    ):
        selected = ["amount", "sender_out_count", "recipient_in_count"]
        saved = self.store.save_features(self.source["id"], {"features": selected})
        directory = self.store._directory(saved["id"])
        manifest_path = next(directory.glob(".features-*/manifest.json"))
        manifest = json.loads(manifest_path.read_text())
        for feature in manifest["features"]:
            if feature["id"] in selected:
                feature["info"] = {
                    "description": "Numeric input supplied to the model from this dataset."
                }
        manifest_path.write_text(json.dumps(manifest))
        checksums_path = manifest_path.with_name("checksums.json")
        checksums = json.loads(checksums_path.read_text())
        checksums["manifest_sha256"] = hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest()
        checksums_path.write_text(json.dumps(checksums))
        before = {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob("*")
            if path.is_file()
        }
        public = self.store.describe(self.store.get(saved["id"]))
        features = {
            feature["id"]: feature
            for feature in self.store.features(saved["id"])["features"]
        }
        for feature in public["feature_definitions"]:
            self.assertEqual(feature["info"], features[feature["id"]]["info"])
            self.assertNotIn("Numeric input supplied", feature["info"]["description"])
        self.assertIn(
            "sender sent", features["sender_out_count"]["info"]["description"]
        )
        self.assertIn(
            "recipient received", features["recipient_in_count"]["info"]["description"]
        )
        self.assertEqual(
            self.store.get(saved["id"])["fingerprint"], saved["fingerprint"]
        )
        self.assertEqual(
            before,
            {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in directory.rglob("*")
                if path.is_file()
            },
        )

    def test_help_reads_pinned_saved_definitions_without_loading_or_recalculating_values(
        self,
    ):
        selected = ["amount", "sender_out_count", "graph_ppr_sender_to_recipient"]
        saved = self.store.save_features(self.source["id"], {"features": selected})
        before = self.store.features(saved["id"])
        directory = self.store._directory(saved["id"])
        snapshots = {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob("*")
            if path.is_file()
        }
        with (
            patch.object(
                self.store,
                "_feature_cache",
                side_effect=AssertionError("No feature calculation"),
            ),
            patch(
                "framework.pipeline_data.read_numeric",
                side_effect=AssertionError("No array loading"),
            ),
            patch("datasets.features.FEATURE_RECIPES", []),
        ):
            public = next(
                item
                for item in self.store.catalog()["datasets"]
                if item["id"] == saved["id"]
            )
        self.assertEqual(
            [feature["id"] for feature in public["feature_definitions"]], selected
        )
        pinned = {feature["id"]: feature for feature in before["features"]}
        for feature in public["feature_definitions"]:
            self.assertEqual(
                feature,
                {
                    key: value
                    for key, value in pinned[feature["id"]].items()
                    if key != "enabled"
                },
            )
        self.assertEqual(public["fingerprint"], saved["fingerprint"])
        self.assertEqual(self.store.get(saved["id"]), saved)
        self.assertEqual(
            snapshots,
            {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in directory.rglob("*")
                if path.is_file()
            },
        )

    def test_custom_recipe_help_wins_and_enrichment_does_not_mutate_its_input(self):
        definition = {
            "id": "custom_signal",
            "label": "Custom signal",
            "kind": "derived",
            "description": "Signal supplied by an extension.",
            "transform": "sqrt(amount)",
            "info": {
                "description": "Extension explanation.",
                "sections": [
                    {
                        "title": "Interpretation",
                        "text": "The extension owns this explanation.",
                    }
                ],
            },
        }
        before = copy.deepcopy(definition)
        enriched = describe_feature(definition)
        self.assertEqual(enriched["info"], definition["info"])
        enriched["info"]["sections"][0]["text"] = "changed"
        self.assertEqual(definition, before)
        source = describe_feature(
            {
                "id": "custom_column",
                "kind": "source",
                "description": "Provider-specific units and meaning.",
            }
        )
        self.assertEqual(
            source["info"]["description"], "Provider-specific units and meaning."
        )
        json.dumps(source_definition("unknown_column"), allow_nan=False)

    def test_http_preview_carries_help_without_requesting_statistics(self):
        service = PipelineService.__new__(PipelineService)
        service.store = self.store
        with patch.object(
            self.store,
            "_feature_cache",
            side_effect=AssertionError("No graph computation"),
        ):
            response = service.get("/api/pipeline/datasets/" + self.source["id"])
        self.assertEqual(
            [item["id"] for item in response["dataset"]["feature_definitions"]],
            response["sample"]["columns"],
        )
        self.assertIn(
            "payment amount",
            response["dataset"]["feature_definitions"][0]["description"],
        )


if __name__ == "__main__":
    unittest.main()
