"""Public schema adapters, disk ordering and outcome isolation regressions."""
import csv
import hashlib
from dataclasses import replace
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from datasets.adapters import ADAPTERS, source_catalog
from datasets.stream import (
    EntityRef,
    Outcome,
    open_prepared,
    open_source,
    prepare_source,
)


HANDBOOK_FIELDS = [
    "TRANSACTION_ID",
    "TX_TIME_SECONDS",
    "CUSTOMER_ID",
    "TERMINAL_ID",
    "TX_AMOUNT",
    "TX_FRAUD",
    "TX_FRAUD_SCENARIO",
    "TX_DATETIME",
]


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)

    def csv(self, name, fields, rows):
        path = self.base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            writer.writerows(rows)
        return path

    def handbook(self, rows=None, name="handbook.csv"):
        self.csv(
            name,
            HANDBOOK_FIELDS,
            rows
            or [
                ["late", 30, 42, 42, 20, 1, 2, "2018-04-01 00:00:30"],
                ["early-a", 10, 42, 42, 10, 0, 0, "2018-04-01 00:00:10"],
                ["early-b", 10, 3, 42, 30, "", 0, "2018-04-01 00:00:10"],
            ],
        )
        return {"dataset": "handbook", "path": name, "release": "fixture-v1"}

    def test_repeatable_sorted_iteration_and_half_open_batches(self):
        with open_source(self.handbook(), self.base) as stream:
            expected = ["early-a", "early-b", "late"]
            self.assertEqual(stream.count, 3)
            self.assertEqual(
                [event.event_id for event in stream.iter_events()], expected
            )
            first, second = stream.iter_events(), stream.iter_events()
            self.assertEqual(next(first), next(second))
            self.assertEqual(list(first), list(second))
            for size in (1, 2, 4):
                self.assertEqual(
                    [
                        event.event_id
                        for batch in stream.iter_batches(size)
                        for event in batch
                    ],
                    expected,
                )
            self.assertEqual(
                [event.event_id for event in stream.iter_events(start=10, stop=30)],
                expected[:2],
            )
            self.assertEqual(list(stream.iter_events(start=10, stop=10)), [])

    def test_candidates_are_immutable_and_truth_is_separate(self):
        with open_source(self.handbook(), self.base) as stream:
            event = next(stream.iter_events())
            self.assertEqual(dict(event.features), {"amount": 10})
            self.assertFalse(hasattr(event, "diagnostics"))
            self.assertFalse(hasattr(event, "label"))
            with self.assertRaises(TypeError):
                event.features["TX_FRAUD"] = 1
            truth = stream.truth_for(["late", "early-b", "missing"])
            self.assertEqual(set(truth), {"late", "early-b"})
            self.assertEqual(truth["early-b"].label, -1)
            self.assertIsNone(truth["late"].available_at)
            self.assertEqual(truth["late"].diagnostics["TX_FRAUD_SCENARIO"], "2")
            with self.assertRaises(TypeError):
                truth["late"].diagnostics["TX_FRAUD_SCENARIO"] = 0

    def test_distinct_entity_types_preserve_equal_source_identifiers(self):
        with open_source(self.handbook(), self.base) as stream:
            event = next(stream.iter_events())
            self.assertEqual(
                [(entity.kind, entity.id, entity.role) for entity in event.entities],
                [("customer", "42", "source"), ("terminal", "42", "destination")],
            )
            self.assertIsNone(stream.descriptor["currency"])
            self.assertIsNone(stream.descriptor["clock"]["timezone"])
            self.assertEqual(stream.descriptor["clock"]["kind"], "relative")
            self.assertFalse(stream.descriptor["ordering"]["source_order_is_causal"])

    def test_partition_order_and_cross_file_sort(self):
        self.handbook([["b", 20, 1, 2, 10, 0, 0, ""]], "parts/02.csv")
        self.handbook(
            [["a", 20, 1, 2, 10, 0, 0, ""], ["c", 10, 1, 2, 10, 0, 0, ""]],
            "parts/01.csv",
        )
        with open_source({"dataset": "handbook", "path": "parts"}, self.base) as stream:
            self.assertEqual(
                [event.event_id for event in stream.iter_events()], ["c", "a", "b"]
            )
        with open_source(
            {"dataset": "handbook", "paths": ["parts/02.csv", "parts/01.csv"]},
            self.base,
        ) as stream:
            self.assertEqual(
                [event.event_id for event in stream.iter_events()], ["c", "b", "a"]
            )

    def test_durable_preparation_roundtrip_and_no_overwrite(self):
        config = self.handbook()
        output = self.base / "prepared.sqlite"
        with prepare_source(config, self.base, output) as stream:
            events, descriptor, provenance = (
                list(stream.iter_events()),
                stream.descriptor,
                stream.provenance,
            )
        self.assertTrue(output.exists())
        before = output.read_bytes()
        with self.assertRaisesRegex(ValueError, "already exists"):
            prepare_source(config, self.base, output)
        self.assertEqual(output.read_bytes(), before)
        (self.base / "handbook.csv").unlink()
        with open_prepared(output) as restored:
            self.assertEqual(list(restored.iter_events()), events)
            self.assertEqual(restored.descriptor, descriptor)
            self.assertEqual(restored.provenance, provenance)
            self.assertEqual(restored.truth_for(["late"])["late"].label, 1)

    def test_temporary_cleanup_and_closed_stream(self):
        with open_source(self.handbook(), self.base) as stream:
            path = stream.path
            self.assertTrue(path.exists())
        self.assertFalse(path.exists())
        stream.close()
        with self.assertRaisesRegex(ValueError, "closed"):
            list(stream.iter_events())

    def test_source_fingerprint_tracks_content(self):
        config = self.handbook()
        path = self.base / "handbook.csv"
        with open_source(config, self.base) as stream:
            fingerprint = stream.provenance["source_sha256"]
            self.assertEqual(fingerprint, hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertNotIn("prepared_sha256", stream.provenance)
        path.write_text(path.read_text().replace("late,30", "late,31"))
        with open_source(config, self.base) as stream:
            self.assertNotEqual(fingerprint, stream.provenance["source_sha256"])

    def test_prepared_fingerprint_detects_valid_feature_edits(self):
        path = self.base / "edited.sqlite"
        with prepare_source(self.handbook(), self.base, path) as stream:
            source_hash = stream.provenance["source_sha256"]
            prepared_hash = stream.provenance["prepared_sha256"]
            self.assertEqual(
                prepared_hash, hashlib.sha256(path.read_bytes()).hexdigest()
            )
        with sqlite3.connect(path) as database:
            database.execute(
                "UPDATE events SET features=? WHERE event_id=?",
                ('{"amount": 999}', "late"),
            )
        with open_prepared(path) as stream:
            self.assertEqual(stream.provenance["source_sha256"], source_hash)
            self.assertNotEqual(stream.provenance["prepared_sha256"], prepared_hash)
            self.assertEqual(list(stream.iter_events())[-1].features["amount"], 999)

    def test_duplicate_ids_across_partitions_leave_no_output(self):
        self.handbook([["duplicate", 20, 1, 2, 10, 0, 0, ""]], "parts/01.csv")
        self.handbook([["duplicate", 30, 1, 2, 10, 0, 0, ""]], "parts/02.csv")
        output = self.base / "bad.sqlite"
        with self.assertRaisesRegex(ValueError, "duplicate event ID"):
            prepare_source({"dataset": "handbook", "path": "parts"}, self.base, output)
        self.assertFalse(output.exists())

    def test_unknown_labels_and_missing_label_column(self):
        fields = HANDBOOK_FIELDS[:5]
        self.csv("unlabeled.csv", fields, [["id", 0, 1, 2, 10]])
        with open_source(
            {"dataset": "handbook", "path": "unlabeled.csv"}, self.base
        ) as stream:
            self.assertEqual(stream.truth_for(["id"])["id"].label, -1)

    def test_ulb_has_no_fabricated_graph_and_stable_physical_row_ids(self):
        fields = ["Time", "Amount", *[f"V{i}" for i in range(1, 29)], "Class"]
        rows = [[20, 12, *range(28), 1], [10, 8, *range(28), 0]]
        self.csv("ulb.csv", fields, rows)
        with open_source({"dataset": "ulb", "path": "ulb.csv"}, self.base) as stream:
            events = list(stream.iter_events())
            self.assertEqual(
                [event.event_id for event in events], ["ulb:0:2", "ulb:0:1"]
            )
            self.assertEqual(events[0].entities, ())
            self.assertIsNone(stream.descriptor["graph"])
            self.assertEqual(len(events[0].features), 29)
            self.assertEqual(stream.descriptor["feature_names"][0], "amount")
            self.assertNotIn("Class", events[0].features)
            self.assertNotIn("Time", events[0].features)
            self.assertEqual(stream.truth_for(["ulb:0:1"])["ulb:0:1"].label, 1)

    def test_paysim_hours_pretransaction_balances_and_postevent_diagnostics(self):
        fields = [
            "step",
            "type",
            "amount",
            "nameOrig",
            "oldbalanceOrg",
            "newbalanceOrig",
            "nameDest",
            "oldbalanceDest",
            "newbalanceDest",
            "isFraud",
            "isFlaggedFraud",
        ]
        self.csv(
            "paysim.csv",
            fields,
            [[2, "CASH_OUT", 20, "C1", 100, 80, "C2", 30, 50, 1, 0]],
        )
        with open_source(
            {"dataset": "paysim", "path": "paysim.csv"}, self.base
        ) as stream:
            event = next(stream.iter_events())
            self.assertEqual(event.event_time, 7200)
            self.assertEqual(event.event_type, "CASH_OUT")
            self.assertEqual(
                dict(event.features),
                {"amount": 20, "oldbalanceOrg": 100, "oldbalanceDest": 30},
            )
            self.assertEqual(
                [(entity.id, entity.role) for entity in event.entities],
                [("C1", "source"), ("C2", "destination")],
            )
            self.assertEqual(
                dict(stream.truth_for([event.event_id])[event.event_id].diagnostics),
                {"newbalanceOrig": 80, "newbalanceDest": 50, "isFlaggedFraud": 0},
            )

    def test_invalid_source_values_are_rejected_with_row_context(self):
        for column, value in [
            ("TX_TIME_SECONDS", "nan"),
            ("TX_AMOUNT", "-1"),
            ("CUSTOMER_ID", ""),
            ("TX_FRAUD", "2"),
        ]:
            with self.subTest(column=column):
                row = ["id", 1, 1, 2, 10, 0, 0, ""]
                row[HANDBOOK_FIELDS.index(column)] = value
                config = self.handbook([row])
                with self.assertRaisesRegex(ValueError, "row 1"):
                    open_source(config, self.base)

    def test_malformed_headers_rows_and_empty_sources(self):
        for contents, error in [
            ("a,a\n1,2\n", "unique"),
            ("a,b\n1,2\n", "Missing CSV"),
            (",".join(HANDBOOK_FIELDS) + "\n", "no transactions"),
            (",".join(HANDBOOK_FIELDS) + "\nid,1\n", "width"),
        ]:
            with self.subTest(error=error):
                (self.base / "bad.csv").write_text(contents)
                with self.assertRaisesRegex(ValueError, error):
                    open_source({"dataset": "handbook", "path": "bad.csv"}, self.base)

    def test_source_selection_and_iteration_options_validate(self):
        config = self.handbook()
        for change in (
            {"dataset": "other"},
            {"paths": ["handbook.csv"]},
            {"currency": ""},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                open_source({**config, **change}, self.base)
        with open_source(config, self.base) as stream:
            for size in (0, -1, True, 1.5):
                with self.subTest(size=size), self.assertRaises(ValueError):
                    list(stream.iter_batches(size))
            for bounds in (
                {"start": float("nan")},
                {"stop": True},
                {"start": 20, "stop": 10},
            ):
                with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                    list(stream.iter_events(**bounds))

    def test_catalog_and_invalid_prepared_store(self):
        self.assertEqual(
            {item["dataset_id"] for item in source_catalog()},
            {"handbook", "ulb", "paysim"},
        )
        path = self.base / "not.sqlite"
        path.write_text("not a database")
        with self.assertRaisesRegex(ValueError, "Invalid prepared"):
            open_prepared(path)

    def test_prepared_outcome_identity_and_availability_corruption(self):
        for mutation in (
            "UPDATE outcomes SET event_id='extra' WHERE event_id='late'",
            "UPDATE outcomes SET available_at=0 WHERE event_id='late'",
        ):
            with self.subTest(mutation=mutation):
                path = self.base / "corrupted.sqlite"
                with prepare_source(self.handbook(), self.base, path):
                    pass
                with sqlite3.connect(path) as database:
                    database.execute(mutation)
                with self.assertRaisesRegex(ValueError, "identities or availability"):
                    open_prepared(path)
                path.unlink()

    def test_extension_invariants_are_checked_before_writing(self):
        config = self.handbook()
        original = ADAPTERS["handbook"]["parse"]
        mutations = [
            lambda event, truth: (event, replace(truth, event_id="wrong-id")),
            lambda event, truth: (replace(event, event_time=float("inf")), truth),
            lambda event, truth: (
                replace(event, features={"amount": float("nan")}),
                truth,
            ),
            lambda event, truth: (replace(event, features={"wrong-feature": 1}), truth),
            lambda event, truth: (
                replace(event, entities=(EntityRef("customer", "1", "bad-role"),)),
                truth,
            ),
            lambda event, truth: (event, replace(truth, available_at=0)),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):

                def parse(row, generated_id):
                    return mutation(*original(row, generated_id))

                with patch.dict(
                    ADAPTERS["handbook"], {"parse": parse}
                ), self.assertRaises(ValueError):
                    open_source(config, self.base)

    def test_nested_diagnostics_are_immutable_and_roundtrip(self):
        config = self.handbook()
        original = ADAPTERS["handbook"]["parse"]

        def parse(row, generated_id):
            event, truth = original(row, generated_id)
            return event, Outcome(
                truth.event_id, truth.label, diagnostics={"nested": {"values": [1, 2]}}
            )

        with patch.dict(ADAPTERS["handbook"], {"parse": parse}), open_source(
            config, self.base
        ) as stream:
            diagnostics = stream.truth_for(["late"])["late"].diagnostics
            self.assertEqual(diagnostics["nested"]["values"], (1, 2))
            with self.assertRaises(TypeError):
                diagnostics["nested"]["another"] = 3

    def test_extension_recorded_availability_updates_descriptor(self):
        config = self.handbook()
        original = ADAPTERS["handbook"]["parse"]
        for available_ids, expected in (
            ({"late", "early-a"}, "as-of"),
            ({"late"}, "mixed"),
            (set(), "retrospective"),
        ):
            with self.subTest(expected=expected):

                def parse(row, generated_id):
                    event, truth = original(row, generated_id)
                    if event.event_id in available_ids:
                        truth = replace(truth, available_at=event.event_time + 10)
                    return event, truth

                with patch.dict(ADAPTERS["handbook"], {"parse": parse}), open_source(
                    config, self.base
                ) as stream:
                    self.assertEqual(stream.descriptor["label_availability"], expected)
                    self.assertEqual(
                        stream.provenance["dataset_descriptor"]["label_availability"],
                        expected,
                    )
                    self.assertIsNone(
                        stream.truth_for(["early-b"])["early-b"].available_at
                    )


if __name__ == "__main__":
    unittest.main()
