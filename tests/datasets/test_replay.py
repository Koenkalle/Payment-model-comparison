"""Causality and identity contracts independent of any external dataset/model."""
from dataclasses import FrozenInstanceError
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from datasets.replay import HistoryView, join_predictions, replay
from datasets.stream import EntityRef, Outcome, TransactionEvent


def event(identifier, time, *, entities=()):
    return TransactionEvent(identifier, time, "payment", entities,
                            {"amount": 12.0, "nested": {"values": [1, 2]}})


class Stream:
    schema = "transaction-stream/v1"
    descriptor = {"dataset_id": "test", "version": "1", "clock": {"unit": "seconds"}}
    provenance = {"source_sha256": "abc"}

    def __init__(self, events, outcomes=()):
        self.events = tuple(events)
        self.count = len(self.events)
        self.outcomes = {outcome.event_id: outcome for outcome in outcomes}

    def iter_events(self, *, start=None, stop=None):
        return (event for event in self.events
                if (start is None or event.event_time >= start)
                and (stop is None or event.event_time < stop))

    def iter_batches(self, batch_size=1024, *, start=None, stop=None):
        events = tuple(self.iter_events(start=start, stop=stop))
        for index in range(0, len(events), batch_size):
            yield events[index:index + batch_size]

    def truth_for(self, ids):
        return {identifier: self.outcomes[identifier] for identifier in ids if identifier in self.outcomes}


class Model:
    def __init__(self):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1
        self.calls, self.learned = [], []

    def predict(self, events, history):
        self.calls.append(("predict", tuple(event.event_id for event in events),
                           tuple(event.event_id for event in history.events),
                           tuple(self.learned)))
        return {event.event_id: 0.25 for event in reversed(events)}

    def observe(self, events, history):
        self.calls.append(("observe", tuple(event.event_id for event in events),
                           tuple(event.event_id for event in history.events)))

    def learn(self, feedback, history):
        for item in feedback:
            assert not hasattr(item, "diagnostics")
            assert not hasattr(item.event, "label")
            self.learned.append(item.event.event_id)
        self.calls.append(("learn", tuple(item.event.event_id for item in feedback),
                           tuple(event.event_id for event in history.events)))


class ReplayTests(unittest.TestCase):
    def test_ties_are_atomic_across_batches_and_prefix_is_unscored(self):
        stream = Stream([event("prefix", 0), event("a", 1), event("b", 1), event("c", 2)])
        model = Model()
        result = replay(stream, lambda: model, split_time=1, batch_size=1)
        predictions = [call for call in model.calls if call[0] == "predict"]
        self.assertEqual(predictions, [("predict", ("a", "b"), ("prefix",), ()),
                                       ("predict", ("c",), ("prefix", "a", "b"), ())])
        self.assertEqual([row["event_id"] for row in result["rows"]], ["a", "b", "c"])
        self.assertEqual(result["manifest"]["prefix_count"], 1)
        self.assertEqual(result["manifest"]["ties"], "atomic_timestamp_groups")

    def test_history_and_nested_candidates_are_immutable(self):
        stream = Stream([event("p", 0), event("a", 1)])
        checks = self

        class Inspect:
            def predict(self, events, history):
                with checks.assertRaises(FrozenInstanceError):
                    history.events = ()
                with checks.assertRaises(TypeError):
                    events[0].features["amount"] = 0
                with checks.assertRaises(TypeError):
                    history.events[0].features["nested"]["values"][0] = 0
                checks.assertFalse(hasattr(events[0], "label"))
                return {events[0].event_id: 0.5}

        replay(stream, Inspect, split_time=1)

    def test_fixed_and_growing_history_with_independent_retention(self):
        stream = Stream([event(str(index), index) for index in range(5)])
        fixed = Model()
        result = replay(stream, lambda: fixed, split_time=3, history="fixed", max_events=2)
        self.assertEqual([call[2] for call in fixed.calls if call[0] == "predict"],
                         [("1", "2"), ("1", "2")])
        self.assertEqual(len([call for call in fixed.calls if call[0] == "observe"]), 3)
        self.assertEqual(result["manifest"]["retained_event_count"], 2)
        growing = Model()
        replay(stream, lambda: growing, split_time=3, max_events=2)
        self.assertEqual([call[2] for call in growing.calls if call[0] == "predict"],
                         [("1", "2"), ("2", "3")])

    def test_graph_preserves_typed_identity_and_absent_relations(self):
        customer = EntityRef("customer", "42", "source")
        terminal = EntityRef("terminal", "42", "destination")
        history = HistoryView((event("a", 0, entities=(customer, terminal)), event("b", 1)))
        self.assertEqual(len(history.edges), 1)
        self.assertEqual(history.edges[0].event_id, "a")
        self.assertEqual(history.neighbors(customer), (terminal,))
        self.assertEqual(history.neighbors(terminal, direction="incoming"), (customer,))
        self.assertEqual(history.neighbors(terminal, direction="outgoing"), ())
        self.assertEqual(HistoryView((event("ulb", 0),)).edges, ())

    def test_zero_delay_releases_only_after_entire_timestamp_group(self):
        stream = Stream([event("p", 0), event("a", 1), event("b", 1), event("c", 2)],
                        [Outcome(identifier, 1) for identifier in ("p", "a", "b", "c")])
        model = Model()
        result = replay(stream, lambda: model, split_time=1, learning=True, label_delay=0, batch_size=1)
        predictions = [call for call in model.calls if call[0] == "predict"]
        self.assertEqual(predictions[0][3], ("p",))
        self.assertEqual(predictions[1][3], ("p", "a", "b"))
        self.assertEqual(result["manifest"]["feedback"]["released"], 4)
        self.assertEqual(result["manifest"]["feedback"]["pending"], 0)

    def test_recorded_feedback_respects_availability_and_retrospective_truth(self):
        stream = Stream([event("p", 0), event("a", 1), event("b", 2), event("c", 3)],
                        [Outcome("p", 1, 2), Outcome("a", 1, diagnostics={"scenario": 99}),
                         Outcome("b", -1), Outcome("c", 0, 10)])
        model = Model()
        result = replay(stream, lambda: model, split_time=1, learning=True)
        predictions = [call for call in model.calls if call[0] == "predict"]
        self.assertEqual(predictions[0][3], ())
        self.assertEqual(predictions[1][3], ("p",))
        self.assertEqual(model.learned, ["p"])
        self.assertEqual(result["rows"][0]["label"], 1)
        self.assertEqual(result["rows"][0]["diagnostics"], {"scenario": 99})
        self.assertEqual(result["manifest"]["evaluation_maturity"],
                         {"unknown": 1, "retrospective": 1, "mature": 0, "pending": 1})
        self.assertEqual(result["manifest"]["feedback"]["pending"], 1)

    def test_explicit_delay_overrides_recorded_timing_and_does_not_train_unknown(self):
        stream = Stream([event("a", 0), event("b", 1), event("c", 2)],
                        [Outcome("a", 1, 100), Outcome("b", -1), Outcome("c", 0)])
        model = Model()
        result = replay(stream, lambda: model, split_time=0, learning=True, label_delay=1)
        self.assertEqual(model.learned, ["a"])
        self.assertEqual(result["manifest"]["feedback"]["pending"], 1)
        self.assertEqual(result["manifest"]["feedback"]["unknown"], 1)

    def test_learning_is_independent_of_fixed_history(self):
        stream = Stream([event("a", 0), event("b", 1)], [Outcome("a", 1), Outcome("b", 0)])
        model = Model()
        replay(stream, lambda: model, split_time=0, learning=True, label_delay=0, history="fixed")
        self.assertEqual(model.learned, ["a", "b"])
        self.assertTrue(all(call[2] == () for call in model.calls))

    def test_factory_reset_isolation_and_frozen_parameters(self):
        stream = Stream([event("a", 0)], [Outcome("a", 1, 0)])
        model = Model()
        first = replay(stream, lambda: model, split_time=0)
        second = replay(stream, lambda: model, split_time=0)
        self.assertEqual(first, second)
        self.assertEqual(model.reset_count, 2)
        self.assertEqual(model.learned, [])

    def test_prediction_alignment_and_validation(self):
        stream = Stream([event("a", 0), event("b", 0)])

        class Rows:
            def predict(self, events, history):
                return [{"event_id": "b", "score": 0.8}, {"event_id": "a", "score": 0.2}]

        self.assertEqual([row["score"] for row in replay(stream, Rows, split_time=0)["rows"]], [0.2, 0.8])
        for scores, message in [({"a": 0.1}, "missing"), ({"a": 0.1, "b": 0.2, "c": 0.3}, "extra"),
                                ({"a": float("nan"), "b": 0.2}, "finite"),
                                ({"a": 2, "b": 0.2}, "probability")]:
            class Invalid:
                def predict(self, events, history):
                    return scores
            with self.subTest(scores=scores), self.assertRaisesRegex(ValueError, message):
                replay(stream, Invalid, split_time=0)

    def test_protocol_parameter_and_chronology_validation(self):
        stream = Stream([event("a", 0)])
        for parameters in ({"history": "random"}, {"batch_size": 0}, {"max_events": True},
                           {"label_delay": -1}, {"label_delay": float("inf")},
                           {"score_kind": "mystery"}, {"learning": "false"}):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                replay(stream, Model, split_time=0, **parameters)
        with self.assertRaisesRegex(ValueError, "chronological"):
            replay(Stream([event("a", 1), event("b", 0)]), Model, split_time=0)
        with self.assertRaisesRegex(ValueError, "unique"):
            replay(Stream([event("a", 0), event("a", 1)]), Model, split_time=0)
        with self.assertRaisesRegex(ValueError, "precede"):
            replay(Stream([event("a", 1)], [Outcome("a", 1, 0)]), Model, split_time=0, learning=True)

    def test_future_suffix_changes_cannot_change_earlier_scores(self):
        class AmountHistoryModel:
            def predict(self, events, history):
                prior_amount = sum(item.features["amount"] for item in history.events)
                return {item.event_id: prior_amount + item.features["amount"] for item in events}

        shared = [event("prefix", 0), event("a", 1), event("b", 1)]
        first = Stream(shared + [event("future", 2)], [Outcome("a", 1)])
        changed_future = TransactionEvent("future", 2, "payment", (), {"amount": 100000})
        second = Stream(shared + [changed_future], [Outcome("a", 0)])
        before = replay(first, AmountHistoryModel, split_time=1, score_kind="risk", batch_size=1)
        after = replay(second, AmountHistoryModel, split_time=1, score_kind="risk", batch_size=3)
        self.assertEqual([row["score"] for row in before["rows"][:2]],
                         [row["score"] for row in after["rows"][:2]])
        self.assertNotEqual(before["rows"][2]["score"], after["rows"][2]["score"])

    def test_final_timestamp_zero_delay_feedback_follows_all_scores(self):
        stream = Stream([event("a", 0), event("b", 0)], [Outcome("a", 1), Outcome("b", 0)])
        model = Model()
        replay(stream, lambda: model, split_time=0, learning=True, label_delay=0, batch_size=1)
        self.assertEqual([call[0] for call in model.calls], ["predict", "observe", "learn"])
        self.assertEqual(model.calls[0][1], ("a", "b"))
        self.assertEqual(model.calls[0][3], ())
        self.assertEqual(model.calls[2][1], ("a", "b"))

    def test_empty_evaluation_is_rejected_but_cold_start_is_allowed(self):
        for stream, split_time in [(Stream([]), 0), (Stream([event("a", 0)]), 1)]:
            with self.assertRaisesRegex(ValueError, "no evaluation events"):
                replay(stream, Model, split_time=split_time)
        self.assertEqual(replay(Stream([event("a", 0)]), Model, split_time=0)["manifest"]["prefix_count"], 0)


class PredictionJoinTests(unittest.TestCase):
    def setUp(self):
        self.stream = Stream([event("a", 0), event("b", 1), event("c", 1)],
                             [Outcome("a", 0), Outcome("b", 1, 4), Outcome("c", -1)])

    def test_join_uses_identity_and_keeps_unknown_labels(self):
        rows = join_predictions(self.stream, [{"event_id": "c", "score": 0.3, "label": 0},
                                             {"event_id": "b", "score": 0.2},
                                             {"event_id": "a", "score": 0.1}])
        self.assertEqual([row["event_id"] for row in rows], ["a", "b", "c"])
        self.assertEqual([row["label"] for row in rows], [0, 1, -1])
        self.assertEqual(rows[1]["label_available_at"], 4)

    def test_explicit_subset_preserves_source_order(self):
        rows = join_predictions(self.stream, {"c": -3, "b": 7},
                                event_ids=["c", "b"], score_kind="risk")
        self.assertEqual([row["event_id"] for row in rows], ["b", "c"])
        self.assertEqual([row["score"] for row in rows], [7, -3])

    def test_duplicate_missing_extra_and_wrong_dataset_are_rejected(self):
        cases = [([{"event_id": "a", "score": 0.2}] * 2, {}, "duplicate"),
                 ({"a": 0.1}, {}, "missing"),
                 ({"a": 0.1, "b": 0.2, "c": 0.3, "other": 0.4}, {}, "extra"),
                 ({"absent": 0.1}, {"event_ids": ["absent"]}, "extra"),
                 ({"a": 0.1}, {"event_ids": ["a", "a"]}, "unique"),
                 ({"a": 0.1}, {"event_ids": ["a"], "dataset_identity": {"source_sha256": "wrong"}}, "identity")]
        for rows, options, message in cases:
            with self.subTest(options=options, message=message), self.assertRaisesRegex(ValueError, message):
                join_predictions(self.stream, rows, **options)
        result = replay(self.stream, Model, split_time=0)
        joined = join_predictions(self.stream, result["rows"], dataset_identity=result["dataset"])
        self.assertEqual(joined, result["rows"])


if __name__ == "__main__":
    unittest.main()
