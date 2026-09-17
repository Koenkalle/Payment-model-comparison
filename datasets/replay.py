"""Causal transaction replay and identity-safe joins for external predictions.

Models implement ``predict(events, history)`` and return scores keyed by event
ID, or rows containing ``event_id`` and ``score``. Optional ``observe`` and
``learn`` hooks receive immutable history. A factory supplies a fresh model;
an optional ``reset()`` hook is called before every run.
"""
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
import heapq
import math
from numbers import Real


SCORE_KINDS = {
    "probability": "Fraud probability in [0, 1]; higher means more likely fraud.",
    "risk": "Uncalibrated finite risk score; higher means more likely fraud.",
    "logit": "Finite fraud log odds; higher means more likely fraud.",
}


@dataclass(frozen=True)
class EventEdge:
    """An observed directed transaction, with typed, role-bearing endpoints."""

    event_id: str
    event_time: float
    event_type: str
    source: object
    destination: object


@dataclass(frozen=True)
class HistoryView:
    """Immutable retained history; empty entities produce no graph edges.

    Entity identity is the pair ``(kind, id)``. Roles describe participation
    in an event and are not part of node identity. This lightweight projection
    needs neither a graph library nor fabricated identifiers.
    """

    events: tuple

    @property
    def edges(self):
        return tuple(
            EventEdge(
                event.event_id, event.event_time, event.event_type, source, destination
            )
            for event in self.events
            for source in event.entities
            if source.role == "source"
            for destination in event.entities
            if destination.role == "destination"
        )

    def neighbors(self, entity, *, direction="both"):
        if direction not in {"both", "incoming", "outgoing"}:
            raise ValueError("direction must be both, incoming, or outgoing")
        identity = (entity.kind, entity.id)
        found = {}
        for edge in self.edges:
            if (
                direction in {"both", "outgoing"}
                and (edge.source.kind, edge.source.id) == identity
            ):
                found.setdefault(
                    (edge.destination.kind, edge.destination.id), edge.destination
                )
            if (
                direction in {"both", "incoming"}
                and (edge.destination.kind, edge.destination.id) == identity
            ):
                found.setdefault((edge.source.kind, edge.source.id), edge.source)
        return tuple(found.values())


@dataclass(frozen=True)
class Feedback:
    """Released supervision only; source diagnostics never enter model hooks."""

    event: object
    label: int
    available_at: float


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _finite(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _score_kind(kind):
    if kind not in SCORE_KINDS:
        raise ValueError(f"score_kind must be one of {', '.join(SCORE_KINDS)}")


def _prediction_rows(rows, score_kind):
    _score_kind(score_kind)
    if isinstance(rows, Mapping):
        rows = (
            {"event_id": identifier, "score": score}
            for identifier, score in rows.items()
        )
    result = {}
    for source in rows:
        if not isinstance(source, Mapping):
            raise ValueError("prediction rows must contain event_id and score")
        row = dict(source)
        identifier = row.get("event_id", row.get("id"))
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("prediction event_id must be a nonempty string")
        if "id" in row and row["id"] != identifier:
            raise ValueError("prediction id and event_id disagree")
        if identifier in result:
            raise ValueError(f"duplicate prediction event_id: {identifier}")
        if "score" not in row:
            raise ValueError(f"prediction {identifier} is missing score")
        score = _finite(row["score"], "prediction score")
        if score_kind == "probability" and not 0 <= score <= 1:
            raise ValueError("probability scores must lie in [0, 1]")
        row.pop("id", None)
        row.update(event_id=identifier, score=score)
        result[identifier] = row
    return result


def _check_ids(rows, expected):
    missing, extra = set(expected) - rows.keys(), rows.keys() - set(expected)
    if missing or extra:
        raise ValueError(
            f"prediction IDs do not match: missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}"
        )


def _dataset_metadata(stream):
    return {
        "descriptor": _plain(stream.descriptor),
        "provenance": _plain(stream.provenance),
    }


def _check_identity(stream, identity):
    if identity is None:
        return
    actual = _dataset_metadata(stream)
    # Passing a previous result's dataset object verifies both descriptor and
    # provenance. A partial mapping can explicitly pin selected identity fields.
    flattened = {**actual["descriptor"], **actual["provenance"]}
    flattened.update(actual)
    if not isinstance(identity, Mapping) or not identity:
        raise ValueError("dataset_identity must be a nonempty dataset identity mapping")
    for key, expected in identity.items():
        if key not in flattened or flattened[key] != _plain(expected):
            raise ValueError(f"prediction dataset identity mismatch for {key}")


def _joined_row(event, prediction, outcome):
    row = dict(prediction)
    row.update(
        event_id=event.event_id,
        event_time=event.event_time,
        label=-1 if outcome is None else outcome.label,
        label_available_at=None if outcome is None else outcome.available_at,
    )
    # Outcomes are joined only in this evaluator-owned output, never history.
    row["diagnostics"] = {} if outcome is None else _plain(outcome.diagnostics)
    return row


def join_predictions(
    stream, rows, *, score_kind="probability", event_ids=None, dataset_identity=None
):
    """Join predictions to final outcomes in stream order, with exact ID checks.

    By default every source event needs one prediction. ``event_ids`` explicitly
    selects an evaluation subset; its order does not change the source order.
    The output keeps unknown labels as -1. Optional ``dataset_identity`` can be
    a previous replay result's ``dataset`` object, or pinned descriptor/
    provenance fields such as ``dataset_id`` and ``source_sha256``.
    """
    _check_identity(stream, dataset_identity)
    predictions = _prediction_rows(rows, score_kind)
    selected = None
    if event_ids is not None:
        requested = tuple(event_ids)
        if any(
            not isinstance(identifier, str) or not identifier
            for identifier in requested
        ):
            raise ValueError("event_ids must contain nonempty strings")
        selected = set(requested)
        if len(selected) != len(requested):
            raise ValueError("event_ids must be unique")
        _check_ids(predictions, selected)
    output, seen = [], set()
    for batch in stream.iter_batches(batch_size=1024):
        events = tuple(
            event for event in batch if selected is None or event.event_id in selected
        )
        truth = stream.truth_for([event.event_id for event in events]) if events else {}
        for event in events:
            if event.event_id in seen:
                raise ValueError(f"duplicate stream event_id: {event.event_id}")
            seen.add(event.event_id)
            if event.event_id in predictions:
                output.append(
                    _joined_row(
                        event, predictions[event.event_id], truth.get(event.event_id)
                    )
                )
    _check_ids(predictions, seen)
    return output


def _groups(stream, batch_size):
    """Group across I/O boundaries: a batch boundary is never a causal boundary."""
    current, previous = [], None
    for batch in stream.iter_batches(batch_size=batch_size):
        for event in batch:
            time = _finite(event.event_time, "event_time")
            if previous is not None and time < previous:
                raise ValueError("replay requires chronological event order")
            if current and time != previous:
                yield tuple(current)
                current = []
            current.append(event)
            previous = time
    if current:
        yield tuple(current)


def replay(
    stream,
    model_factory,
    *,
    split_time,
    history="growing",
    label_delay=None,
    learning=False,
    batch_size=1024,
    max_events=None,
    score_kind="probability",
):
    """Predict the suffix ``event_time >= split_time`` against causal history.

    Timestamps are atomic even across input batches. The prefix is observed
    without scoring. ``history='fixed'`` freezes observed history at the split;
    ``'growing'`` observes each evaluation group after scoring it. ``max_events``
    bounds the runner's retained history, not model-owned state, duplicate-ID
    validation, pending feedback, atomic timestamp groups, or output rows.

    ``learning=True`` calls ``learn(feedback, history)`` only when labels become
    available. Recorded availability is used by default; retrospective labels
    with no timestamp never reach the learner. ``label_delay`` explicitly
    overrides recorded availability with event time plus this fixed delay,
    expressed in the stream's time units. Zero-delay feedback follows scoring
    of the entire timestamp group. Pending feedback is not flushed into the
    future when the stream ends. Model factories must return a fresh model or
    implement reset() to clear all state between runs. An empty prefix allows
    cold-start evaluation; an empty evaluation suffix is an error.
    """
    if getattr(stream, "schema", None) != "transaction-stream/v1":
        raise ValueError("replay requires transaction-stream/v1")
    split_time = _finite(split_time, "split_time")
    if history not in {"fixed", "growing"}:
        raise ValueError("history must be fixed or growing")
    if not isinstance(learning, bool):
        raise ValueError("learning must be a boolean")
    _positive_int(batch_size, "batch_size")
    if max_events is not None:
        _positive_int(max_events, "max_events")
    _score_kind(score_kind)
    if label_delay is not None:
        label_delay = _finite(label_delay, "label_delay")
        if label_delay < 0:
            raise ValueError("label_delay must be nonnegative")
    if not callable(model_factory):
        raise ValueError("model_factory must create a fresh model for each run")
    model = model_factory()
    if not callable(getattr(model, "predict", None)):
        raise ValueError("replay model must implement predict(events, history)")
    if learning and not callable(getattr(model, "learn", None)):
        raise ValueError("learning requires model.learn(feedback, history)")
    if callable(getattr(model, "reset", None)):
        model.reset()

    retained = deque(maxlen=max_events)
    pending, output, seen = [], [], set()
    sequence = released = prefix_count = retrospective = pending_unknown = 0
    last_time = None

    def view():
        return HistoryView(tuple(retained))

    def release(time):
        nonlocal released
        ready = []
        while pending and pending[0][0] <= time:
            ready.append(heapq.heappop(pending)[2])
        if ready:
            model.learn(tuple(ready), view())
            released += len(ready)

    for events in _groups(stream, batch_size):
        time = events[0].event_time
        last_time = time
        identifiers = tuple(event.event_id for event in events)
        if len(set(identifiers)) != len(identifiers) or seen.intersection(identifiers):
            raise ValueError("stream event_id values must be unique")
        seen.update(identifiers)
        if learning:
            release(time)
        prefix = time < split_time
        if prefix:
            prefix_count += len(events)
        else:
            predictions = _prediction_rows(model.predict(events, view()), score_kind)
            _check_ids(predictions, identifiers)

        # Only after the group is scored may observable events enter history.
        if prefix or history == "growing":
            retained.extend(events)
            if callable(getattr(model, "observe", None)):
                model.observe(events, view())

        truth = stream.truth_for(identifiers)
        if not prefix:
            output.extend(
                _joined_row(
                    event, predictions[event.event_id], truth.get(event.event_id)
                )
                for event in events
            )
        if learning:
            for event in events:
                outcome = truth.get(event.event_id)
                if outcome is None or outcome.label == -1:
                    pending_unknown += 1
                    continue
                available_at = (
                    outcome.available_at
                    if label_delay is None
                    else event.event_time + label_delay
                )
                if available_at is None:
                    retrospective += 1
                    continue
                available_at = _finite(available_at, "label available_at")
                if available_at < event.event_time:
                    raise ValueError("label availability cannot precede its event")
                sequence += 1
                heapq.heappush(
                    pending,
                    (
                        available_at,
                        sequence,
                        Feedback(event, outcome.label, available_at),
                    ),
                )
            release(time)

    if not output:
        raise ValueError(
            "replay split contains no evaluation events; choose an earlier split_time"
        )

    def maturity(row):
        available = (
            row["label_available_at"]
            if label_delay is None
            else row["event_time"] + label_delay
        )
        if row["label"] == -1:
            return "unknown"
        if available is None:
            return "retrospective"
        return "mature" if available <= last_time else "pending"

    maturity_counts = {"unknown": 0, "retrospective": 0, "mature": 0, "pending": 0}
    for row in output:
        maturity_counts[maturity(row)] += 1
    return {
        "schema": "transaction-replay/v1",
        "dataset": _dataset_metadata(stream),
        "manifest": {
            "split_time": split_time,
            "split_rule": "prefix event_time < split_time; evaluation event_time >= split_time",
            "history": history,
            "retention": {
                "max_events": max_events,
                "scope": "runner_history_only",
                "policy": "unbounded" if max_events is None else "last_observed_events",
            },
            "ties": "atomic_timestamp_groups",
            "learning": learning,
            "feedback": {
                "policy": "recorded" if label_delay is None else "fixed_delay",
                "label_delay": label_delay,
                "released": released,
                "pending": len(pending),
                "retrospective_unavailable": retrospective,
                "unknown": pending_unknown,
                "horizon": last_time,
            },
            "score_kind": score_kind,
            "score_semantics": SCORE_KINDS[score_kind],
            "evaluation_labels": "retrospective final outcomes joined by event_id",
            "evaluation_maturity": maturity_counts,
            "prefix_count": prefix_count,
            "evaluation_count": len(output),
            "retained_event_count": len(retained),
        },
        "rows": output,
    }
