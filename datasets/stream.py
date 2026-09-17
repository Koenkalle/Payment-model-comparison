"""Repeatable transaction candidates and separately stored evaluation outcomes.

Prepared datasets use SQLite so sorting, uniqueness validation and iteration do
not require holding the source dataset in memory. Each iterator has its own
read-only connection; model-facing events never contain outcome metadata.
"""
from contextlib import closing
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import sqlite3
from types import MappingProxyType
from typing import Mapping


SCHEMA = "transaction-stream/v1"


def _readonly(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _readonly(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_readonly(item) for item in value)
    return value


@dataclass(frozen=True)
class EntityRef:
    kind: str
    id: str
    role: str


@dataclass(frozen=True)
class TransactionEvent:
    event_id: str
    event_time: float
    event_type: str
    entities: tuple[EntityRef, ...]
    features: Mapping

    def __post_init__(self):
        object.__setattr__(self, "entities", tuple(self.entities))
        object.__setattr__(self, "features", _readonly(self.features))


@dataclass(frozen=True)
class Outcome:
    event_id: str
    label: int
    available_at: float | None = None
    diagnostics: Mapping = field(default_factory=dict)

    def __post_init__(self):
        if type(self.label) is not int or self.label not in (-1, 0, 1):
            raise ValueError("Outcome label must be -1, 0 or 1.")
        if self.available_at is not None and not math.isfinite(self.available_at):
            raise ValueError("Outcome availability must be finite or absent.")
        object.__setattr__(self, "diagnostics", _readonly(self.diagnostics))


class TransactionStream:
    """An ordered, repeatable view of a prepared local transaction dataset.

    Times and half-open selection bounds use source-relative seconds. Physical
    source order resolves ties reproducibly; it is not a measured causal order.
    Temporary stores are removed by close(), including context manager exit.
    """

    schema = SCHEMA

    def __init__(self, path, *, temporary=False):
        self.path = Path(path).resolve()
        self._temporary = temporary
        self._closed = False
        if not self.path.is_file():
            raise ValueError("Prepared dataset does not exist: " + str(self.path))
        try:
            with closing(self._connect()) as database:
                metadata = {
                    key: json.loads(value)
                    for key, value in database.execute(
                        "SELECT key, value FROM metadata"
                    )
                }
                if metadata.get("schema") != SCHEMA:
                    raise ValueError("Unsupported prepared dataset schema.")
                self.descriptor = metadata["descriptor"]
                self.provenance = metadata["provenance"]
                self.count = database.execute("SELECT COUNT(*) FROM events").fetchone()[
                    0
                ]
                outcomes = database.execute("SELECT COUNT(*) FROM outcomes").fetchone()[
                    0
                ]
                if outcomes != self.count or self.count == 0:
                    raise ValueError(
                        "Prepared dataset has missing outcomes or no events."
                    )
                invalid = database.execute(
                    """
                    SELECT 1 FROM events e LEFT JOIN outcomes o ON e.event_id=o.event_id
                    WHERE o.event_id IS NULL OR o.label NOT IN (-1,0,1)
                       OR (o.available_at IS NOT NULL AND o.available_at < e.event_time)
                    LIMIT 1
                """
                ).fetchone()
                if invalid:
                    raise ValueError(
                        "Prepared dataset has invalid outcome identities or availability."
                    )
        except (sqlite3.Error, KeyError, json.JSONDecodeError) as error:
            raise ValueError(
                "Invalid prepared transaction dataset: " + str(error)
            ) from error

    def _connect(self):
        if self._closed:
            raise ValueError("Transaction stream is closed.")
        return sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)

    def iter_events(self, *, start=None, stop=None):
        """Open a fresh iterator over observable events in [start, stop)."""
        for name, value in (("start", start), ("stop", stop)):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(name + " must be finite seconds.")
        if start is not None and stop is not None and start > stop:
            raise ValueError("start must not exceed stop.")
        clauses, parameters = [], []
        if start is not None:
            clauses.append("event_time >= ?")
            parameters.append(start)
        if stop is not None:
            clauses.append("event_time < ?")
            parameters.append(stop)
        query = "SELECT event_id,event_time,event_type,entities,features FROM events"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY event_time, arrival_seq"
        with closing(self._connect()) as database:
            for (
                event_id,
                event_time,
                event_type,
                entities,
                features,
            ) in database.execute(query, parameters):
                yield TransactionEvent(
                    event_id,
                    event_time,
                    event_type,
                    tuple(EntityRef(**entity) for entity in json.loads(entities)),
                    json.loads(features),
                )

    def iter_batches(self, batch_size=1024, *, start=None, stop=None):
        """Yield bounded tuples; timestamp groups may span storage batches."""
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        batch = []
        iterator = self.iter_events(start=start, stop=stop)
        try:
            for event in iterator:
                batch.append(event)
                if len(batch) == batch_size:
                    yield tuple(batch)
                    batch = []
            if batch:
                yield tuple(batch)
        finally:
            iterator.close()

    def truth_for(self, ids):
        """Fetch trusted outcomes by identity, including unknown labels (-1).

        Missing identities are absent from the result. The request is processed
        in bounded SQL batches, so callers can supply a generator of IDs.
        """
        result, pending = {}, []
        with closing(self._connect()) as database:

            def read_batch():
                placeholders = ",".join("?" for _ in pending)
                query = (
                    "SELECT event_id,label,available_at,diagnostics FROM outcomes WHERE event_id IN ("
                    + placeholders
                    + ")"
                )
                for event_id, label, available_at, diagnostics in database.execute(
                    query, pending
                ):
                    result[event_id] = Outcome(
                        event_id, label, available_at, json.loads(diagnostics)
                    )

            for identifier in ids:
                pending.append(identifier)
                if len(pending) == 500:
                    read_batch()
                    pending.clear()
            if pending:
                read_batch()
        return result

    def close(self):
        if not self._closed:
            self._closed = True
            if self._temporary:
                self.path.unlink(missing_ok=True)

    def __enter__(self):
        if self._closed:
            raise ValueError("Transaction stream is closed.")
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        if getattr(self, "_temporary", False):
            try:
                self.close()
            except OSError:
                pass


def open_source(config, base=None):
    """Prepare source CSVs into a temporary stream; see datasets.adapters."""
    from .adapters import open_source as implementation

    return implementation(config, base)


def prepare_source(config, base, output):
    """Create a durable, ordered stream without overwriting existing data."""
    from .adapters import prepare_source as implementation

    return implementation(config, base, output)


def open_prepared(path):
    """Open a durable store independently of its original CSV locations."""
    from .adapters import open_prepared as implementation

    return implementation(path)
