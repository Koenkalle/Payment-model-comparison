"""Local CSV adapters for public fraud dataset schemas.

Adapters translate source rows, without fitting transforms, inventing entities
or currency conversions. Preparation validates and orders records on disk.
"""
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import tempfile

from .stream import SCHEMA, EntityRef, TransactionEvent, Outcome, TransactionStream


ADAPTER_VERSION = "1"
HANDBOOK_SOURCE = "https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/SimulatedDataset.html"
ULB_SOURCE = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
PAYSIM_SOURCE = "https://www.kaggle.com/datasets/ealaxi/paysim1"


def _number(value, name, *, nonnegative=False):
    try:
        number = float(value)
    except (ValueError, TypeError):
        raise ValueError(name + " must be numeric.") from None
    if not math.isfinite(number) or (nonnegative and number < 0):
        raise ValueError(
            name + " must be finite" + (" and nonnegative." if nonnegative else ".")
        )
    return number


def _identity(value, name):
    value = str(value).strip()
    if not value:
        raise ValueError(name + " must be a nonempty identity.")
    return value


def _label(value):
    if value is None or str(value).strip().lower() in ("", "-1", "unknown", "null"):
        return -1
    value = str(value).strip()
    if value not in ("0", "1"):
        raise ValueError("Fraud labels must be 0, 1 or unknown.")
    return int(value)


def _handbook(row, generated_id):
    identifier = _identity(row["TRANSACTION_ID"], "TRANSACTION_ID")
    time = _number(row["TX_TIME_SECONDS"], "TX_TIME_SECONDS", nonnegative=True)
    entities = (
        EntityRef("customer", _identity(row["CUSTOMER_ID"], "CUSTOMER_ID"), "source"),
        EntityRef(
            "terminal", _identity(row["TERMINAL_ID"], "TERMINAL_ID"), "destination"
        ),
    )
    event = TransactionEvent(
        identifier,
        time,
        "payment",
        entities,
        {"amount": _number(row["TX_AMOUNT"], "TX_AMOUNT", nonnegative=True)},
    )
    used = {
        "TRANSACTION_ID",
        "TX_TIME_SECONDS",
        "CUSTOMER_ID",
        "TERMINAL_ID",
        "TX_AMOUNT",
        "TX_FRAUD",
    }
    diagnostics = {key: value for key, value in row.items() if key not in used}
    return event, Outcome(
        identifier, _label(row.get("TX_FRAUD")), diagnostics=diagnostics
    )


def _ulb(row, generated_id):
    time = _number(row["Time"], "Time", nonnegative=True)
    features = {"amount": _number(row["Amount"], "Amount", nonnegative=True)}
    features.update(
        {f"V{index}": _number(row[f"V{index}"], f"V{index}") for index in range(1, 29)}
    )
    used = {"Time", "Amount", "Class", *[f"V{index}" for index in range(1, 29)]}
    return (
        TransactionEvent(generated_id, time, "payment", (), features),
        Outcome(
            generated_id,
            _label(row.get("Class")),
            diagnostics={key: value for key, value in row.items() if key not in used},
        ),
    )


def _paysim(row, generated_id):
    step = _number(row["step"], "step", nonnegative=True)
    if not step.is_integer():
        raise ValueError("PaySim step must be an integer number of hours.")
    time = _number(step * 3600, "PaySim time", nonnegative=True)
    kind = row["type"].strip().upper().replace("-", "_")
    if kind not in ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"):
        raise ValueError("Unknown PaySim transaction type: " + kind)
    features = {
        "amount": _number(row["amount"], "amount", nonnegative=True),
        "oldbalanceOrg": _number(
            row["oldbalanceOrg"], "oldbalanceOrg", nonnegative=True
        ),
        "oldbalanceDest": _number(
            row["oldbalanceDest"], "oldbalanceDest", nonnegative=True
        ),
    }
    entities = (
        EntityRef("account", _identity(row["nameOrig"], "nameOrig"), "source"),
        EntityRef("account", _identity(row["nameDest"], "nameDest"), "destination"),
    )
    used = {
        "step",
        "type",
        "amount",
        "nameOrig",
        "nameDest",
        "oldbalanceOrg",
        "oldbalanceDest",
        "isFraud",
    }
    diagnostics = {key: value for key, value in row.items() if key not in used}
    for name in ("newbalanceOrig", "newbalanceDest"):
        if name in row:
            diagnostics[name] = _number(row[name], name, nonnegative=True)
    if "isFlaggedFraud" in row:
        diagnostics["isFlaggedFraud"] = _label(row["isFlaggedFraud"])
    return TransactionEvent(generated_id, time, kind, entities, features), Outcome(
        generated_id, _label(row.get("isFraud")), diagnostics=diagnostics
    )


ADAPTERS = {
    "handbook": {
        "parse": _handbook,
        "required": (
            "TRANSACTION_ID",
            "TX_TIME_SECONDS",
            "CUSTOMER_ID",
            "TERMINAL_ID",
            "TX_AMOUNT",
        ),
        "feature_names": ["amount"],
        "entity_types": ["customer", "terminal"],
        "source_unit": "seconds",
        "source_refs": [HANDBOOK_SOURCE],
        "event_identity": "source-transaction-id",
    },
    "ulb": {
        "parse": _ulb,
        "required": ("Time", "Amount", *[f"V{index}" for index in range(1, 29)]),
        "feature_names": ["amount", *[f"V{index}" for index in range(1, 29)]],
        "entity_types": [],
        "source_unit": "seconds",
        "source_refs": [ULB_SOURCE],
        "event_identity": "source-file-position-and-row",
    },
    "paysim": {
        "parse": _paysim,
        "required": (
            "step",
            "type",
            "amount",
            "nameOrig",
            "nameDest",
            "oldbalanceOrg",
            "oldbalanceDest",
        ),
        "feature_names": ["amount", "oldbalanceOrg", "oldbalanceDest"],
        "entity_types": ["account"],
        "source_unit": "hours",
        "source_refs": [PAYSIM_SOURCE],
        "event_identity": "source-file-position-and-row",
    },
}


def source_catalog():
    """Return supported adapters without opening data or downloading files."""
    return [
        {
            "dataset_id": name,
            "source_refs": list(adapter["source_refs"]),
            "entity_types": list(adapter["entity_types"]),
            "feature_names": list(adapter["feature_names"]),
        }
        for name, adapter in ADAPTERS.items()
    ]


def _paths(config, base):
    base = Path(base or ".").resolve()
    if ("path" in config) == ("paths" in config):
        raise ValueError("Provide exactly one of path or paths for source CSV data.")
    values = config.get("paths", [config.get("path")])
    if not isinstance(values, list) or not values:
        raise ValueError("paths must be a nonempty list of CSV files.")
    paths = []
    for value in values:
        if not isinstance(value, (str, Path)) or not str(value):
            raise ValueError("Source paths must be nonempty filenames.")
        path = Path(value)
        path = (path if path.is_absolute() else base / path).resolve()
        if path.is_dir() and "paths" not in config:
            paths.extend(sorted(path.glob("*.csv")))
        elif path.is_file():
            paths.append(path)
        else:
            raise ValueError("Dataset CSV does not exist: " + str(path))
    if not paths:
        raise ValueError(
            "Source directory contains no CSV files; export Handbook pickle partitions to CSV first."
        )
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate source file paths.")
    for path in paths:
        if path.suffix.lower() != ".csv":
            raise ValueError(
                "Source adapters accept CSV files; export pickle partitions to CSV first."
            )
    return paths


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path, required):
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames
        if (
            not fields
            or any(not name for name in fields)
            or len(set(fields)) != len(fields)
        ):
            raise ValueError(
                str(path) + ": CSV requires nonempty, unique column names."
            )
        missing = set(required) - set(fields)
        if missing:
            raise ValueError(
                str(path) + ": Missing CSV columns: " + ", ".join(sorted(missing))
            )
        for index, row in enumerate(reader, 1):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(
                    f"{path}: row {index} has a different width from the header."
                )
            yield index, row


def _validate_record(event, outcome, descriptor):
    """Enforce invariants once for all adapters, including future additions."""
    if not isinstance(event, TransactionEvent) or not isinstance(outcome, Outcome):
        raise ValueError("An adapter must return a TransactionEvent and an Outcome.")
    if not isinstance(event.event_id, str) or not event.event_id.strip():
        raise ValueError("Event identity must be a nonempty string.")
    if outcome.event_id != event.event_id:
        raise ValueError("Outcome identity must match its event identity.")
    if not isinstance(event.event_type, str) or not event.event_type.strip():
        raise ValueError("Event type must be a nonempty string.")
    if isinstance(event.event_time, bool) or not isinstance(
        event.event_time, (int, float)
    ):
        raise ValueError("Event time must be numeric seconds.")
    _number(event.event_time, "Event time", nonnegative=True)
    if outcome.available_at is not None:
        if outcome.label == -1:
            raise ValueError("Unknown outcomes cannot have a label availability time.")
        if isinstance(outcome.available_at, bool) or not isinstance(
            outcome.available_at, (int, float)
        ):
            raise ValueError("Label availability must be numeric seconds.")
        if _number(outcome.available_at, "Label availability") < event.event_time:
            raise ValueError("A label cannot be available before its transaction.")
    if set(event.features) != set(descriptor["feature_names"]):
        raise ValueError("Event features must match the declared feature_names.")
    for name, value in event.features.items():
        if (
            not isinstance(name, str)
            or not name
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise ValueError(
                "Declared features must have nonempty names and finite numeric values."
            )
        _number(value, name)
    roles = []
    for entity in event.entities:
        if (
            not isinstance(entity, EntityRef)
            or entity.kind not in descriptor["entity_types"]
        ):
            raise ValueError("Event entities must use declared entity types.")
        if (
            not isinstance(entity.id, str)
            or not entity.id.strip()
            or not isinstance(entity.role, str)
            or not entity.role
        ):
            raise ValueError("Entity identities and roles must be nonempty strings.")
        roles.append(entity.role)
    graph = descriptor["graph"]
    if graph and (
        roles.count(graph["source_role"]) != 1
        or roles.count(graph["destination_role"]) != 1
    ):
        raise ValueError(
            "Graph events require exactly one source and one destination role."
        )
    if not graph and roles:
        raise ValueError(
            "An adapter without graph entities cannot emit entity references."
        )


def _plain(value):
    """Turn immutable record mappings into JSON values without losing nesting."""
    from collections.abc import Mapping

    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _prepare(config, base, output, *, temporary):
    dataset_id = config.get("dataset")
    if dataset_id not in ADAPTERS:
        raise ValueError(
            "Unknown source dataset: "
            + str(dataset_id)
            + "; choose "
            + ", ".join(ADAPTERS)
        )
    adapter = ADAPTERS[dataset_id]
    paths = _paths(config, base)
    currency = config.get("currency")
    if currency is not None and (not isinstance(currency, str) or not currency.strip()):
        raise ValueError(
            "currency must be a nonempty explicit unit or null for unknown."
        )
    hashes = [_digest(path) for path in paths]
    source_hash = (
        hashes[0]
        if len(hashes) == 1
        else hashlib.sha256(json.dumps(hashes).encode()).hexdigest()
    )
    configuration = json.loads(json.dumps(config))
    descriptor = {
        "dataset_id": dataset_id,
        "version": str(config.get("release", "unspecified")),
        "name": config.get("name", dataset_id),
        "adapter_version": ADAPTER_VERSION,
        "clock": {
            "kind": "relative",
            "unit": "seconds",
            "source_unit": adapter["source_unit"],
            "timezone": None,
        },
        "entity_types": list(adapter["entity_types"]),
        "feature_names": list(adapter["feature_names"]),
        "graph": {"source_role": "source", "destination_role": "destination"}
        if adapter["entity_types"]
        else None,
        "currency": currency,
        "label_availability": "retrospective",
        "label_semantics": {"fraud": 1, "legitimate": 0, "unknown": -1},
        "event_identity": adapter["event_identity"],
        "ordering": {
            "primary": "event_time",
            "ties": "source-file-order-then-row",
            "source_order_is_causal": False,
        },
        "source_refs": list(adapter["source_refs"]),
    }
    provenance = {
        "loader": config.get("loader", "transaction_source"),
        "dataset_id": dataset_id,
        "source_name": paths[0].name
        if len(paths) == 1
        else f"{dataset_id} ({len(paths)} CSV files)",
        "source_sha256": source_hash,
        "configuration": configuration,
        "configuration_sha256": hashlib.sha256(
            json.dumps(configuration, sort_keys=True).encode()
        ).hexdigest(),
        "adapter_version": ADAPTER_VERSION,
        "source_refs": list(adapter["source_refs"]),
        "source_files": [
            {"name": path.name, "sha256": digest} for path, digest in zip(paths, hashes)
        ],
        "origin": "user-supplied",
        "dataset_descriptor": descriptor,
    }
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive reservation protects existing data, including an empty file.
    try:
        with output.open("xb"):
            pass
    except FileExistsError:
        raise ValueError(
            "Prepared dataset output already exists: " + str(output)
        ) from None
    database = None
    try:
        database = sqlite3.connect(output)
        database.execute("PRAGMA temp_store = FILE")
        database.executescript(
            """
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE events (
                event_id TEXT PRIMARY KEY, event_time REAL NOT NULL,
                event_type TEXT NOT NULL, entities TEXT NOT NULL,
                features TEXT NOT NULL, arrival_seq INTEGER NOT NULL UNIQUE);
            CREATE TABLE outcomes (
                event_id TEXT PRIMARY KEY REFERENCES events(event_id),
                label INTEGER NOT NULL CHECK (label IN (-1,0,1)),
                available_at REAL, diagnostics TEXT NOT NULL);
        """
        )
        sequence = known_labels = recorded_labels = 0
        for file_index, path in enumerate(paths):
            for row_number, row in _rows(path, adapter["required"]):
                generated_id = f"{dataset_id}:{file_index}:{row_number}"
                try:
                    event, outcome = adapter["parse"](row, generated_id)
                    _validate_record(event, outcome, descriptor)
                    database.execute(
                        "INSERT INTO events VALUES (?,?,?,?,?,?)",
                        (
                            event.event_id,
                            event.event_time,
                            event.event_type,
                            json.dumps([asdict(entity) for entity in event.entities]),
                            json.dumps(_plain(event.features), allow_nan=False),
                            sequence,
                        ),
                    )
                    database.execute(
                        "INSERT INTO outcomes VALUES (?,?,?,?)",
                        (
                            outcome.event_id,
                            outcome.label,
                            outcome.available_at,
                            json.dumps(_plain(outcome.diagnostics), allow_nan=False),
                        ),
                    )
                except sqlite3.IntegrityError as error:
                    raise ValueError(
                        f"{path}: row {row_number}: duplicate event ID {event.event_id!r}."
                    ) from error
                except ValueError as error:
                    raise ValueError(f"{path}: row {row_number}: {error}") from error
                sequence += 1
                known_labels += int(outcome.label >= 0)
                recorded_labels += int(
                    outcome.label >= 0 and outcome.available_at is not None
                )
            # Fingerprint must describe the content actually consumed.
            if _digest(path) != hashes[file_index]:
                raise ValueError("Source CSV changed during preparation: " + str(path))
        if not sequence:
            raise ValueError("Source dataset contains no transactions.")
        descriptor["label_availability"] = (
            "as-of"
            if known_labels and recorded_labels == known_labels
            else "mixed"
            if recorded_labels
            else "retrospective"
        )
        database.execute(
            "CREATE INDEX chronological_events ON events(event_time,arrival_seq)"
        )
        database.executemany(
            "INSERT INTO metadata VALUES (?,?)",
            [
                ("schema", json.dumps(SCHEMA)),
                ("descriptor", json.dumps(descriptor)),
                ("provenance", json.dumps(provenance)),
            ],
        )
        database.commit()
        database.close()
        database = None
        return (
            TransactionStream(output, temporary=True)
            if temporary
            else open_prepared(output)
        )
    except BaseException:
        if database is not None:
            database.close()
        output.unlink(missing_ok=True)
        raise


def open_source(config, base=None):
    """Prepare source CSVs into a temporary, ordered store and open it."""
    handle, path = tempfile.mkstemp(prefix="payment-data-", suffix=".sqlite")
    import os

    os.close(handle)
    Path(path).unlink()
    return _prepare(config, base, path, temporary=True)


def prepare_source(config, base, output):
    """Prepare a reusable SQLite file without replacing an existing output."""
    return _prepare(config, base, output, temporary=False)


def open_prepared(path):
    """Open a previously prepared store without rereading the source CSVs."""
    stream = TransactionStream(path)
    stream.provenance = {**stream.provenance, "prepared_sha256": _digest(stream.path)}
    return stream
