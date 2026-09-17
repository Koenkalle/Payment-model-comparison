"""Browser access to registered datasets and bounded CSV uploads.

Only startup configuration chooses server-side files. HTTP clients select an
opaque catalog ID or submit CSV contents; they cannot request filesystem paths.
"""
import json
from pathlib import Path
import tempfile
import threading

from datasets.adapters import source_catalog
from datasets.views import _selection, payments_view
from .registry import load_dataset


ROOT = Path(__file__).resolve().parents[1]
MAX_UPLOAD_BYTES = 6 * 1024 * 1024  # Leave room for JSON escaping in the HTTP envelope.
MAX_EVENTS = 20000
MAX_ACCOUNTS = 256
LABELS = {
    "handbook": "Fraud Detection Handbook",
    "paysim": "PaySim",
    "ulb": "ULB credit-card fraud",
}


class DatasetService:
    def __init__(self, config_paths=()):
        self._lock = threading.Lock()
        self._configured = {}
        paths = [
            ("handbook-demo", ROOT / "examples/datasets/handbook-payments.config.json")
        ]
        paths += [
            (f"local:{index + 1}", Path(path).resolve())
            for index, path in enumerate(config_paths)
        ]
        for identifier, path in paths:
            config = json.loads(path.read_text())
            if not isinstance(config, dict) or config.get("loader") not in (
                "fraud_dataset",
                "prepared_fraud",
            ):
                raise ValueError(
                    "Web datasets require fraud_dataset or prepared_fraud configurations."
                )
            self._configured[identifier] = (config, path.parent)

    def catalog(self):
        sources = {entry["dataset_id"]: entry for entry in source_catalog()}
        entries = []
        for identifier, (config, _) in self._configured.items():
            source = sources.get(config.get("dataset"))
            supported = source is None or bool(source["entity_types"])
            entries.append(
                {
                    "id": identifier,
                    "label": config.get("name", identifier),
                    "upload": False,
                    "supported": supported,
                    "requires_conversion": config.get("currency") != "EUR"
                    and "amount_to_eur" not in config,
                    "selection": config.get("selection", {}),
                    "description": (
                        "Invented 48-transaction fixture for trying the comparison; not a real benchmark. One fixture unit is treated as one EUR."
                        if identifier == "handbook-demo"
                        else "Registered local dataset. Load a time interval within the comparison limits."
                        if supported
                        else "Tabular only: this source has no payment endpoints for the comparison."
                    ),
                }
            )
        for identifier, source in sources.items():
            supported = bool(source["entity_types"])
            entries.append(
                {
                    "id": "csv:" + identifier,
                    "label": LABELS.get(identifier, identifier) + " CSV",
                    "upload": True,
                    "supported": supported,
                    "requires_conversion": True,
                    "description": (
                        "Choose a CSV and specify how its amounts convert to EUR. "
                        + (
                            "PaySim selection times are step × 3,600 seconds."
                            if identifier == "paysim"
                            else "Selection times use source-relative seconds."
                        )
                        if supported
                        else "ULB has no account or merchant identities. Use its numeric view with the Python experiment runner; it cannot populate this payment comparison."
                    ),
                }
            )
        return {
            "datasets": entries,
            "max_upload_bytes": MAX_UPLOAD_BYTES,
            "max_events": MAX_EVENTS,
            "max_accounts": MAX_ACCOUNTS,
        }

    def load(self, payload):
        if not isinstance(payload, dict) or set(payload) - {
            "id",
            "csv",
            "name",
            "selection",
            "amount_to_eur",
            "release",
        }:
            raise ValueError(
                "Choose a catalog dataset ID, optional CSV contents, interval and conversion."
            )
        identifier = payload.get("id")
        entries = {entry["id"]: entry for entry in self.catalog()["datasets"]}
        if not isinstance(identifier, str) or identifier not in entries:
            raise ValueError("Unknown dataset ID.")
        entry = entries[identifier]
        if not entry["supported"]:
            raise ValueError(entry["description"])
        if not entry["upload"] and set(payload) & {"csv", "name", "release"}:
            raise ValueError(
                "CSV contents and source names apply only to upload entries."
            )
        with self._lock, tempfile.TemporaryDirectory(
            prefix="comparison-dataset-"
        ) as temporary:
            if entry["upload"]:
                csv = payload.get("csv")
                if (
                    not isinstance(csv, str)
                    or not csv
                    or len(csv.encode("utf-8")) > MAX_UPLOAD_BYTES
                ):
                    raise ValueError(
                        "Upload a nonempty CSV of at most 6 MiB, or register a larger local dataset with --dataset-config."
                    )
                for key in ("name", "release"):
                    if key in payload and (
                        not isinstance(payload[key], str)
                        or not 0 < len(payload[key]) <= 200
                    ):
                        raise ValueError(
                            key
                            + " must be a nonempty string of at most 200 characters."
                        )
                base = Path(temporary)
                (base / "upload.csv").write_text(csv, encoding="utf-8")
                config = {
                    "loader": "fraud_dataset",
                    "dataset": identifier[4:],
                    "path": "upload.csv",
                    "name": payload.get(
                        "name", LABELS.get(identifier[4:], identifier[4:])
                    ),
                    "release": payload.get("release", "unspecified"),
                }
            else:
                original, base = self._configured[identifier]
                config = dict(original)
            for key in ("selection", "amount_to_eur"):
                if key in payload:
                    config[key] = payload[key]
            selection = _selection(config)
            source_config = {
                key: value
                for key, value in config.items()
                if key not in ("selection", "features")
            }
            source_config["view"] = "stream"
            with load_dataset(source_config, base) as stream:
                # Reject oversized selections before building model/browser arrays.
                count, entities = 0, set()
                for batch in stream.iter_batches(512, **selection):
                    for event in batch:
                        count += 1
                        entities.update(
                            (entity.kind, entity.id) for entity in event.entities
                        )
                        if count > MAX_EVENTS or len(entities) > MAX_ACCOUNTS:
                            raise ValueError(
                                "This interval exceeds 20,000 events or 256 accounts. Select a smaller time interval; no rows were silently dropped."
                            )
                document = payments_view(
                    stream,
                    {key: value for key, value in config.items() if key != "features"},
                ).document
                known = len(document["truth"])
                return {
                    "dataset": document,
                    "summary": {
                        "events": count,
                        "accounts": len(document["accounts"]),
                        "known": known,
                        "unknown": count - known,
                        "source": stream.descriptor["dataset_id"],
                        "release": stream.descriptor["version"],
                    },
                }
