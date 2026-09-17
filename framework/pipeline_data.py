"""Durable, inspectable datasets shared by the web generation/training pipeline.

HTTP callers choose opaque IDs or submit data, never local filenames. Registered
server configurations are snapshotted at startup and remain usable independently
of the original files. A dataset is published only after its payload validates.
"""
from contextlib import contextmanager
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil
import threading
import uuid

import numpy as np

from datasets.adapters import source_catalog
from datasets.feature_info import describe_feature, source_definition
from datasets.implementations.payment_json import normalize
from datasets.implementations.dataset_features import read_numeric, write_numeric
from datasets.stream import open_prepared, prepare_source
from datasets.views import _selection
from framework.dataset_service import MAX_ACCOUNTS, MAX_EVENTS, MAX_UPLOAD_BYTES
from framework.registry import ROOT, load_dataset
from framework.contracts import NumericDataset


_ID = re.compile(r"ds-[0-9a-f]{32}\Z")
_LABELS = {
    "handbook": "Fraud Detection Handbook",
    "ulb": "ULB credit-card fraud",
    "paysim": "PaySim",
}
_REPLAY_NOTE = (
    "Selected columns are shared by numeric models and graph edge attributes. "
    "Payment replay retains the original events and amounts; browser policies use their own event inputs."
)
_GENERATORS = [
    {
        "id": "synthetic_payments",
        "label": "Payment scenarios",
        "description": "Deterministic EUR payment scenarios with background commerce, concentrated fraud attempts and delayed reports. These are replay scenarios: a chronological validation interval may contain only legitimate payments. Use a fixed probability cutoff in the trainer when validation lacks both classes; benign-only scenarios cannot train a supervised fraud model.",
        "parameters": [
            {
                "name": "name",
                "label": "Scenario",
                "type": "select",
                "default": "mixed",
                "options": [
                    {"value": value, "label": label}
                    for value, label in (
                        ("mixed", "Mixed week with a traffic surge"),
                        ("relay", "Collection and long relay"),
                        ("split", "Split, merge, and cycle"),
                        ("takeover", "Account takeover"),
                        ("benign", "Busy merchants and payroll"),
                    )
                ],
            },
            {
                "name": "size",
                "label": "Size",
                "type": "select",
                "default": "medium",
                "options": [
                    {"value": value, "label": value.title()}
                    for value in ("small", "medium", "large")
                ],
            },
            {
                "name": "seed",
                "label": "Random seed",
                "type": "integer",
                "default": 42,
                "min": 0,
                "max": 4294967295,
            },
            {
                "name": "reportDelay",
                "label": "Report delay (minutes)",
                "type": "number",
                "default": 1440,
                "min": 0,
                "max": 525600,
            },
            {
                "name": "forwardDelay",
                "label": "Forwarding delay (minutes)",
                "type": "number",
                "default": 2,
                "min": 0,
                "max": 1440,
            },
        ],
    },
    {
        "id": "handbook_generator",
        "label": "Configurable customer and terminal payments",
        "description": "Invented EUR transactions in the Handbook CSV schema, with adjustable population and fraud prevalence. This is an integration and simulation generator, not a reproduction of the Handbook benchmark.",
        "parameters": [
            {
                "name": "customer_count",
                "label": "Customers",
                "type": "integer",
                "default": 24,
                "min": 2,
                "max": 10000,
            },
            {
                "name": "terminal_count",
                "label": "Terminals",
                "type": "integer",
                "default": 12,
                "min": 1,
                "max": 10000,
            },
            {
                "name": "transactions",
                "label": "Transactions",
                "type": "integer",
                "default": 1000,
                "min": 20,
                "max": 100000,
            },
            {
                "name": "duration_days",
                "label": "Duration (days)",
                "type": "number",
                "default": 14,
                "min": 0.01,
                "max": 3650,
            },
            {
                "name": "fraud_rate",
                "label": "Fraud fraction",
                "type": "number",
                "default": 0.1,
                "min": 0,
                "max": 1,
            },
            {
                "name": "seed",
                "label": "Random seed",
                "type": "integer",
                "default": 42,
                "min": 0,
                "max": 4294967295,
            },
        ],
    },
]
# The configurable generator distributes fraud throughout time and is the
# useful default for the complete generation, fitting and evaluation pipeline.
_GENERATORS.sort(key=lambda entry: entry["id"] != "handbook_generator")


def _json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_payments(document, bounds):
    """Preserve the source clock and identities while selecting a stored interval."""
    if not bounds:
        return document
    document = copy.deepcopy(document)
    document["events"] = [
        event
        for event in document["events"]
        if ("start" not in bounds or event["t"] * 60 >= bounds["start"])
        and ("stop" not in bounds or event["t"] * 60 < bounds["stop"])
    ]
    if not document["events"]:
        raise ValueError("Dataset selection is empty.")
    ids = {event["id"] for event in document["events"] if event["kind"] == "payment"}
    document["truth"] = {
        key: value for key, value in document["truth"].items() if key in ids
    }
    document["label_available_at"] = {
        key: value
        for key, value in document.get("label_available_at", {}).items()
        if key in ids
    }
    return document


def _name(value, field="name"):
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise ValueError(
            field + " must be a nonempty string of at most 200 characters."
        )
    return value.strip()


def _conversion(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("amount_to_eur must be a finite positive conversion factor.")
    return value


def _public_provenance(value):
    """Configurations are inspectable without disclosing server filesystem paths."""
    if isinstance(value, dict):
        return {
            key: _public_provenance(item)
            for key, item in value.items()
            if key not in ("path", "paths", "python_module", "registration_key")
        }
    if isinstance(value, list):
        return [_public_provenance(item) for item in value]
    return value


def _statistics(values):
    """Population statistics across this snapshot, including unlabeled rows."""
    finite = np.asarray(values)[np.isfinite(values)]
    count = len(finite)
    result = {
        "count": count,
        "missing": len(values) - count,
        "unique": int(len(np.unique(finite))),
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
        "quantiles": {"p25": None, "p50": None, "p75": None},
        "histogram": [],
    }
    if not count:
        return result
    precise = finite.astype(np.longdouble)

    def number(value):
        value = float(value)
        return value if math.isfinite(value) else None

    result.update(
        min=float(finite.min()),
        max=float(finite.max()),
        mean=number(precise.mean()),
        std=number(precise.std()),
    )
    result["quantiles"] = {
        key: number(value)
        for key, value in zip(
            ("p25", "p50", "p75"), np.quantile(precise, [0.25, 0.5, 0.75])
        )
    }
    lower, upper = precise.min(), precise.max()
    if lower == upper:
        result["histogram"] = [
            {"min": float(lower), "max": float(upper), "count": count}
        ]
    else:
        bins = min(10, result["unique"])
        indices = np.minimum(
            ((precise - lower) / (upper - lower) * bins).astype(np.int64), bins - 1
        )
        counts = np.bincount(indices, minlength=bins)
        edges = lower + np.linspace(0, 1, bins + 1, dtype=np.longdouble) * (
            upper - lower
        )
        result["histogram"] = [
            {
                "min": number(edges[index]),
                "max": number(edges[index + 1]),
                "count": int(counts[index]),
            }
            for index in range(bins)
        ]
    return result


class DatasetStore:
    def __init__(self, root, config_paths=()):
        self.root = Path(root).resolve()
        self.directory = self.root / "datasets"
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._bootstrap(
            "handbook-fixture-v1",
            ROOT / "examples/datasets/handbook-payments.config.json",
        )
        for path in config_paths:
            path = Path(path).resolve()
            self._bootstrap("config-" + _hash(str(path)), path)

    def _directory(self, identifier):
        if not isinstance(identifier, str) or not _ID.fullmatch(identifier):
            raise ValueError("Unknown dataset ID.")
        path = self.directory / identifier
        if (
            path.is_symlink()
            or not path.is_dir()
            or not (path / "metadata.json").is_file()
        ):
            raise ValueError("Unknown dataset ID.")
        return path

    @contextmanager
    def _new(self):
        """Publish a complete snapshot in one directory rename; clean failures."""
        identifier = "ds-" + uuid.uuid4().hex
        temporary = self.directory / (".creating-" + identifier)
        temporary.mkdir()
        try:
            yield identifier, temporary
            temporary.rename(self.directory / identifier)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def generators(self):
        return copy.deepcopy(_GENERATORS)

    def catalog(self):
        entries = []
        with self._lock:
            for path in self.directory.glob("ds-*"):
                if (
                    _ID.fullmatch(path.name)
                    and not path.is_symlink()
                    and (path / "metadata.json").is_file()
                ):
                    entries.append(self.describe(self.get(path.name)))
        sources = [
            {
                "id": source["dataset_id"],
                "label": _LABELS[source["dataset_id"]] + " CSV",
                "description": (
                    "Numeric transactions with 28 anonymized features; no payment account identities."
                    if not source["entity_types"]
                    else "Store a CSV snapshot. Supply its amount-to-EUR conversion to enable payment replay and fraud graph training."
                ),
                "requires_conversion": bool(source["entity_types"]),
                "views": ["numeric", "graph", "labeled_graph", "payments"]
                if source["entity_types"]
                else ["numeric"],
            }
            for source in source_catalog()
        ]
        return {
            "datasets": sorted(
                entries, key=lambda entry: entry["created_at"], reverse=True
            ),
            "generators": self.generators(),
            "sources": sources,
            "limits": {
                "max_upload_bytes": MAX_UPLOAD_BYTES,
                "max_events": MAX_EVENTS,
                "max_accounts": MAX_ACCOUNTS,
            },
        }

    def get(self, identifier):
        path = self._directory(identifier)
        return json.loads((path / "metadata.json").read_text(encoding="utf-8"))

    def describe(self, metadata):
        """Add display definitions without calculating features or loading arrays.

        Saved datasets use their pinned manifest, including older recipe versions.
        A damaged help catalog must not prevent listing the rest of the library;
        normal feature loading still validates the complete cache and its values.
        """
        if metadata.get("kind") != "features":
            definitions = [
                source_definition(name, metadata.get("currency"))
                for name in metadata.get("feature_names", [])
            ]
        else:
            catalog = {}
            cache = self._directory(metadata["id"]) / (
                ".features-" + _hash(metadata["feature_recipe_version"])[:16]
            )
            try:
                if cache.is_symlink() or any(
                    (cache / name).is_symlink()
                    for name in ("manifest.json", "checksums.json")
                ):
                    raise ValueError("Invalid feature documentation cache.")
                checksums = json.loads(
                    (cache / "checksums.json").read_text(encoding="utf-8")
                )
                if _file_hash(cache / "manifest.json") != checksums["manifest_sha256"]:
                    raise ValueError("Feature documentation checksum changed.")
                manifest = json.loads(
                    (cache / "manifest.json").read_text(encoding="utf-8")
                )
                if manifest["recipe_version"] != metadata["feature_recipe_version"]:
                    raise ValueError("Feature documentation version changed.")
                if manifest["source_fingerprint"] != metadata["source_fingerprint"]:
                    raise ValueError("Feature documentation source changed.")
                catalog = {feature["id"]: feature for feature in manifest["features"]}
            except (OSError, ValueError, KeyError, TypeError):
                pass
            definitions = []
            for name in metadata.get("feature_names", []):
                if name in catalog:
                    definitions.append(describe_feature(catalog[name]))
                else:
                    definitions.append(
                        {
                            "id": name,
                            "label": name,
                            "kind": "derived",
                            "description": f"Stored feature “{name}”. Its original definition is unavailable, so its meaning cannot be verified from this snapshot.",
                            "info": {
                                "sections": [
                                    {
                                        "title": "Saved definition unavailable",
                                        "text": "The original feature documentation could not be read. Open this dataset "
                                        "in Data lab to inspect its stored recipe. No current recipe has been substituted.",
                                    }
                                ]
                            },
                        }
                    )
        return {**metadata, "feature_definitions": definitions}

    def _config(self, identifier):
        path = self._directory(identifier)
        metadata = self.get(identifier)
        if (
            metadata.get("config_sha256")
            and _file_hash(path / "config.json") != metadata["config_sha256"]
        ):
            raise ValueError(
                "Stored dataset configuration checksum changed. Import a new dataset snapshot."
            )
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        filename = config["path"]
        # Internal snapshots use only flat filenames generated by this store.
        if (
            filename not in ("transactions.sqlite", "payments.json")
            or (path / filename).is_symlink()
        ):
            raise ValueError("Invalid stored dataset payload.")
        if (
            metadata.get("payload_sha256")
            and _file_hash(path / filename) != metadata["payload_sha256"]
        ):
            raise ValueError(
                "Stored dataset payload checksum changed. Import a new dataset snapshot."
            )
        config["path"] = str(path / filename)
        return config

    def source_config(self, identifier, view="numeric"):
        """The original observable data, independent of selected model columns."""
        metadata = self.get(identifier)
        if view not in metadata["views"]:
            raise ValueError(
                "Dataset does not support the "
                + str(view)
                + " view."
                + (
                    " Supply an explicit amount-to-EUR conversion when importing this dataset."
                    if metadata.get("requires_conversion")
                    and view in ("payments", "labeled_graph")
                    else ""
                )
            )
        config = self._config(identifier)
        if config["loader"] == "prepared_fraud":
            config["view"] = view
            if view != "numeric":
                config.pop("features", None)
            return config
        if view == "payments":
            return config
        if view == "graph":
            return {"loader": "payment_graph", "dataset": config}
        return {"loader": "pipeline_numeric", "view": view, "dataset": config}

    def dataset_config(self, identifier, view="numeric"):
        source = self.source_config(identifier, view)
        metadata = self.get(identifier)
        if metadata.get("kind") != "features" or view == "payments":
            return source
        path = self._directory(identifier) / "features.npz"
        if path.is_symlink() or not path.is_file():
            raise ValueError("Invalid stored feature payload.")
        if _file_hash(path) != metadata["feature_payload_sha256"]:
            raise ValueError(
                "Stored feature payload checksum changed. Create a new dataset snapshot."
            )
        return {
            "loader": "dataset_features",
            "view": view,
            "path": str(path),
            "payload_sha256": metadata["feature_payload_sha256"],
            "dataset": source,
            "feature_names": metadata["feature_names"],
            "recipe_version": metadata["feature_recipe_version"],
            "fingerprint": metadata["fingerprint"],
        }

    def _feature_cache(self, identifier):
        """Calculate all columns once per original snapshot and recipe version."""
        from datasets.features import RECIPE_VERSION, prepare_features

        metadata = self.get(identifier)
        config = self._config(identifier)
        source_id = metadata.get("source_dataset_id", identifier)
        source_fingerprint = metadata.get("source_fingerprint", metadata["fingerprint"])
        recipe_version = metadata.get("feature_recipe_version", RECIPE_VERSION)
        directory = self._directory(identifier)
        cache = directory / (".features-" + _hash(recipe_version)[:16])
        if cache.is_symlink():
            raise ValueError("Invalid stored feature cache.")
        if not cache.exists():
            if metadata.get("kind") == "features":
                raise ValueError(
                    "This saved dataset is missing its original feature recipe cache. Restore the dataset snapshot."
                )
            temporary = directory / (".calculating-features-" + uuid.uuid4().hex)
            temporary.mkdir()
            try:
                numeric, definitions = prepare_features(config, directory)
                numeric = NumericDataset(
                    numeric.ids,
                    numeric.times,
                    numeric.features,
                    numeric.labels,
                    numeric.feature_names,
                    _public_provenance(numeric.provenance),
                )
                write_numeric(temporary / "columns.npz", numeric)
                columns = {
                    name: index for index, name in enumerate(numeric.feature_names)
                }
                manifest = {
                    "source_dataset_id": source_id,
                    "source_fingerprint": source_fingerprint,
                    "recipe_version": recipe_version,
                    "row_count": len(numeric.ids),
                    "features": [
                        {
                            **definition,
                            "statistics": _statistics(
                                numeric.features[:, columns[definition["id"]]]
                            )
                            if definition["available"]
                            else None,
                        }
                        for definition in definitions
                    ],
                }
                _json(temporary / "manifest.json", manifest)
                _json(
                    temporary / "checksums.json",
                    {
                        "manifest_sha256": _file_hash(temporary / "manifest.json"),
                        "payload_sha256": _file_hash(temporary / "columns.npz"),
                    },
                )
                try:
                    temporary.rename(cache)
                except OSError:
                    if not cache.is_dir() or cache.is_symlink():
                        raise
                    # Another service process may have published the same recipe.
                    shutil.rmtree(temporary)
            except BaseException:
                shutil.rmtree(temporary, ignore_errors=True)
                raise
        for filename in ("manifest.json", "checksums.json", "columns.npz"):
            if (cache / filename).is_symlink() or not (cache / filename).is_file():
                raise ValueError("Invalid stored feature cache.")
        checksums = json.loads((cache / "checksums.json").read_text(encoding="utf-8"))
        if not isinstance(checksums, dict) or any(
            not isinstance(checksums.get(key), str)
            or not re.fullmatch(r"[0-9a-f]{64}", checksums[key])
            for key in ("manifest_sha256", "payload_sha256")
        ):
            raise ValueError("Invalid stored feature cache checksums.")
        if _file_hash(cache / "manifest.json") != checksums.get("manifest_sha256"):
            raise ValueError(
                "Stored feature cache metadata checksum changed. Create a new dataset snapshot."
            )
        manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
        if (
            manifest.get("recipe_version") != recipe_version
            or manifest.get("source_fingerprint") != source_fingerprint
        ):
            raise ValueError(
                "Stored feature cache does not match this dataset snapshot."
            )
        numeric = read_numeric(cache / "columns.npz", checksums.get("payload_sha256"))
        return numeric, manifest

    def features(self, identifier):
        from datasets.features import RECIPE_VERSION

        with self._lock:
            metadata = self.get(identifier)
            self.dataset_config(
                identifier
            )  # Also verify a saved variant's feature checksum.
            _, manifest = self._feature_cache(identifier)
            enabled = metadata["feature_names"]
            return {
                **manifest,
                "dataset_id": identifier,
                "fingerprint": metadata["fingerprint"],
                "latest_recipe_version": RECIPE_VERSION,
                "enabled_features": list(enabled),
                "replay_note": _REPLAY_NOTE,
                "features": [
                    {
                        **describe_feature(definition),
                        "enabled": definition["id"] in enabled,
                    }
                    for definition in manifest["features"]
                ],
            }

    def save_features(self, identifier, payload):
        """Publish a new dataset; existing snapshots and experiment inputs stay fixed."""
        if not isinstance(payload, dict) or set(payload) - {"features", "name"}:
            raise ValueError(
                "Feature selection accepts only features and an optional dataset name."
            )
        names = payload.get("features")
        if (
            not isinstance(names, list)
            or not names
            or not all(isinstance(name, str) for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError(
                "features must list nonempty, distinct available feature IDs."
            )
        with self._lock:
            parent = self.get(identifier)
            name = _name(payload.get("name", (parent["name"][:186] + " · features")))
            self.dataset_config(identifier)
            numeric, manifest = self._feature_cache(identifier)
            definitions = {
                definition["id"]: definition for definition in manifest["features"]
            }
            if any(
                name not in definitions or not definitions[name]["available"]
                for name in names
            ):
                raise ValueError(
                    "Select available observable feature IDs from the dataset feature catalog."
                )
            indices = [numeric.feature_names.index(name) for name in names]
            values = numeric.features[:, indices].copy()
            recipe = [
                {"id": name, "version": definitions[name]["version"]} for name in names
            ]
            fingerprint = _hash(
                {
                    "source_fingerprint": manifest["source_fingerprint"],
                    "recipe_version": manifest["recipe_version"],
                    "features": recipe,
                    "values_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
                }
            )
            selected = NumericDataset(
                numeric.ids,
                numeric.times,
                values,
                numeric.labels,
                tuple(names),
                {
                    **numeric.provenance,
                    "feature_recipe_version": manifest["recipe_version"],
                    "dataset_feature_fingerprint": fingerprint,
                    "features_stored": True,
                },
            )
            original_directory = self._directory(identifier)
            source_config = json.loads(
                (original_directory / "config.json").read_text(encoding="utf-8")
            )
            with self._new() as (new_id, destination):
                shutil.copyfile(
                    original_directory / source_config["path"],
                    destination / source_config["path"],
                )
                cache_name = ".features-" + _hash(manifest["recipe_version"])[:16]
                shutil.copytree(
                    original_directory / cache_name, destination / cache_name
                )
                _json(destination / "config.json", source_config)
                write_numeric(destination / "features.npz", selected)
                metadata = {
                    **copy.deepcopy(parent),
                    "id": new_id,
                    "name": name,
                    "kind": "features",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "parent_dataset_id": identifier,
                    "source_dataset_id": manifest["source_dataset_id"],
                    "source_fingerprint": manifest["source_fingerprint"],
                    "fingerprint": fingerprint,
                    "feature_names": list(names),
                    "feature_recipe": recipe,
                    "feature_recipe_version": manifest["recipe_version"],
                    "feature_replay_note": _REPLAY_NOTE,
                    "config_sha256": _file_hash(destination / "config.json"),
                    "feature_payload_sha256": _file_hash(destination / "features.npz"),
                    "provenance": {
                        **parent.get("provenance", {}),
                        "feature_recipe_version": manifest["recipe_version"],
                        "features_stored": True,
                        "outcomes_used_for_features": False,
                    },
                }
                _json(destination / "metadata.json", metadata)
                return metadata

    def _bootstrap(self, registration, path):
        with self._lock:
            for stored in self.directory.glob("ds-*/registration.json"):
                if json.loads(stored.read_text()).get("key") == registration:
                    if (stored.parent / "metadata.json").is_file():
                        return
            config = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(config, dict) or config.get("loader") not in (
                "fraud_dataset",
                "prepared_fraud",
                "payment_json",
            ):
                raise ValueError(
                    "Registered web datasets require fraud_dataset, prepared_fraud or payment_json configurations."
                )
            with self._new() as (identifier, destination):
                self._snapshot(
                    identifier,
                    destination,
                    config,
                    path.parent,
                    kind="fixed",
                    parameters={},
                )
                _json(destination / "registration.json", {"key": registration})

    def _snapshot(
        self, identifier, destination, config, base, *, kind, parameters, generator=None
    ):
        config = copy.deepcopy(config)
        _selection(config)
        if "amount_to_eur" in config:
            _conversion(config["amount_to_eur"])
        if config["loader"] == "payment_json":
            dataset = load_dataset(config, base)
            document = _selected_payments(dataset.document, _selection(config))
            _json(destination / "payments.json", document)
            stored = {"loader": "payment_json", "path": "payments.json"}
            metadata = self._payment_metadata(document)
            metadata["selection"] = config.get("selection", {})
        else:
            output = destination / "transactions.sqlite"
            if config["loader"] == "fraud_dataset":
                # Keep supplied CSV bytes alongside the canonical prepared store.
                from datasets.adapters import _paths

                paths = _paths(config, base)
                copied = []
                for index, path in enumerate(paths):
                    filename = f"source-{index + 1}.csv"
                    shutil.copyfile(path, destination / filename)
                    copied.append(filename)
                preparation = {
                    key: value
                    for key, value in config.items()
                    if key not in ("path", "paths", "selection")
                }
                preparation["paths"] = copied
                with prepare_source(preparation, destination, output):
                    pass
            else:
                source = Path(config["path"])
                shutil.copyfile(
                    source if source.is_absolute() else base / source, output
                )
            stored = {
                key: value
                for key, value in config.items()
                if key in ("name", "features", "selection", "amount_to_eur")
            }
            stored.update(loader="prepared_fraud", path="transactions.sqlite")
            with open_prepared(output) as stream:
                metadata = self._stream_metadata(stream, config)
        metadata.update(
            id=identifier,
            name=config.get("name", metadata.get("name", "Stored dataset")),
            kind=kind,
            created_at=datetime.now(timezone.utc).isoformat(),
            generator=generator,
            parameters=copy.deepcopy(parameters),
        )
        if generator:
            metadata["provenance"]["origin"] = "synthetic"
            metadata["provenance"]["generator"] = generator
            metadata["provenance"]["parameters"] = copy.deepcopy(parameters)
        _json(destination / "config.json", stored)
        metadata["config_sha256"] = _file_hash(destination / "config.json")
        metadata["payload_sha256"] = _file_hash(destination / stored["path"])
        _json(destination / "metadata.json", metadata)
        return metadata

    @staticmethod
    def _stream_metadata(stream, config):
        digest = hashlib.sha256()
        counts = {
            "rows": 0,
            "payment_rows": 0,
            "known": 0,
            "fraud": 0,
            "legitimate": 0,
            "unknown": 0,
        }
        entities, start, end = set(), None, None
        for batch in stream.iter_batches(512, **_selection(config)):
            truth = stream.truth_for(event.event_id for event in batch)
            for event in batch:
                outcome = truth[event.event_id]
                counts["rows"] += 1
                counts[
                    "payment_rows"
                ] += 1  # Each source transaction becomes a payment candidate.
                counts["known"] += int(outcome.label >= 0)
                counts["fraud"] += int(outcome.label == 1)
                counts["legitimate"] += int(outcome.label == 0)
                counts["unknown"] += int(outcome.label < 0)
                references = [
                    (entity.kind, entity.id, entity.role) for entity in event.entities
                ]
                entities.update((entity.kind, entity.id) for entity in event.entities)
                digest.update(
                    _hash(
                        [
                            event.event_id,
                            event.event_time,
                            event.event_type,
                            references,
                            dict(event.features),
                            outcome.label,
                            outcome.available_at,
                        ]
                    ).encode()
                )
                start = event.event_time if start is None else start
                end = event.event_time
        if not counts["rows"]:
            raise ValueError("Dataset selection is empty.")
        descriptor = stream.descriptor
        has_graph = bool(descriptor["graph"])
        conversion = config.get(
            "amount_to_eur", 1.0 if descriptor.get("currency") == "EUR" else None
        )
        views = ["numeric"] + (["graph"] if has_graph else [])
        if has_graph and conversion is not None:
            views += ["labeled_graph", "payments"]
        fingerprint = _hash(
            {
                "observations_and_labels": digest.hexdigest(),
                "features": descriptor["feature_names"],
                "currency": descriptor.get("currency"),
                "amount_to_eur": conversion,
            }
        )
        return {
            **counts,
            "name": descriptor["name"],
            "schema": stream.schema,
            "views": views,
            "source": descriptor["dataset_id"],
            "source_namespace": [descriptor["dataset_id"], descriptor["version"]],
            "accounts": len(entities),
            "time_range": {"start": start, "end": end, "unit": "seconds"},
            "feature_names": list(config.get("features", descriptor["feature_names"])),
            "currency": descriptor.get("currency"),
            "amount_to_eur": conversion,
            "requires_conversion": has_graph and conversion is None,
            "fingerprint": fingerprint,
            "selection": config.get("selection", {}),
            "provenance": _public_provenance(stream.provenance),
        }

    @staticmethod
    def _payment_metadata(document):
        payments = [event for event in document["events"] if event["kind"] == "payment"]
        if not payments:
            raise ValueError("A stored payment dataset requires payment rows.")
        truth = document["truth"]
        fingerprint = _hash(
            {
                key: document.get(key)
                for key in (
                    "accounts",
                    "events",
                    "truth",
                    "label_available_at",
                    "units",
                )
            }
        )
        return {
            "name": document["name"],
            "schema": "payment-events/v1",
            "source": "payment_json",
            "source_namespace": fingerprint,
            "views": ["numeric", "graph", "labeled_graph", "payments"],
            "rows": len(document["events"]),
            "payment_rows": len(payments),
            "known": len(truth),
            "fraud": sum(bool(value) for value in truth.values()),
            "legitimate": sum(not value for value in truth.values()),
            "unknown": len(payments) - len(truth),
            "accounts": len(document["accounts"]),
            "time_range": {
                "start": payments[0]["t"] * 60,
                "end": payments[-1]["t"] * 60,
                "unit": "seconds",
            },
            "feature_names": [
                "log1p_amount",
                "prior_source_count",
                "prior_destination_count",
            ],
            "currency": "EUR",
            "amount_to_eur": 1.0,
            "requires_conversion": False,
            "fingerprint": fingerprint,
            "selection": {},
            "provenance": _public_provenance(document.get("provenance", {})),
        }

    def generate(self, payload):
        if not isinstance(payload, dict) or set(payload) - {
            "generator",
            "name",
            "parameters",
        }:
            raise ValueError(
                "Choose a generator, optional dataset name and generator parameters."
            )
        schema = next(
            (entry for entry in _GENERATORS if entry["id"] == payload.get("generator")),
            None,
        )
        if schema is None:
            raise ValueError("Unknown generator ID.")
        supplied = payload.get("parameters", {})
        definitions = {entry["name"]: entry for entry in schema["parameters"]}
        if not isinstance(supplied, dict) or set(supplied) - set(definitions):
            raise ValueError("Unknown generator parameters.")
        parameters = {}
        for key, definition in definitions.items():
            value = supplied.get(key, definition["default"])
            if definition["type"] == "select":
                if value not in [entry["value"] for entry in definition["options"]]:
                    raise ValueError("Unsupported " + key + " option.")
            elif (
                type(value)
                not in ((int,) if definition["type"] == "integer" else (int, float))
                or not math.isfinite(value)
                or not definition["min"] <= value <= definition["max"]
            ):
                raise ValueError(
                    f'{key} must be a {definition["type"]} between {definition["min"]} and {definition["max"]}.'
                )
            parameters[key] = value
        name = _name(
            payload.get("name", schema["label"] + " · seed " + str(parameters["seed"]))
        )
        with self._lock, self._new() as (identifier, destination):
            if schema["id"] == "synthetic_payments":
                dataset = load_dataset(
                    {"loader": "synthetic_payments", "options": parameters}
                )
                document = normalize(dataset.document, dataset.provenance)
                document["name"] = name
                # Delay/scenario changes must not rename the same population or
                # hide reused background observations from overlap checks.
                namespace = (
                    "synthetic-"
                    + _hash({key: parameters[key] for key in ("seed", "size")})[:24]
                )
                for account in document["accounts"]:
                    account["external_id"] = namespace + ":" + str(account["id"])
                # Reports are observable events; only recorded fraud confirmations
                # get explicit availability times. Other labels stay retrospective.
                confirmed = {}
                for event in document["events"]:
                    reference = event.get("reference")
                    if event["kind"] == "report" and reference in document["truth"]:
                        confirmed[reference] = min(
                            confirmed.get(reference, event["t"]), event["t"]
                        )
                if confirmed:
                    document["label_available_at"] = confirmed
                _json(destination / "generated.json", document)
                config = {
                    "loader": "payment_json",
                    "path": "generated.json",
                    "name": name,
                }
            else:
                self._handbook_csv(destination / "generated.csv", parameters)
                config = {
                    "loader": "fraud_dataset",
                    "dataset": "handbook",
                    "path": "generated.csv",
                    "name": name,
                    "currency": "EUR",
                    "amount_to_eur": 1.0,
                    "release": "invented-"
                    + _hash(
                        {
                            key: parameters[key]
                            for key in ("seed", "customer_count", "terminal_count")
                        }
                    )[:24],
                }
            metadata = self._snapshot(
                identifier,
                destination,
                config,
                destination,
                kind="generator",
                parameters=parameters,
                generator=schema["id"],
            )
            (
                destination
                / (
                    "generated.json"
                    if schema["id"] == "synthetic_payments"
                    else "generated.csv"
                )
            ).unlink()
            return metadata

    @staticmethod
    def _handbook_csv(path, parameters):
        rng = random.Random(parameters["seed"])
        count = parameters["transactions"]
        frauds = set(rng.sample(range(count), round(count * parameters["fraud_rate"])))
        duration = parameters["duration_days"] * 86400
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "TRANSACTION_ID",
                    "TX_TIME_SECONDS",
                    "CUSTOMER_ID",
                    "TERMINAL_ID",
                    "TX_AMOUNT",
                    "TX_FRAUD",
                ]
            )
            for index in range(count):
                fraud = int(index in frauds)
                amount = round(math.exp(rng.gauss(4.9 if fraud else 3.3, 0.8)), 2)
                writer.writerow(
                    [
                        f"TX{index:07d}",
                        round((index + rng.random()) * duration / count, 6),
                        rng.randrange(parameters["customer_count"]),
                        rng.randrange(parameters["terminal_count"]),
                        amount,
                        fraud,
                    ]
                )

    def import_source(self, payload):
        allowed = {"source", "csv", "name", "release", "amount_to_eur"}
        if not isinstance(payload, dict) or set(payload) - allowed:
            raise ValueError(
                "Import accepts a source adapter, CSV contents, name, release and amount-to-EUR conversion."
            )
        source = payload.get("source")
        if not isinstance(source, str) or source not in _LABELS:
            raise ValueError("Choose the handbook, ulb or paysim CSV source.")
        text = payload.get("csv")
        if (
            not isinstance(text, str)
            or not text.strip()
            or len(text.encode("utf-8")) > MAX_UPLOAD_BYTES
        ):
            raise ValueError(
                "Upload a nonempty CSV of at most 6 MiB, or register a larger dataset at server startup."
            )
        config = {
            "loader": "fraud_dataset",
            "dataset": source,
            "path": "upload.csv",
            "name": _name(payload.get("name", _LABELS[source])),
            "release": _name(payload.get("release", "unspecified"), "release"),
        }
        if "amount_to_eur" in payload:
            config["amount_to_eur"] = _conversion(payload["amount_to_eur"])
        with self._lock, self._new() as (identifier, destination):
            (destination / "upload.csv").write_text(text, encoding="utf-8")
            metadata = self._snapshot(
                identifier,
                destination,
                config,
                destination,
                kind="fixed",
                parameters={},
            )
            (destination / "upload.csv").unlink()
            return metadata

    def document(self, identifier, selection=None):
        config = self.dataset_config(identifier, "payments")
        bounds = (
            _selection({"selection": selection})
            if selection is not None
            else _selection(config)
        )
        if config["loader"] == "prepared_fraud":
            config["selection"] = bounds
            # Count before materializing the browser payload, with early rejection.
            entities, count = set(), 0
            with open_prepared(config["path"]) as stream:
                for event in stream.iter_events(**bounds):
                    count += 1
                    entities.update(
                        (entity.kind, entity.id) for entity in event.entities
                    )
                    if count > MAX_EVENTS or len(entities) > MAX_ACCOUNTS:
                        raise ValueError(
                            "This interval exceeds 20,000 events or 256 accounts. Select a smaller time interval; no rows were silently dropped."
                        )
            document = load_dataset(config).document
        else:
            document = _selected_payments(load_dataset(config).document, bounds)
            active = sorted(
                {
                    event[side]
                    for event in document["events"]
                    for side in ("u", "v")
                    if event[side] >= 0
                }
            )
            if len(document["events"]) > MAX_EVENTS or len(active) > MAX_ACCOUNTS:
                raise ValueError(
                    "This interval exceeds 20,000 events or 256 accounts. Select a smaller time interval; no rows were silently dropped."
                )
            mapping = {old: index for index, old in enumerate(active)}
            document["accounts"] = [
                {**document["accounts"][old], "id": mapping[old]} for old in active
            ]
            for event in document["events"]:
                event["u"] = mapping[event["u"]] if event["u"] >= 0 else -1
                event["v"] = mapping[event["v"]]
            document = normalize(document)
        document["provenance"] = _public_provenance(document.get("provenance", {}))
        payments = sum(event["kind"] == "payment" for event in document["events"])
        return {
            "dataset": document,
            "summary": {
                "events": len(document["events"]),
                "payments": payments,
                "accounts": len(document["accounts"]),
                "known": len(document["truth"]),
                "unknown": payments - len(document["truth"]),
                "source": self.get(identifier)["source"],
                "dataset_id": identifier,
                "time_unit": "minutes",
            },
        }

    def sample(self, identifier, max_rows=20):
        if type(max_rows) is not int or not 1 <= max_rows <= 100:
            raise ValueError("Sample row count must be between 1 and 100.")
        config = self.dataset_config(identifier, "numeric")
        if config["loader"] == "prepared_fraud":
            from datasets.views import _feature_names, _values

            with open_prepared(config["path"]) as stream:
                names = _feature_names(stream, config)
                events = []
                iterator = stream.iter_events(**_selection(config))
                try:
                    for event in iterator:
                        events.append(event)
                        if len(events) == max_rows:
                            break
                finally:
                    iterator.close()
                labels = stream.truth_for(event.event_id for event in events)
                values = _values(events, names)
                rows = [
                    {
                        "id": event.event_id,
                        "time": event.event_time,
                        "features": dict(zip(names, values[index].tolist())),
                        "label": labels[event.event_id].label,
                    }
                    for index, event in enumerate(events)
                ]
        else:
            data = load_dataset(config)
            names = data.feature_names
            rows = [
                {
                    "id": data.ids[index],
                    "time": float(data.times[index]),
                    "features": dict(zip(names, data.features[index].tolist())),
                    "label": int(data.labels[index]),
                }
                for index in range(min(max_rows, len(data.ids)))
            ]
        return {
            "columns": list(names),
            "rows": rows,
            "time_unit": "seconds",
            "label_semantics": {"fraud": 1, "legitimate": 0, "unknown": -1},
        }
