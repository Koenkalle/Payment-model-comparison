"""Input validation shared by file loaders; no model-specific feature engineering."""
import csv, hashlib, json, math
from datetime import datetime
from pathlib import Path


def source(config, base):
    path = Path(config["path"])
    path = path if path.is_absolute() else base / path
    if not path.is_file():
        raise ValueError("Dataset file does not exist: " + str(path))
    return path


def provenance(config, path):
    return {
        "loader": config["loader"],
        "source_name": path.name,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "configuration": dict(config),
        "configuration_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest(),
        "origin": "user-supplied",
    }


def number(value, field):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be numeric, got {value!r}") from None
    if not math.isfinite(result):
        raise ValueError(field + " must be finite.")
    return result


def label(value, mapping=None):
    if value is None or str(value).strip().lower() in ("", "-1", "null", "unknown"):
        return -1
    if mapping is not None:
        key = str(value)
        if key not in mapping:
            raise ValueError("Unknown outcome value: " + key)
        value = mapping[key]
    if value in (0, "0", False, "false"):
        return 0
    if value in (1, "1", True, "true"):
        return 1
    raise ValueError(
        "Outcomes must be 0, 1 or unknown; configure label_values for other encodings."
    )


def timestamp(value, unit):
    if unit == "iso8601":
        try:
            date = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            raise ValueError("Invalid ISO timestamp: " + str(value)) from None
        if date.tzinfo is None:
            raise ValueError("ISO timestamps must include a timezone.")
        return date.timestamp()
    factors = {"seconds": 1, "minutes": 60, "hours": 3600}
    if unit not in factors:
        raise ValueError("time_unit must be seconds, minutes, hours or iso8601.")
    return number(value, "timestamp") * factors[unit]


def rows(path, required):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(
            reader.fieldnames
        ):
            raise ValueError("CSV needs unique column names.")
        missing = set(required) - set(reader.fieldnames)
        if missing:
            raise ValueError("Missing CSV columns: " + ", ".join(sorted(missing)))
        result = list(reader)
    if not result:
        raise ValueError("Dataset is empty.")
    if any(
        None in row or any(value is None for value in row.values()) for row in result
    ):
        raise ValueError("CSV row width differs from the header.")
    return result
