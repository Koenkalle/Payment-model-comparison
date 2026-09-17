"""Resolve file locations before hashing a dataset view's configuration."""
from pathlib import Path


def resolved_config(config, base):
    base = Path(base or ".").resolve()
    result = dict(config)

    def resolve(value):
        if not isinstance(value, (str, Path)) or not str(value):
            raise ValueError("Dataset paths must be nonempty filenames.")
        path = Path(value)
        return str((path if path.is_absolute() else base / path).resolve())

    if "path" in result:
        result["path"] = resolve(result["path"])
    if "paths" in result:
        if not isinstance(result["paths"], list):
            raise ValueError("paths must be a nonempty list of CSV files.")
        result["paths"] = [resolve(value) for value in result["paths"]]
    return result
