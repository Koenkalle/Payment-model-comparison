"""Reuse an ordered local transaction store without parsing its source CSVs."""
from pathlib import Path

from datasets.stream import open_prepared
from datasets.views import project
from datasets.configuration import resolved_config


def load(config, base):
    config = resolved_config(config, base)
    path = Path(config['path'])
    stream = open_prepared(path if path.is_absolute() else base / path)
    try:
        result = project(stream, config.get('view', 'stream'), config)
    except BaseException:
        stream.close()
        raise
    if result is not stream:
        stream.close()
    return result
