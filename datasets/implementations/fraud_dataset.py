"""Registered source adapters with explicit projections for existing runners."""
from datasets.stream import open_source
from datasets.views import project
from datasets.configuration import resolved_config


def load(config, base):
    config = resolved_config(config, base)
    stream = open_source(config, base)
    try:
        result = project(stream, config.get("view", "stream"), config)
    except BaseException:
        stream.close()
        raise
    if result is not stream:
        stream.close()
    return result
