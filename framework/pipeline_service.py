"""HTTP-facing coordinator for persistent datasets and experiment jobs.

Only opaque dataset, run and job identifiers cross this boundary. The store
owns source paths and the training service owns experiment configurations.
"""
from pathlib import Path
from urllib.parse import parse_qs

from .pipeline_data import DatasetStore
from .dataset_graph import DatasetGraphService
from .pipeline_training import TrainingService


DEFAULT_PIPELINE_DIR = (
    Path(__file__).resolve().parents[1] / "artifacts" / "web-pipeline"
)


class PipelineService:
    def __init__(self, root=DEFAULT_PIPELINE_DIR, config_paths=()):
        self.root = Path(root).resolve()
        self.store = DatasetStore(self.root, config_paths=config_paths)
        self.graph = DatasetGraphService(self.store)
        self.training = TrainingService(self.store, self.root)

    def close(self):
        self.training.close()

    def get(self, path, query=""):
        parts = path.removeprefix("/api/pipeline/").split("/")
        if parts == ["datasets"]:
            return self.store.catalog()
        if parts == ["generators"]:
            return {"generators": self.store.generators()}
        if parts == ["models"]:
            params = parse_qs(query, max_num_fields=10)
            return {"models": self.training.models(params.get("dataset_id", [None])[0])}
        if parts == ["runs"]:
            return {"runs": self.training.runs()}
        if parts == ["jobs"]:
            return {"jobs": self.training.jobs()}
        if len(parts) == 2 and parts[0] == "jobs":
            return {"job": self.training.job(parts[1])}
        if len(parts) == 2 and parts[0] == "datasets":
            return {
                "dataset": self.store.describe(self.store.get(parts[1])),
                "sample": self.store.sample(parts[1]),
            }
        if len(parts) == 3 and parts[0] == "datasets" and parts[2] == "payments":
            return self.store.document(parts[1])
        if len(parts) == 3 and parts[0] == "datasets" and parts[2] == "features":
            return self.store.features(parts[1])
        if len(parts) == 3 and parts[0] == "datasets" and parts[2] == "graph":
            return self.graph.summary(parts[1])
        if (
            len(parts) == 4
            and parts[0] == "datasets"
            and parts[2:] == ["graph", "search"]
        ):
            params = parse_qs(query, max_num_fields=2, keep_blank_values=True)
            if set(params) - {"q", "limit"} or any(
                len(values) != 1 for values in params.values()
            ):
                raise ValueError("Graph search accepts only q and limit.")
            raw_limit = params.get("limit", ["20"])[0]
            if not raw_limit.isascii() or not raw_limit.isdecimal():
                raise ValueError("limit must be an integer from 1 to 50.")
            return self.graph.search(parts[1], params.get("q", [""])[0], int(raw_limit))
        raise FileNotFoundError("Unknown pipeline endpoint.")

    def post(self, path, payload):
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object.")
        if path == "/api/pipeline/datasets/generate":
            return 201, {"dataset": self.store.generate(payload)}
        if path == "/api/pipeline/datasets/import":
            return 201, {"dataset": self.store.import_source(payload)}
        if path == "/api/pipeline/train":
            return 202, {"job": self.training.start_training(payload)}
        if path == "/api/pipeline/compare":
            return 202, {"job": self.training.start_comparison(payload)}
        parts = path.removeprefix("/api/pipeline/").split("/")
        if len(parts) == 3 and parts[0] == "datasets" and parts[2] == "features":
            return 201, {"dataset": self.store.save_features(parts[1], payload)}
        if (
            len(parts) == 4
            and parts[0] == "datasets"
            and parts[2:] == ["graph", "query"]
        ):
            return 200, self.graph.query(parts[1], payload)
        raise FileNotFoundError("Unknown pipeline endpoint.")
