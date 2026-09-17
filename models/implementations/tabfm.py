"""Google TabFM classification with bounded, training-only in-context examples.

The upstream neural weights stay frozen. The JSON artifact stores the selected
numeric context and pinned model provenance so the estimator can be rebuilt
without unpickling Python objects. Weights remain in the Hugging Face cache.
"""
import importlib
import importlib.util
import inspect
import json
import logging
import sys
from pathlib import Path

import numpy as np

from framework.contracts import Predictions


LOGGER = logging.getLogger(__name__)
WEIGHTS_REPOSITORY = "google/tabfm-1.0.0-pytorch"
WEIGHTS_REVISION = "77cb9cc1b4fd3a9c77fbb9552c218200bb4dab83"
UPSTREAM_COMMIT = "fbb665569425fd2f490c6576b3af967876fe11ff"
MAX_FEATURES = 500
DEFAULT_PARAMETERS = {
    "max_context_rows": 100,
    "n_estimators": 1,
    "inference_batch_size": 128,
    "seed": 42,
}
PARAMETER_LIMITS = {
    "max_context_rows": (2, 2048),
    "n_estimators": (1, 8),
    "inference_batch_size": (1, 256),
    "seed": (0, 2147483647),
}
INSTALL_MESSAGE = "Install TabFM with Python 3.11+: python -m pip install -r requirements-tabfm.txt."
MODEL_PROVENANCE = {
    "repository": WEIGHTS_REPOSITORY,
    "revision": WEIGHTS_REVISION,
    "subfolder": "classification",
    "release": "1.0.0",
    "upstream_commit": UPSTREAM_COMMIT,
    "backend": "pytorch",
    "device": "cpu",
    "dtype": "float32",
    "weights_license": "tabfm-non-commercial-v1.0",
    "context_selection": "seeded-proportional-stratified/v1",
}


def _dependencies():
    """Import and check the public API without downloading a checkpoint."""
    if sys.version_info < (3, 11):
        raise RuntimeError(INSTALL_MESSAGE)
    try:
        tabfm = importlib.import_module("tabfm")
        torch = importlib.import_module("torch")
        hub = importlib.import_module("huggingface_hub")
        importlib.import_module("safetensors.torch")
        classifier = tabfm.TabFMClassifier
        loader = tabfm.tabfm_v1_0_0_pytorch.load
        classifier_arguments = inspect.signature(classifier).parameters
        loader_arguments = inspect.signature(loader).parameters
        required_classifier_arguments = {
            "model", "n_estimators", "norm_methods", "max_num_features", "max_num_rows",
            "batch_size", "random_state", "use_amp", "cache_context",
            "maybe_quantize_kv_cache", "keep_cache_on_device",
        }
        if not required_classifier_arguments.issubset(classifier_arguments):
            raise AttributeError("Unsupported TabFMClassifier API")
        if not {"model_type", "checkpoint_path", "device", "dtype", "use_cache"}.issubset(loader_arguments):
            raise AttributeError("Unsupported TabFM PyTorch loader API")
        versions = {
            "tabfm": str(tabfm.__version__),
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
            "scikit-learn": str(importlib.import_module("sklearn").__version__),
            "scipy": str(importlib.import_module("scipy").__version__),
        }
    except (ImportError, AttributeError, TypeError, ValueError) as error:
        raise RuntimeError(INSTALL_MESSAGE + " The official TabFM PyTorch API and its dependencies are required.") from error

    def load_cpu_weights(**parameters):
        # Match the trainer's two-thread CPU workers. Discovery only checks this
        # API; the process-wide setting changes when actual preparation starts.
        torch.set_num_threads(2)
        return loader(**parameters)

    return classifier, load_cpu_weights, hub.snapshot_download, versions


def availability_error():
    """Return a setup problem, or None; never fetch model weights."""
    if sys.version_info < (3, 11):
        return INSTALL_MESSAGE
    try:
        for name in ("tabfm", "torch", "huggingface_hub", "safetensors", "sklearn", "scipy", "pandas"):
            if importlib.util.find_spec(name) is None:
                return INSTALL_MESSAGE
        _dependencies()
    except Exception as error:
        # An incompatible optional native stack must not break the model list.
        return INSTALL_MESSAGE + " " + str(error)
    return None


def _parameters(parameters):
    if not isinstance(parameters, dict):
        raise ValueError("TabFM parameters must be an object.")
    unknown = set(parameters) - set(DEFAULT_PARAMETERS)
    if unknown:
        raise ValueError("Unknown TabFM parameter(s): " + ", ".join(sorted(map(str, unknown))))
    result = {**DEFAULT_PARAMETERS, **parameters}
    for name, (minimum, maximum) in PARAMETER_LIMITS.items():
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise ValueError(f"TabFM {name} must be an integer between {minimum} and {maximum}.")
    return result


def _feature_names(feature_names):
    names = list(feature_names)
    if not 1 <= len(names) <= MAX_FEATURES:
        raise ValueError(f"TabFM requires 1 to {MAX_FEATURES} numeric features.")
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("TabFM feature names must be nonempty, unique strings.")
    return names


def _matrix(x, names):
    values = np.asarray(x)
    if values.ndim != 2 or values.shape[1] != len(names) or values.dtype.kind not in "biuf":
        raise ValueError("TabFM needs a numeric matrix matching the feature names and order.")
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("TabFM numeric features must be finite.")
    return values


def _labels(y, rows):
    values = np.asarray(y)
    if values.shape != (rows,) or values.dtype.kind not in "biuf" or not np.isin(values, [0, 1]).all():
        raise ValueError("TabFM training labels must be known binary labels 0 and 1.")
    if set(values.tolist()) != {0, 1}:
        raise ValueError("TabFM training context requires both classes 0 and 1.")
    return values.astype(np.int64)


def _context_indices(labels, limit, seed):
    if len(labels) <= limit:
        return np.arange(len(labels))
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    # Preserve the training class ratio where possible, reserving at least one
    # example of each class even when fraud is exceptionally rare.
    count = int(np.floor(limit * len(positive) / len(labels) + .5))
    count = max(1, limit - len(negative), min(count, len(positive), limit - 1))
    rng = np.random.default_rng(seed)
    selected = np.concatenate((rng.choice(positive, count, replace=False),
                               rng.choice(negative, limit - count, replace=False)))
    return np.sort(selected)


class Model:
    def __init__(self):
        self.classifier = None
        self.state = None

    def _prepare(self, context_x, context_y, parameters, expected_versions=None):
        classifier_type, load_weights, snapshot_download, versions = _dependencies()
        if expected_versions is not None and versions != expected_versions:
            raise ValueError("TabFM artifact library versions differ from this environment; restore the recorded versions before evaluating.")
        try:
            LOGGER.info("Loading pinned TabFM classification weights from the Hugging Face cache; the first download is approximately 6.1 GiB.")
            snapshot = Path(snapshot_download(
                repo_id=WEIGHTS_REPOSITORY,
                revision=WEIGHTS_REVISION,
                allow_patterns=["classification/config.json", "classification/model.safetensors"],
            ))
            checkpoint = snapshot / "classification"
            # Require the safe weights format explicitly. A missing safetensors
            # file must not cause a library fallback to a pickle checkpoint.
            if not (checkpoint / "config.json").is_file() or not (checkpoint / "model.safetensors").is_file():
                raise RuntimeError("The pinned TabFM classification snapshot is incomplete.")
            LOGGER.info("Loading TabFM weights into CPU memory using float32.")
            weights = load_weights(model_type="classification", checkpoint_path=str(checkpoint),
                                   device="cpu", dtype=None, use_cache=True)
        except (OSError, RuntimeError, ValueError) as error:
            raise RuntimeError("Unable to load the pinned TabFM classification weights. Check the Hugging Face cache, network access and available memory. " + str(error)) from error
        classifier = classifier_type(
            model=weights,
            n_estimators=parameters["n_estimators"],
            norm_methods=["none"],
            max_num_features=MAX_FEATURES,
            max_num_rows=len(context_y),
            batch_size=1,
            random_state=parameters["seed"],
            use_amp=False,
            cache_context=True,
            maybe_quantize_kv_cache=False,
            keep_cache_on_device=True,
        )
        LOGGER.info("Preparing TabFM context cache: %d rows, %d features, %d ensemble member(s).",
                    len(context_y), context_x.shape[1], parameters["n_estimators"])
        classifier.fit(context_x.copy(), context_y.copy())
        classes = np.asarray(classifier.classes_)
        if classes.shape != (2,) or set(classes.tolist()) != {0, 1}:
            raise ValueError("TabFM classifier did not preserve binary classes 0 and 1.")
        LOGGER.info("TabFM context cache is ready.")
        return classifier, versions

    def fit(self, x, y, feature_names, parameters):
        names = _feature_names(feature_names)
        parameters = _parameters(parameters)
        x = _matrix(x, names)
        y = _labels(y, len(x))
        indices = _context_indices(y, parameters["max_context_rows"], parameters["seed"])
        context_x, context_y = x[indices].copy(), y[indices].copy()
        LOGGER.info("Selected %d TabFM context rows from %d eligible training rows (%d flagged).",
                    len(context_y), len(y), int(context_y.sum()))
        classifier, versions = self._prepare(context_x, context_y, parameters)
        self.state = {
            "version": 1,
            "schema": "tabfm-context/v1",
            "feature_names": names,
            "parameters": parameters,
            "provenance": dict(MODEL_PROVENANCE),
            "library_versions": versions,
            "training_rows": len(y),
            "context_indices": indices.tolist(),
            "context_x": context_x.tolist(),
            "context_y": context_y.tolist(),
        }
        self.classifier = classifier
        self.parameters = parameters
        self.library_version = versions["tabfm"]
        self.provenance = {**MODEL_PROVENANCE, "training_rows": len(y), "context_rows": len(indices), "library_versions": versions}

    def predict(self, x, feature_names, explain=False):
        if explain:
            raise ValueError("TabFM does not provide feature explanations.")
        if self.classifier is None:
            raise ValueError("TabFM has not been fitted or loaded.")
        if list(feature_names) != self.state["feature_names"]:
            raise ValueError("Feature names or order differ from the fitted TabFM model.")
        x = _matrix(x, self.state["feature_names"])
        probabilities = np.empty(len(x), dtype=float)
        positive = int(np.flatnonzero(np.asarray(self.classifier.classes_) == 1)[0])
        chunk_size = self.parameters["inference_batch_size"]
        total_batches = (len(x) + chunk_size - 1) // chunk_size
        progress_interval = max(1, (total_batches + 9) // 10)
        for start in range(0, len(x), chunk_size):
            end = min(start + chunk_size, len(x))
            batch = start // chunk_size
            if batch % progress_interval == 0 or end == len(x):
                LOGGER.info("Scoring TabFM rows %d-%d of %d.", start + 1, end, len(x))
            values = np.asarray(self.classifier.predict_proba(x[start:end].copy()), dtype=float)
            if (values.shape != (end - start, 2) or not np.isfinite(values).all()
                    or np.any((values < 0) | (values > 1))
                    or not np.allclose(values.sum(axis=1), 1, atol=1e-5, rtol=1e-5)):
                raise ValueError("TabFM must return two finite class probabilities summing to one per row.")
            probabilities[start:end] = values[:, positive]
        # These are probability log-odds, not unpublished internal model logits.
        clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
        margins = np.log(clipped) - np.log1p(-clipped)
        return Predictions(probabilities, margins)

    def save(self, path):
        if self.state is None:
            raise ValueError("TabFM has not been fitted or loaded.")
        Path(path).write_text(json.dumps(self.state, allow_nan=False), encoding="utf-8")

    def load(self, path, feature_names):
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        names = _feature_names(feature_names)
        if (not isinstance(state, dict) or state.get("version") != 1
                or state.get("schema") != "tabfm-context/v1"
                or state.get("feature_names") != names):
            raise ValueError("Incompatible TabFM context artifact or feature order.")
        if state.get("provenance") != MODEL_PROVENANCE:
            raise ValueError("TabFM artifact model provenance differs from the pinned implementation.")
        parameters = _parameters(state.get("parameters"))
        if set(state["parameters"]) != set(DEFAULT_PARAMETERS):
            raise ValueError("TabFM artifact parameters are incomplete.")
        context_x = _matrix(state.get("context_x"), names)
        context_y = _labels(state.get("context_y"), len(context_x))
        if len(context_y) > parameters["max_context_rows"]:
            raise ValueError("TabFM artifact exceeds its context row limit.")
        training_rows = state.get("training_rows")
        indices = state.get("context_indices")
        if (isinstance(training_rows, bool) or not isinstance(training_rows, int)
                or training_rows < len(context_y) or not isinstance(indices, list)
                or len(indices) != len(context_y)
                or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < training_rows for i in indices)
                or indices != sorted(set(indices))):
            raise ValueError("Invalid TabFM context membership metadata.")
        versions = state.get("library_versions")
        if (not isinstance(versions, dict) or set(versions) != {"tabfm", "torch", "numpy", "scikit-learn", "scipy"}
                or any(not isinstance(value, str) or not value for value in versions.values())):
            raise ValueError("Invalid TabFM artifact library versions.")
        classifier, _ = self._prepare(context_x, context_y, parameters, expected_versions=versions)
        self.state = state
        self.classifier = classifier
        self.parameters = parameters
        self.library_version = versions["tabfm"]
        self.provenance = {**MODEL_PROVENANCE, "training_rows": training_rows, "context_rows": len(context_y), "library_versions": versions}


def create():
    return Model()
