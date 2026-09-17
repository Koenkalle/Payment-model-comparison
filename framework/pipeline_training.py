"""Durable, bounded model jobs for the dataset → training → comparison pipeline.

The experiment runners own fitting and artifact formats. This service supplies a
small web-safe configuration surface, job persistence, and comparable fraud
predictions on one explicitly held-out population.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import copy
import csv
import hashlib
import importlib
import json
import logging
import math
import os
from pathlib import Path
import re
import threading
import uuid

import numpy as np

from .experiments import (
    chronological_split,
    digest,
    evaluate_artifact,
    metrics,
    train_experiment,
)
from .registry import ROOT, load_dataset, model_entry


def _info(description, behavior, tuning):
    """Plain-text help shared by hover, keyboard and touch presentations.

    Keep model-specific semantics beside the parameter declaration. Defaults,
    bounds and choices come from the field itself, so help cannot drift from
    validation when the supported configuration surface changes.
    """
    return {
        "description": description,
        "sections": [
            {"title": "How it works here", "text": behavior},
            {"title": "Choosing a value", "text": tuning},
        ],
    }


def _field(
    name,
    label,
    default,
    minimum=None,
    maximum=None,
    options=None,
    *,
    info,
    default_label=None,
):
    kind = (
        "select"
        if options is not None
        else ("integer" if type(default) is int else "number")
    )
    result = {
        "name": name,
        "label": label,
        "type": kind,
        "default": default,
        "info": {"title": label, **info},
    }
    if minimum is not None:
        result["min"] = minimum
    if maximum is not None:
        result["max"] = maximum
    if options is not None:
        result["options"] = options
    if default_label is not None:
        result["default_label"] = default_label
    return result


MODEL_SETTINGS = {
    "logistic_regression": {
        "label": "Logistic regression",
        "view": "numeric",
        "dependencies": ["sklearn"],
        "task": "fraud-classification",
        "parameters": [
            _field(
                "C",
                "Inverse regularization strength",
                1.0,
                0.0001,
                1000,
                info=_info(
                    "Controls how strongly logistic regression penalizes large feature coefficients. Smaller C means stronger regularization.",
                    "Numeric features are standardized using training rows before fitting. The default solver uses an L2 penalty, which shrinks coefficients toward zero.",
                    "Lower C can help with correlated features or limited training data, but too much shrinkage can miss useful patterns. Higher C fits training data more freely and can overfit.",
                ),
            ),
            _field(
                "max_iter",
                "Maximum iterations",
                500,
                10,
                5000,
                info=_info(
                    "Caps the number of optimization steps used to fit logistic regression.",
                    "The L-BFGS solver may stop earlier when it converges. This is an optimization limit; validation outcomes do not select an earlier iteration.",
                    "Increase this if the run reports a convergence warning. Once fitting has converged, a larger limit usually adds no benefit. It does not control model complexity.",
                ),
            ),
            _field(
                "class_weight",
                "Class weighting",
                None,
                options=[None, "balanced"],
                default_label="None (equal row weights)",
                info=_info(
                    "Changes the importance of fraud and legitimate examples in the training loss.",
                    "None gives each row equal weight. Balanced weights each class inversely to its frequency among the eligible training labels, giving the two classes equal total weight.",
                    "Balanced can help the model attend to rare fraud, but may increase false positives and change probability calibration. Compare both settings using the same held-out data.",
                ),
            ),
            _field(
                "random_state",
                "Random seed",
                42,
                0,
                2147483647,
                info=_info(
                    "Records a seed for estimator operations that use randomness.",
                    "The current logistic regression configuration uses the deterministic L-BFGS solver, which does not use this seed. Changing only this field should not change the fitted model.",
                    "Keep the default for repeatable experiment configurations. This seed does not change the chronological train, validation or test partitions.",
                ),
            ),
        ],
    },
    "xgboost_native": {
        "label": "XGBoost (native)",
        "view": "numeric",
        "dependencies": ["xgboost"],
        "task": "fraud-classification",
        "parameters": [
            _field(
                "num_boost_round",
                "Boosting rounds",
                100,
                1,
                1000,
                info=_info(
                    "Sets how many sequential boosting rounds add trees to the fraud classifier.",
                    "Each round tries to improve the current ensemble. This implementation runs the requested number of rounds without validation-based early stopping; validation selects the final decision cutoff.",
                    "More rounds take longer and can fit finer patterns or overfit. Smaller learning rates commonly need more rounds, so consider these two controls together.",
                ),
            ),
            _field(
                "max_depth",
                "Maximum tree depth",
                4,
                1,
                12,
                info=_info(
                    "Limits how many splits a path through each decision tree can contain.",
                    "Deeper trees can represent more detailed interactions between dataset features. Training uses XGBoost’s histogram tree method.",
                    "Small depths favor simpler patterns. Larger depths can increase training cost and overfitting sharply, especially when there are few fraud examples.",
                ),
            ),
            _field(
                "eta",
                "Learning rate",
                0.08,
                0.0001,
                1,
                info=_info(
                    "Scales the contribution of each newly added tree to the ensemble.",
                    "A value of 0.1 applies one tenth of each tree’s unshrunk contribution. It changes the size of boosting updates, not the classification cutoff.",
                    "Lower values make learning more gradual and often need more boosting rounds. Higher values fit faster but may overshoot useful patterns or overfit sooner.",
                ),
            ),
            _field(
                "subsample",
                "Row sampling fraction",
                1.0,
                0.1,
                1,
                info=_info(
                    "Sets the fraction of training rows sampled for each boosting iteration.",
                    "A value of 1 uses all training rows. Smaller values randomly sample rows within the training partition; the validation and test partitions stay fixed.",
                    "Moderate sampling can reduce overfitting and computation. Very small fractions can discard too many rare fraud examples in an iteration and make learning unstable.",
                ),
            ),
            _field(
                "colsample_bytree",
                "Feature sampling fraction",
                1.0,
                0.1,
                1,
                info=_info(
                    "Sets the fraction of available dataset feature columns considered when building each tree.",
                    "A value of 1 makes all selected columns available to every tree. Lower values draw a random feature subset for each tree; they do not remove columns from the saved dataset.",
                    "Sampling can reduce reliance on a few correlated features. With only a small feature set, aggressive sampling may repeatedly hide the strongest predictors.",
                ),
            ),
            _field(
                "scale_pos_weight",
                "Fraud class weight",
                1.0,
                0.01,
                1000,
                info=_info(
                    "Multiplies the training loss contribution of fraud examples relative to legitimate examples.",
                    "A value of 1 applies no extra fraud weighting. Values above 1 emphasize the positive fraud class; values below 1 reduce its influence. The value is not calculated automatically from the dataset.",
                    "For rare fraud, the training ratio of legitimate to fraud rows is a possible starting point, then evaluate it. Larger weights can improve recall while increasing false positives and affecting calibration.",
                ),
            ),
            _field(
                "seed",
                "Random seed",
                42,
                0,
                2147483647,
                info=_info(
                    "Controls XGBoost’s random sampling choices during training.",
                    "It is especially relevant when row or feature sampling is below 1. The chronological data split is independent of this seed.",
                    "Use the same seed when comparing one parameter change. Try several seeds to assess sensitivity to sampling; identical seeds do not guarantee identical results across library versions or platforms.",
                ),
            ),
        ],
    },
    "tabfm": {
        "label": "TabFM (Google foundation model)",
        "view": "numeric",
        "dependencies": [],
        "task": "fraud-classification",
        "availability_check": "availability_error",
        "training_mode": "in_context",
        "max_features": 500,
        "description": "Predicts fraud from labelled examples using frozen pretrained weights. "
        "The training partition supplies context; validation selects the decision cutoff.",
        "usage_note": "Google’s pretrained weights are for non-commercial, non-production use. "
        "The first run downloads about 6.1 GiB of weights from Hugging Face; "
        "CPU inference needs additional memory. "
        "Saved models retain the selected labelled context rows.",
        "parameters": [
            _field(
                "max_context_rows",
                "Maximum context rows",
                100,
                2,
                2048,
                info=_info(
                    "Caps the number of labeled training examples supplied to TabFM as context for predictions.",
                    "TabFM keeps its pretrained weights frozen. If needed, this tool samples training rows using the seed, approximately preserving class proportions and retaining at least one example of each class. The chosen context is saved with the model.",
                    "More context can expose additional patterns but increases preparation and prediction memory and time. The actual count is also limited by eligible training rows. TabFM supports at most 500 selected numeric features.",
                ),
            ),
            _field(
                "n_estimators",
                "Ensemble members",
                1,
                1,
                8,
                info=_info(
                    "Sets the number of TabFM ensemble members combined for each prediction.",
                    "The members reuse the frozen pretrained foundation model. This setting is passed to TabFM’s classifier; it does not train this many independent neural networks or multiply the number of stored context rows.",
                    "More members can make predictions less sensitive to a single ensemble configuration, but cost more inference work and context-cache memory. Start with one and compare held-out performance before increasing it.",
                ),
            ),
            _field(
                "inference_batch_size",
                "Prediction batch size",
                128,
                1,
                256,
                info=_info(
                    "Caps the number of query rows passed to one TabFM prediction call.",
                    "The tool predicts in chunks using the same saved training context. This is distinct from the context-row limit and does not allow query labels into the context or update pretrained weights.",
                    "Smaller chunks reduce peak query memory but require more calls. Larger chunks may improve throughput when memory permits. Changing this setting does not change the train, validation or test membership.",
                ),
            ),
            _field(
                "seed",
                "Random seed",
                42,
                0,
                2147483647,
                info=_info(
                    "Controls which training examples are sampled for context and TabFM’s seeded ensemble choices.",
                    "Context selection is stratified by training label when there are more eligible rows than the limit. The selected rows and seed are retained in the saved model; pretrained weights remain frozen.",
                    "Keep the seed fixed for controlled comparisons. Compare several seeds to check whether a small context is representative. The seed does not randomize the chronological split.",
                ),
            ),
        ],
    },
    "dyg_tami_native": {
        "label": "DyGFormer + TAMI (supervised fraud)",
        "view": "labeled_graph",
        "dependencies": ["torch", "sklearn"],
        "task": "temporal-fraud-classification",
        "parameters": [
            _field(
                "epochs",
                "Maximum epochs",
                5,
                1,
                200,
                info=_info(
                    "Caps the number of complete training passes for the temporal encoder and fraud prediction head.",
                    "This tool initializes the network randomly and fine-tunes it on known training outcomes. After each epoch it measures model-validation loss, keeps the best weights and may stop early according to patience.",
                    "More epochs allow additional learning but increase runtime and can overfit. Inspect the loss history and best epoch before increasing the limit.",
                ),
            ),
            _field(
                "patience",
                "Early stopping patience",
                3,
                1,
                30,
                info=_info(
                    "Sets how many consecutive epochs without a lower model-validation loss are allowed before training stops.",
                    "An improvement resets the counter. The model restores the epoch with the lowest validation binary cross-entropy loss, which measures probability fit. This uses the first half of validation; the second half selects the decision cutoff.",
                    "Smaller patience saves time but may stop during a temporary plateau. Larger patience tolerates noisy progress and can run up to the maximum epoch limit.",
                ),
            ),
            _field(
                "batch_size",
                "Batch size",
                32,
                1,
                256,
                info=_info(
                    "Sets the target number of labeled training payments accumulated before an optimizer update.",
                    "Payments are processed in chronological timestamp groups. A group is never split for an update, so the actual accumulated count can exceed this target. Gradients are averaged over the labeled examples in that update.",
                    "Smaller values make updates more frequent and potentially noisier. Larger values average more examples per update. This is not a cap on the size of a simultaneous timestamp group.",
                ),
            ),
            _field(
                "learning_rate",
                "Learning rate",
                0.001,
                0.000001,
                0.1,
                info=_info(
                    "Controls the step size used by the Adam optimizer to update the temporal network and fraud head.",
                    "The same rate is used for trainable encoder and head parameters throughout the run. It affects learned weights; the fraud decision cutoff is selected separately.",
                    "Lower rates learn more slowly and may need more epochs. Higher rates can reduce initial loss faster but may cause unstable training or poor validation performance.",
                ),
            ),
            _field(
                "weight_decay",
                "Weight decay",
                0.0,
                0,
                1,
                info=_info(
                    "Sets the L2 penalty applied by Adam to trainable network weights.",
                    "Zero disables the penalty. Positive values discourage large weights in the temporal encoder and fraud head while they are optimized.",
                    "A modest penalty can help when training loss improves while validation loss worsens. Excessive decay can suppress useful patterns and lead to underfitting.",
                ),
            ),
            _field(
                "time_feat_dim",
                "Time feature dimensions",
                8,
                options=[4, 8, 16, 32],
                info=_info(
                    "Sets the size of the learned vector used to represent elapsed time since earlier interactions.",
                    "The TAMI time encoder transforms time differences into this many components before projecting them into the transformer’s time channel. These are internal learned dimensions, separate from dataset feature columns.",
                    "More dimensions can represent richer timing patterns at additional parameter and computation cost. Smaller values keep the model compact when timing evidence is limited.",
                ),
            ),
            _field(
                "channel_embedding_dim",
                "Channel embedding dimensions",
                8,
                options=[4, 8, 16, 32],
                info=_info(
                    "Sets the width of each of the four internal information channels used by the temporal transformer.",
                    "The channels represent node attributes, edge attributes, elapsed time and neighbor co-occurrence. They are concatenated, so the attention width is four times this value. This does not add or remove dataset columns.",
                    "Larger widths increase representation capacity, memory use and training cost. The total attention width must be divisible by the head count; the displayed choices satisfy that constraint.",
                ),
            ),
            _field(
                "num_layers",
                "Transformer layers",
                1,
                1,
                4,
                info=_info(
                    "Sets how many transformer blocks refine the combined sender and recipient histories.",
                    "Each block uses attention and a feed-forward network to mix information from the observed interaction sequence. All blocks are trained using the known fraud outcomes.",
                    "More layers can learn more complex interactions but add parameters and runtime. Shallow networks are easier to fit on small labeled datasets.",
                ),
            ),
            _field(
                "num_heads",
                "Attention heads",
                2,
                options=[1, 2, 4, 8],
                info=_info(
                    "Splits each transformer layer’s attention into parallel heads that can focus on different historical relationships.",
                    "Heads share the fixed total width of four times the channel embedding dimension. More heads divide that width into smaller per-head vectors; they do not multiply the total embedding width.",
                    "Different head counts change how attention is organized, without guaranteeing better predictions. The attention width must divide evenly by the chosen head count.",
                ),
            ),
            _field(
                "dropout",
                "Dropout",
                0.1,
                0,
                0.8,
                info=_info(
                    "Sets the probability of randomly dropping activations or attention contributions in the transformer during training.",
                    "For example, 0.1 is a ten percent drop probability where dropout is applied. Dropout is disabled when computing validation and prediction outputs.",
                    "Increasing dropout can reduce overfitting by discouraging reliance on particular internal signals. Too much can slow learning or hide useful patterns. Zero disables dropout.",
                ),
            ),
            _field(
                "max_input_sequence_length",
                "History length",
                16,
                options=[8, 16, 32, 64, 128],
                info=_info(
                    "Caps the sequence length retained separately for the sender and recipient when encoding a candidate payment.",
                    "One slot represents the endpoint itself; the remaining slots hold its most recent interactions strictly before the candidate timestamp. A value of 16 therefore allows up to 15 earlier interactions per endpoint. This does not truncate separately saved dataset features or the TAMI pair memory.",
                    "Longer histories expose older behavior but increase attention work and memory sharply. Short histories favor speed and recent behavior. Same-timestamp and future payments never enter the history.",
                ),
            ),
            _field(
                "seed",
                "Random seed",
                42,
                0,
                2147483647,
                info=_info(
                    "Controls PyTorch’s random weight initialization and stochastic training operations such as dropout.",
                    "The temporal model begins with randomly initialized weights in this tool. Payments remain chronologically ordered, and changing the seed does not reshuffle partition membership.",
                    "Keep the seed fixed when comparing an individual setting. Repeat promising configurations with multiple seeds to assess sensitivity to initialization. Library or platform changes can still affect reproducibility.",
                ),
            ),
        ],
    },
}
for _settings in MODEL_SETTINGS.values():
    _settings["parameters"].append(
        {
            **_field(
                "decision_threshold",
                "Fixed fraud probability cutoff (optional)",
                None,
                0,
                0.999999,
                default_label="Automatic (validation)",
                info=_info(
                    "Converts the model’s fraud probability into a block or allow decision. A payment is blocked only when its probability is strictly greater than the cutoff.",
                    (
                        "Leave blank to maximize F1 on known outcomes in the policy-validation partition, the second half of validation. The first half selects model weights. If policy validation lacks either class, the cutoff falls back to 0.5."
                        if _settings["view"] == "labeled_graph"
                        else "Leave blank to maximize F1 on known validation outcomes; both classes must be present. F1 balances precision and recall; this cutoff is selected after fitting or preparing context."
                    )
                    + " Entering a value fixes the cutoff instead. Test outcomes never select it.",
                    "Lower cutoffs generally block more payments, improving recall at the cost of more false positives. Higher cutoffs generally block fewer. This changes decisions and reported metrics, not learned weights, context or probability scores.",
                ),
            ),
            "required": False,
            "nullable": True,
            "description": "Leave blank to select the cutoff from validation outcomes.",
        }
    )


def _now():
    return datetime.now(timezone.utc).isoformat()


def _hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "-" + uuid.uuid4().hex)
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _identifier(value, label):
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value
    ):
        raise ValueError(label + " must be a valid identifier.")
    return value


def _validate_parameters(model_id, supplied):
    if not isinstance(supplied, dict):
        raise ValueError("parameters must be an object.")
    fields = {field["name"]: field for field in MODEL_SETTINGS[model_id]["parameters"]}
    extra = set(supplied) - set(fields)
    if extra:
        raise ValueError("Unsupported training parameters: " + ", ".join(sorted(extra)))
    result = {}
    for name, field in fields.items():
        value = supplied.get(name, field["default"])
        if value is None and field.get("nullable"):
            result[name] = None
            continue
        if field["type"] == "select":
            if not any(
                type(value) is type(option) and value == option
                for option in field["options"]
            ):
                raise ValueError(name + " must be one of the displayed options.")
        else:
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(name + " must be a finite number.")
            if field["type"] == "integer" and type(value) is not int:
                raise ValueError(name + " must be an integer.")
            if value < field["min"] or value > field["max"]:
                raise ValueError(
                    f'{name} must be between {field["min"]} and {field["max"]}.'
                )
        result[name] = value
    return result


def _validate_split(value):
    if not isinstance(value, dict) or set(value) - {"train", "validation"}:
        raise ValueError("split accepts train and validation fractions.")
    train, validation = value.get("train", 0.6), value.get("validation", 0.2)
    if (
        any(
            type(part) not in (int, float)
            or not math.isfinite(part)
            or not 0 < part < 1
            for part in (train, validation)
        )
        or train + validation >= 1
    ):
        raise ValueError(
            "Train and validation fractions must be positive and leave a test partition."
        )
    return {"train": train, "validation": validation}


class _JobLog(logging.Handler):
    def __init__(self, service, job_id):
        super().__init__(logging.INFO)
        self.service, self.job_id = service, job_id
        self.thread = threading.get_ident()

    def emit(self, record):
        if record.thread == self.thread:
            self.service._log(self.job_id, record.getMessage())


class TrainingService:
    """One CPU worker, persistent jobs, immutable artifacts, and honest comparisons."""

    def __init__(self, store, root):
        self.store = store
        self.root = Path(root).resolve()
        self._lock = threading.RLock()
        self._records = {}
        self._futures = {}
        self._dependencies = {}
        for directory in ("jobs", "runs", "comparisons"):
            (self.root / directory).mkdir(parents=True, exist_ok=True)
        # The kernel releases this lock when a server exits, including crashes.
        # A second live process must never mark the owner's jobs interrupted.
        self._owner = (self.root / "worker.lock").open("a+b")
        try:
            if os.name == "nt":
                import msvcrt

                if self._owner.tell() == 0:
                    self._owner.write(b"\0")
                    self._owner.flush()
                self._owner.seek(0)
                msvcrt.locking(self._owner.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self._owner.close()
            raise ValueError(
                "Another live training server owns this pipeline directory."
            ) from error
        for path in sorted((self.root / "jobs").glob("*.json")):
            try:
                record = json.loads(path.read_text())
                _identifier(record["id"], "job_id")
                if record["status"] in ("queued", "running"):
                    record.update(
                        status="failed",
                        stage="interrupted",
                        finished_at=_now(),
                        error="Server stopped before this job completed. Start a new job to retry.",
                    )
                    _write(path, record)
                self._records[record["id"]] = record
            except (ValueError, KeyError, OSError):
                logging.getLogger(__name__).warning(
                    "Ignoring unreadable job manifest %s", path.name
                )
        self._worker = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="payment-training"
        )

    def close(self, wait=True):
        self._worker.shutdown(wait=wait, cancel_futures=False)
        if wait:
            self._owner.close()

    def models(self, dataset_id=None):
        dataset = self.store.get(dataset_id) if dataset_id else None
        result = []
        for identifier, settings in MODEL_SETTINGS.items():
            missing = []
            for dependency in settings["dependencies"]:
                if dependency not in self._dependencies:
                    try:
                        importlib.import_module(dependency)
                        self._dependencies[dependency] = None
                    except Exception as error:
                        self._dependencies[dependency] = str(error)
                if self._dependencies[dependency]:
                    missing.append(dependency)
            reason = (
                "Install required Python dependencies: " + ", ".join(missing) + "."
                if missing
                else None
            )
            if reason is None and settings.get("availability_check"):
                module = importlib.import_module(
                    model_entry(identifier)["python_module"]
                )
                reason = getattr(module, settings["availability_check"])()
            if dataset is not None and settings["view"] not in dataset["views"]:
                reason = (
                    "This dataset does not provide the "
                    + settings["view"]
                    + " view required by this model."
                )
            if (
                dataset is not None
                and settings.get("max_features") is not None
                and len(dataset.get("feature_names", [])) > settings["max_features"]
            ):
                reason = f"This model supports at most {settings['max_features']} numeric features."
            if (
                dataset is not None
                and reason is None
                and "fraud" in dataset
                and "legitimate" in dataset
                and (not dataset["fraud"] or not dataset["legitimate"])
            ):
                reason = "Supervised training needs both known fraud and legitimate outcomes in this dataset."
            result.append(
                {
                    "id": identifier,
                    "label": settings["label"],
                    "available": reason is None,
                    "reason": reason,
                    "view": settings["view"],
                    "tasks": [settings["task"]],
                    **{
                        key: settings[key]
                        for key in ("description", "training_mode", "usage_note")
                        if key in settings
                    },
                    "parameters": copy.deepcopy(settings["parameters"]),
                    "score_semantics": "fraud_probability",
                    "validation_note": (
                        "Validation is divided equally between model selection and policy threshold selection."
                        if settings["view"] == "labeled_graph"
                        else "Validation selects the decision threshold."
                    ),
                }
            )
        return result

    def jobs(self):
        with self._lock:
            result = [
                copy.deepcopy(value)
                for value in sorted(
                    self._records.values(),
                    key=lambda row: row["created_at"],
                    reverse=True,
                )
            ]
            for record in result:
                if record["kind"] == "comparison" and "result" in record:
                    record["result"].pop("rows", None)
                    record["result"].pop("evaluation_ids", None)
            return result

    def job(self, identifier):
        _identifier(identifier, "job_id")
        with self._lock:
            if identifier not in self._records:
                raise ValueError("Unknown job: " + identifier)
            return copy.deepcopy(self._records[identifier])

    def _change(self, identifier, **changes):
        with self._lock:
            self._records[identifier].update(changes)
            _write(
                self.root / "jobs" / (identifier + ".json"), self._records[identifier]
            )

    def _log(self, identifier, message):
        with self._lock:
            record = self._records[identifier]
            record["logs"] = (
                record.get("logs", []) + [{"at": _now(), "message": message}]
            )[-100:]
            record["message"] = message
            _write(self.root / "jobs" / (identifier + ".json"), record)

    def _submit(self, kind, payload, function):
        identifier = uuid.uuid4().hex
        with self._lock:
            record = {
                "id": identifier,
                "kind": kind,
                "status": "queued",
                "stage": "queued",
                "created_at": _now(),
                "worker_pid": os.getpid(),
                "payload": copy.deepcopy(payload),
                "logs": [],
            }
            self._records[identifier] = record
            _write(self.root / "jobs" / (identifier + ".json"), record)
            self._futures[identifier] = self._worker.submit(
                self._execute, identifier, function
            )
            return copy.deepcopy(record)

    def _execute(self, identifier, function):
        self._change(identifier, status="running", stage="loading", started_at=_now())
        handler = _JobLog(self, identifier)
        loggers = [
            logging.getLogger(name)
            for name in (
                "framework.temporal_fraud_experiments",
                "models.implementations.dyg_tami_fraud",
                "models.implementations.dyg_tami",
                "models.implementations.tabfm",
            )
        ]
        levels = [logger.level for logger in loggers]
        for logger in loggers:
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        try:
            result = function(identifier)
            changes = {
                "status": "succeeded",
                "stage": "complete",
                "finished_at": _now(),
                "result": result,
            }
            if "run_id" in result:
                changes["run_id"] = result["run_id"]
            self._change(identifier, **changes)
        except Exception as error:
            self._change(
                identifier,
                status="failed",
                stage="failed",
                finished_at=_now(),
                error=str(error) or type(error).__name__,
            )
            logging.getLogger(__name__).exception("Pipeline job %s failed", identifier)
        finally:
            for logger, level in zip(loggers, levels):
                logger.removeHandler(handler)
                logger.setLevel(level)

    def start_training(self, payload):
        if not isinstance(payload, dict) or set(payload) - {
            "dataset_id",
            "model_id",
            "parameters",
            "split",
            "name",
        }:
            raise ValueError(
                "Training accepts dataset_id, model_id, parameters, split and an optional name."
            )
        dataset_id = _identifier(payload.get("dataset_id"), "dataset_id")
        model_id = payload.get("model_id")
        if not isinstance(model_id, str) or model_id not in MODEL_SETTINGS:
            raise ValueError("Choose a supported trainable fraud model.")
        descriptor = next(
            row for row in self.models(dataset_id) if row["id"] == model_id
        )
        if not descriptor["available"]:
            raise ValueError(descriptor["reason"])
        parameters = _validate_parameters(model_id, payload.get("parameters", {}))
        split = _validate_split(payload.get("split", {}))
        name = payload.get("name", descriptor["label"])
        if not isinstance(name, str) or not name.strip() or len(name) > 120:
            raise ValueError("Run name must contain between 1 and 120 characters.")
        value = {
            "dataset_id": dataset_id,
            "model_id": model_id,
            "parameters": parameters,
            "split": split,
            "name": name.strip(),
        }
        return self._submit(
            "training", value, lambda job_id: self._train(job_id, value)
        )

    def _snapshot(self, dataset_id):
        """Independent of labels and display names; retain per-observation identity.

        IDs alone are unsafe (separate generators often reuse payment-1), while
        source checksums alone miss copied subsets. Both dataset identity and
        ID-independent observation signatures are retained for comparison gates.
        """
        metadata = self.store.get(dataset_id)
        availability = {}
        if "payments" in metadata["views"]:
            dataset = load_dataset(
                self.store.dataset_config(dataset_id, "payments"), ROOT
            )
            document = dataset.document
            accounts = {
                account["id"]: str(account.get("external_id", account["id"]))
                for account in document["accounts"]
            }
            availability = {
                identifier: float(value) * 60
                for identifier, value in document.get("label_available_at", {}).items()
            }
            observations = [
                {
                    "id": event["id"],
                    "timestamp_seconds": float(event["t"]) * 60,
                    "signature": _hash(
                        [
                            float(event["t"]),
                            accounts[event["u"]],
                            accounts[event["v"]],
                            float(event["amount"]),
                        ]
                    ),
                }
                for event in document["events"]
                if event["kind"] == "payment"
            ]
        else:
            # A feature selection changes model input columns, not observation
            # identity. Use the preserved source table for overlap checks and
            # label availability, including on numeric-only feature variants.
            source_config = getattr(
                self.store, "source_config", self.store.dataset_config
            )
            numeric_config = source_config(dataset_id, "numeric")
            dataset = load_dataset(numeric_config, ROOT)
            if numeric_config["loader"] in ("prepared_fraud", "fraud_dataset"):
                stream_config = {
                    key: value
                    for key, value in numeric_config.items()
                    if key != "selection"
                }
                stream = load_dataset({**stream_config, "view": "stream"}, ROOT)
                try:
                    availability = {
                        identifier: float(outcome.available_at)
                        for identifier, outcome in stream.truth_for(dataset.ids).items()
                        if outcome.available_at is not None
                    }
                finally:
                    stream.close()
            observations = [
                {
                    "id": identifier,
                    "timestamp_seconds": float(dataset.times[index]),
                    "signature": _hash(
                        [
                            float(dataset.times[index]),
                            list(dataset.feature_names),
                            np.asarray(dataset.features[index], dtype=float).tolist(),
                        ]
                    ),
                }
                for index, identifier in enumerate(dataset.ids)
            ]
        return {
            "fingerprint": _hash(observations),
            "store_fingerprint": metadata.get("fingerprint"),
            "source_namespace": metadata.get("source_namespace"),
            "observations": observations,
            "label_available_at": availability,
        }

    def _numeric_label_cutoffs(self, config, folder, snapshot):
        """Mask late outcomes at each fitting/selection boundary, preserving X.

        The general numeric runner has no per-row label-availability contract.
        A private, checksummed CSV supplies the appropriate as-of targets while
        comparison still evaluates the original stored dataset and test IDs.
        """
        availability = snapshot.get("label_available_at", {})
        if not availability:
            return {
                "assumption": "retrospective",
                "masked": {"train": 0, "validation": 0},
            }
        dataset = load_dataset(config["dataset"], ROOT)
        partitions = chronological_split(dataset, config["split"])
        labels = dataset.labels.copy()
        masked, cutoffs = {}, {}
        for name, following in (("train", "validation"), ("validation", "test")):
            cutoff = float(
                np.nextafter(dataset.times[partitions[following][0]], -np.inf)
            )
            cutoffs[name] = cutoff
            indices = [
                int(index)
                for index in partitions[name]
                if labels[index] >= 0
                and availability.get(dataset.ids[index], -math.inf) > cutoff
            ]
            labels[indices] = -1
            masked[name] = len(indices)
        role_names = []
        for role in ("id", "time", "label"):
            name = "__pipeline_" + role
            while name in dataset.feature_names:
                name += "_"
            role_names.append(name)
        path = folder / "training-data.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(role_names + list(dataset.feature_names))
            for index, identifier in enumerate(dataset.ids):
                writer.writerow(
                    [
                        identifier,
                        float(dataset.times[index]),
                        int(labels[index]),
                        *dataset.features[index].tolist(),
                    ]
                )
        original_config = config["dataset"]
        config["dataset"] = {
            "loader": "numeric_csv",
            "path": str(path),
            "id_column": role_names[0],
            "time_column": role_names[1],
            "label_column": role_names[2],
            "features": list(dataset.feature_names),
        }
        policy = {
            "assumption": "explicit-confirmations-where-provided; remaining labels retrospective",
            "cutoffs_seconds": cutoffs,
            "masked": masked,
        }
        config["pipeline"] = {
            "original_dataset": original_config,
            "label_policy": policy,
            "training_data_sha256": digest(path),
        }
        return policy

    def _train(self, job_id, payload):
        settings = MODEL_SETTINGS[payload["model_id"]]
        dataset_id = payload["dataset_id"]
        snapshot = self._snapshot(dataset_id)
        config = {
            "model": payload["model_id"],
            "dataset": self.store.dataset_config(dataset_id, settings["view"]),
            "parameters": dict(payload["parameters"]),
            "split": dict(payload["split"]),
        }
        decision_threshold = config["parameters"].pop("decision_threshold")
        if decision_threshold is not None and settings["view"] == "numeric":
            config["decision_threshold"] = decision_threshold
        if payload["model_id"] == "xgboost_native":
            config["parameters"]["nthread"] = 2
        if settings["view"] == "labeled_graph":
            config.update(
                task="temporal-fraud-classification",
                graph={"node_feature_dim": 8},
                initialization={"method": "random"},
                encoder_training="finetune",
                prediction_head={"id": "fraud_linear"},
            )
            config["parameters"].update(device="cpu", num_threads=2, patch_size=1)
            config["split"] = {
                "train": payload["split"]["train"],
                "model_validation": payload["split"]["validation"] / 2,
                "policy_validation": payload["split"]["validation"] / 2,
            }
            if decision_threshold is not None:
                config["score_threshold"] = -math.log1p(-decision_threshold) / math.log(
                    2
                )
        run_id = uuid.uuid4().hex
        folder = self.root / "runs" / run_id
        folder.mkdir()
        label_policy = (
            self._numeric_label_cutoffs(config, folder, snapshot)
            if settings["view"] == "numeric"
            else {"assumption": "native-partition-label-cutoffs"}
        )
        _write(folder / "configuration.json", config)
        _write(folder / "observations.json", snapshot)
        in_context = settings.get("training_mode") == "in_context"
        self._change(
            job_id,
            stage="preparing" if in_context else "training",
            message=(
                "Preparing labelled context from the chronological training partition."
                if in_context
                else "Fitting on the chronological training partition."
            ),
        )
        metadata, test_report = train_experiment(config, folder / "artifact", ROOT)
        self._change(
            job_id, stage="saving", message="Saving model and held-out evaluation."
        )
        run = {
            "id": run_id,
            "run_id": run_id,
            "job_id": job_id,
            "status": "ready",
            "name": payload["name"],
            "label": payload["name"],
            "model_id": payload["model_id"],
            "dataset_id": dataset_id,
            "dataset_name": self.store.get(dataset_id)["name"],
            "created_at": _now(),
            "task": settings["task"],
            "view": settings["view"],
            "parameters": payload["parameters"],
            "split": payload["split"],
            "feature_names": metadata.get(
                "feature_names", self.store.get(dataset_id).get("feature_names", [])
            ),
            "label_policy": label_policy,
            "partition_counts": {
                key: len(ids) for key, ids in metadata["split"].items()
            },
            "test_metrics": test_report["metrics"],
            "threshold": metadata["threshold"],
            "threshold_units": metadata.get("threshold_units", "fraud-probability"),
            "threshold_source": metadata["threshold_source"],
            "score_semantics": "fraud_probability",
            "model_sha256": metadata["model_sha256"],
            "artifact_manifest_sha256": digest(folder / "artifact" / "manifest.json"),
            "test_report_sha256": digest(folder / "artifact" / "test-report.json"),
            "observations_sha256": digest(folder / "observations.json"),
            "configuration_sha256": digest(folder / "configuration.json"),
            "dataset_fingerprint": snapshot["fingerprint"],
        }
        if "model_provenance" in metadata:
            run["model_provenance"] = metadata["model_provenance"]
        _write(folder / "run.json", run)
        return {"run_id": run_id, "run": run}

    def _ready(self, identifier):
        _identifier(identifier, "run_id")
        folder = self.root / "runs" / identifier
        try:
            run = json.loads((folder / "run.json").read_text())
            if (
                run["id"] != identifier
                or run["status"] != "ready"
                or run["model_id"] not in MODEL_SETTINGS
            ):
                raise ValueError("Invalid ready-model record.")
            for filename, key in (
                ("artifact/manifest.json", "artifact_manifest_sha256"),
                ("artifact/test-report.json", "test_report_sha256"),
                ("observations.json", "observations_sha256"),
                ("configuration.json", "configuration_sha256"),
            ):
                if digest(folder / filename) != run[key]:
                    raise ValueError(
                        "Ready model " + identifier + " has a changed " + filename + "."
                    )
            metadata = json.loads((folder / "artifact" / "manifest.json").read_text())
            config = json.loads((folder / "configuration.json").read_text())
            staged_checksum = config.get("pipeline", {}).get("training_data_sha256")
            if (
                staged_checksum is not None
                and digest(folder / "training-data.csv") != staged_checksum
            ):
                raise ValueError(
                    "Stored as-of training data checksum differs from its configuration."
                )
            model_file = "model.npz" if run["view"] == "labeled_graph" else "model.json"
            if digest(folder / "artifact" / model_file) != metadata["model_sha256"]:
                raise ValueError(
                    "Ready model weights checksum differs from its manifest."
                )
            descriptor = model_entry(run["model_id"])
            if run["view"] == "labeled_graph":
                from .temporal_experiments import source_hashes

                selected = descriptor["tasks"]["temporal-fraud-classification"]
                descriptor = {**descriptor, **selected}
                current = source_hashes(descriptor)
            else:
                current = digest(
                    ROOT / (descriptor["python_module"].replace(".", "/") + ".py")
                )
            if current != metadata["implementation_sha256"]:
                raise ValueError(
                    "Model implementation changed; train this model again."
                )
            return run, metadata, folder
        except (OSError, KeyError, json.JSONDecodeError) as error:
            raise ValueError(
                "Ready model is missing or invalid: " + identifier
            ) from error

    def runs(self):
        result = []
        for path in (self.root / "runs").glob("*/run.json"):
            try:
                run, _, _ = self._ready(path.parent.name)
                result.append(run)
            except ValueError:
                continue
        return sorted(result, key=lambda row: row["created_at"], reverse=True)

    def start_comparison(self, payload):
        if not isinstance(payload, dict) or set(payload) - {
            "dataset_id",
            "run_ids",
            "partition",
        }:
            raise ValueError("Comparison accepts dataset_id, run_ids and partition.")
        dataset_id = _identifier(payload.get("dataset_id"), "dataset_id")
        dataset = self.store.get(dataset_id)
        run_ids = payload.get("run_ids")
        if (
            not isinstance(run_ids, list)
            or not 1 <= len(run_ids) <= 20
            or not all(isinstance(value, str) for value in run_ids)
            or len(set(run_ids)) != len(run_ids)
        ):
            raise ValueError("Choose between 1 and 20 distinct ready models.")
        partition = payload.get("partition", "test")
        if partition not in ("test", "all"):
            raise ValueError("Comparison partition must be test or all.")
        for identifier in run_ids:
            run, _, _ = self._ready(identifier)
            if run["view"] not in dataset["views"]:
                raise ValueError(
                    run["name"] + " requires a " + run["view"] + " dataset view."
                )
            model = next(row for row in self.models() if row["id"] == run["model_id"])
            if not model["available"]:
                raise ValueError(model["reason"])
        value = {
            "dataset_id": dataset_id,
            "run_ids": list(run_ids),
            "partition": partition,
        }
        return self._submit(
            "comparison", value, lambda job_id: self._compare(job_id, value)
        )

    def _comparison_population(self, payload, snapshot, records):
        all_ids = {row["id"] for row in snapshot["observations"]}
        signatures = {row["id"]: row["signature"] for row in snapshot["observations"]}
        common = set(all_ids)
        same_count = 0
        origins = []
        for run, metadata, folder in records:
            trained_snapshot = json.loads((folder / "observations.json").read_text())
            same = snapshot["fingerprint"] == trained_snapshot["fingerprint"] or (
                snapshot["store_fingerprint"] is not None
                and snapshot["store_fingerprint"]
                == trained_snapshot.get("store_fingerprint")
            )
            if same:
                same_count += 1
                if payload["partition"] == "all":
                    raise ValueError(
                        "This dataset was used to train a selected model. Choose its held-out test partition."
                    )
                common &= set(metadata["split"]["test"])
            observed_ids = {
                identifier
                for name, ids in metadata["split"].items()
                if name != "test"
                for identifier in ids
            }
            observed = {
                row["signature"]
                for row in trained_snapshot["observations"]
                if row["id"] in observed_ids
            }
            origins.append((run, observed, same, observed_ids, trained_snapshot))
        if payload["partition"] == "test" and not same_count:
            raise ValueError(
                "No saved test partition belongs to this dataset. Explicitly choose all rows of a new dataset."
            )
        if not common:
            raise ValueError("Selected models have no common held-out test payments.")
        for run, observed, same, observed_ids, trained_snapshot in origins:
            if same:
                leaked = common & observed_ids
            else:
                leaked = {
                    identifier
                    for identifier in common
                    if signatures[identifier] in observed
                }
                # Stable source namespaces make row IDs authoritative even if
                # another import selects different numeric feature columns.
                namespace = snapshot.get("source_namespace")
                if namespace is not None and namespace == trained_snapshot.get(
                    "source_namespace"
                ):
                    leaked |= common & observed_ids
            if leaked:
                raise ValueError(
                    f'{len(leaked)} selected rows overlap training or validation observations for {run["name"]}. '
                    "Choose a new dataset or the original held-out test partition."
                )
        return common, same_count

    def _compare(self, job_id, payload):
        records = [self._ready(identifier) for identifier in payload["run_ids"]]
        snapshot = self._snapshot(payload["dataset_id"])
        timestamps = {
            row["id"]: row["timestamp_seconds"] for row in snapshot["observations"]
        }
        wanted, same_count = self._comparison_population(payload, snapshot, records)
        self._change(
            job_id,
            stage="evaluating",
            message=f"Evaluating {len(records)} models on {len(wanted)} common rows.",
        )
        joined, model_results = {}, []
        for index, (run, metadata, folder) in enumerate(records):
            self._change(
                job_id,
                message=f'Evaluating {run["name"]} ({index + 1}/{len(records)}).',
            )
            config = self.store.dataset_config(payload["dataset_id"], run["view"])
            # Numeric models can score only the shared population. Temporal
            # models still receive the complete dataset for graph history.
            subset = {"evaluation_ids": wanted} if run["view"] == "numeric" else {}
            report = evaluate_artifact(
                folder / "artifact", config, ROOT, partition="all", **subset
            )
            by_id = {row["id"]: row for row in report["rows"]}
            if len(by_id) != len(report["rows"]) or not wanted <= set(by_id):
                raise ValueError(
                    "Model predictions must contain every common evaluation ID exactly once."
                )
            rows = [row for row in report["rows"] if row["id"] in wanted]
            is_temporal = run["view"] == "labeled_graph"
            threshold = float(report["threshold"])
            probability_threshold = (
                -math.expm1(-max(0.0, threshold) * math.log(2))
                if is_temporal
                else threshold
            )
            labels, probabilities, decisions = [], [], []
            for row in rows:
                probability = float(
                    row["fraud_probability"] if is_temporal else row["probability"]
                )
                if not math.isfinite(probability) or not 0 <= probability <= 1:
                    raise ValueError(
                        "Comparison needs finite fraud probabilities, not link likelihood scores."
                    )
                label = row["label"]
                if label not in (None, 0, 1):
                    raise ValueError(
                        "Comparison outcomes must be fraud, legitimate or unknown."
                    )
                identifier = row["id"]
                if identifier not in joined:
                    joined[identifier] = {
                        "id": identifier,
                        "timestamp_seconds": timestamps[identifier],
                        "label": label,
                        "predictions": {},
                    }
                elif joined[identifier]["label"] != label:
                    raise ValueError(
                        "Model dataset views disagree on evaluation labels for "
                        + identifier
                        + "."
                    )
                joined[identifier]["predictions"][run["id"]] = {
                    "probability": probability,
                    "decision": row["decision"],
                    "score": float(row["score"]) if is_temporal else probability,
                    "score_units": "fraud-surprise-bits"
                    if is_temporal
                    else "fraud-probability",
                }
                labels.append(-1 if label is None else label)
                probabilities.append(probability)
                decisions.append(row["decision"] == "BLOCK")
            # Use saved decisions directly to avoid a rounding change while
            # converting surprise-bit thresholds back to probabilities.
            statistics = metrics(
                np.asarray(labels), np.asarray(decisions, dtype=float), 0.5
            )
            known = np.asarray(labels) >= 0
            truth, values = np.asarray(labels)[known], np.asarray(probabilities)[known]
            both = set(truth.tolist()) == {0, 1}
            if both:
                from sklearn.metrics import average_precision_score, roc_auc_score

                statistics.update(
                    average_precision=float(average_precision_score(truth, values)),
                    roc_auc=float(roc_auc_score(truth, values)),
                )
            else:
                statistics.update(average_precision=None, roc_auc=None)
            clipped = np.clip(values, 1e-15, 1 - 1e-15)
            statistics["log_loss"] = (
                float(
                    np.mean(-truth * np.log(clipped) - (1 - truth) * np.log1p(-clipped))
                )
                if len(truth)
                else None
            )
            statistics["false_block_rate"] = (
                statistics["fp"] / (statistics["fp"] + statistics["tn"])
                if statistics["fp"] + statistics["tn"]
                else None
            )
            model_results.append(
                {
                    "run_id": run["id"],
                    "model_id": run["model_id"],
                    "label": run["name"],
                    "metrics": statistics,
                    "threshold": threshold,
                    "threshold_units": run["threshold_units"],
                    "probability_threshold": probability_threshold,
                    "threshold_source": run["threshold_source"],
                    "model_sha256": run["model_sha256"],
                }
            )
        rows = sorted(
            joined.values(), key=lambda row: (row["timestamp_seconds"], row["id"])
        )
        result = {
            "id": job_id,
            "schema": "pipeline-comparison/v1",
            "dataset_id": payload["dataset_id"],
            "dataset_name": self.store.get(payload["dataset_id"])["name"],
            "dataset_fingerprint": snapshot["fingerprint"],
            "partition": payload["partition"],
            "population": "common-held-out-test" if same_count else "new-dataset",
            "row_count": len(rows),
            "known": sum(row["label"] is not None for row in rows),
            "unknown": sum(row["label"] is None for row in rows),
            "training_rows_in_evaluation": 0,
            "validation_rows_in_evaluation": 0,
            "evaluation_ids": [row["id"] for row in rows],
            "evaluation_bounds": [
                rows[0]["timestamp_seconds"],
                rows[-1]["timestamp_seconds"],
            ],
            "score_semantics": "fraud_probability",
            "metric_definitions": {
                "positive_class": "fraud",
                "unknown_labels": "excluded from outcome metrics",
                "decision_threshold": "Each model retains its training-time validation-selected threshold.",
                "history": "Temporal models replay strictly earlier observed attempts with frozen weights.",
            },
            "models": model_results,
            "rows": rows,
            "created_at": _now(),
        }
        _write(self.root / "comparisons" / (job_id + ".json"), result)
        return result
