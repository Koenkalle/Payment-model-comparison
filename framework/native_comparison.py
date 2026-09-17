"""Run the trained temporal model on the comparison's current payment stream.

Weights, rank references and threshold-validation outcomes come from a separate
historical artifact. The selected dataset supplies observations, never training
labels. Enforcement commits only allowed payments to both neighbor and pair
memory; every simultaneous group is scored before any of its payments commit.
"""
from collections import OrderedDict
import copy
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch

from datasets.implementations.payment_json import normalize
from datasets.implementations.temporal_csv import graph_dataset
from prediction_heads.registry import create_head, manifest as head_manifest
from .fraud_export import (
    _artifact,
    _calibration_prefix,
    _evaluation_ids,
    _partitions,
    _payment_signature,
    _same_graph,
    BROWSER_MAX_ACCOUNTS,
    BROWSER_MAX_EVENTS,
)
from .registry import create_model, load_dataset
from .temporal_experiments import fingerprint, source_hashes
from .temporal_sampling import timestamp_groups


def _metrics(tp, fp, fn, tn):
    return {
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
        "f2": 5 * tp / (5 * tp + fp + 4 * fn) if 5 * tp + fp + 4 * fn else 0,
        "balanced_accuracy": (tp / (tp + fn) + tn / (tn + fp)) / 2
        if tp + fn and tn + fp
        else None,
    }


def frontier(scores, labels):
    """The same strict-cutoff, tie-aware frontier as FraudPolicy.frontier."""
    rows = sorted(zip(map(float, scores), map(int, labels)), reverse=True)
    positives = sum(label == 1 for _, label in rows)
    negatives = sum(label == 0 for _, label in rows)
    if not positives or not negatives:
        return None
    candidates = []
    fp, fn, blocks = 0, positives, 0

    def add(tau):
        if candidates and candidates[-1][2] == fn:
            return
        if candidates and candidates[-1][1] == fp:
            candidates.pop()
        candidates.append([float(tau), fp, fn, blocks])

    add(rows[0][0])
    i = 0
    while i < len(rows):
        score = rows[i][0]
        while i < len(rows) and rows[i][0] == score:
            label = rows[i][1]
            blocks += 1
            fp += label == 0
            fn -= label == 1
            i += 1
        add(rows[i][0] if i < len(rows) else -float(np.finfo(float).eps))
    return {
        "requests": len(rows),
        "positives": positives,
        "negatives": negatives,
        "unknown": len(rows) - positives - negatives,
        "candidates": candidates,
    }


def select_policy(validation, options):
    if validation is None:
        raise ValueError(
            "Historical threshold tuning needs both confirmed fraud and legitimate outcomes in the reserved validation period."
        )
    selected = None
    automatic = options["decisionPolicy"] == "auto"
    for tau, fp, fn, blocks in validation["candidates"]:
        metrics = _metrics(
            validation["positives"] - fn, fp, fn, validation["negatives"] - fp
        )
        value = (
            metrics[options["objective"]]
            if automatic
            else (options["falseBlockCost"] * fp + options["missedFraudCost"] * fn)
        )
        key = (-value if automatic else value, blocks)
        if selected is None or key < selected[0]:
            selected = (
                key,
                {
                    "tau": tau,
                    "fp": fp,
                    "fn": fn,
                    "blocks": blocks,
                    "metrics": metrics,
                    "value" if automatic else "loss": value,
                },
            )
    result = selected[1]
    result.update(
        {
            key: validation[key]
            for key in ("requests", "positives", "negatives", "unknown")
        }
    )
    result["impliedAlpha"] = result["blocks"] / validation["requests"]
    if automatic:
        result.update(
            objective=options["objective"],
            candidatesTested=len(validation["candidates"]),
        )
    else:
        result["costs"] = {
            "falseBlock": options["falseBlockCost"],
            "missedFraud": options["missedFraudCost"],
        }
    return result


def _options(raw, heads, training_mode="unsupervised", default_head=None):
    raw = raw or {}
    options = {
        "mode": raw.get("mode", "enforce"),
        "trainingMode": raw.get("trainingMode", training_mode),
        "decisionPolicy": raw.get("decisionPolicy", "shared"),
        "predictionHead": raw.get("predictionHead", default_head or next(iter(heads))),
        "alpha": raw.get("alpha", 0.02),
        "eta": raw.get("eta", 0.025),
        "warmup": raw.get("warmup", 128),
        "manualTau": raw.get("manualTau"),
        "falseBlockCost": raw.get("falseBlockCost", 1),
        "missedFraudCost": raw.get("missedFraudCost", 20),
        "objective": raw.get("objective", "f1"),
    }
    if options["mode"] not in ("shadow", "enforce"):
        raise ValueError("Unknown comparison execution mode.")
    if options["trainingMode"] != training_mode:
        raise ValueError(
            "This native artifact supports " + training_mode + " scoring only."
        )
    if options["decisionPolicy"] not in ("shared", "manual", "tuned", "auto"):
        raise ValueError("Unknown decision policy.")
    if options["predictionHead"] not in heads:
        raise ValueError("Unknown fraud prediction head.")
    if options["objective"] not in ("f1", "f2", "balanced_accuracy"):
        raise ValueError("Unknown automatic tuning objective.")
    for key in ("alpha", "eta", "falseBlockCost", "missedFraudCost"):
        value = options[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
        ):
            raise ValueError(key + " must be finite.")
    if not 0 < options["alpha"] < 1:
        raise ValueError("The comparison budget must be between zero and one.")
    if (
        options["eta"] < 0
        or options["falseBlockCost"] <= 0
        or options["missedFraudCost"] <= 0
    ):
        raise ValueError("Error costs must be positive and eta nonnegative.")
    if type(options["warmup"]) is not int or options["warmup"] < 1:
        raise ValueError("warmup must be a positive integer.")
    if options["decisionPolicy"] == "manual":
        value = options["manualTau"]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
        ):
            raise ValueError("Manual tau must be finite.")
    elif options["manualTau"] is not None:
        # Unused UI state must still be safe to serialize in a run.
        value = options["manualTau"]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
        ):
            raise ValueError("Manual tau must be finite when supplied.")
    return options


class _AcceptedNeighbors:
    """Bounded first-hop history whose rows exist only after settlement."""

    sample_neighbor_strategy = "recent"

    def __init__(self, graph, limit):
        self.graph = graph
        self.limit = limit
        self.histories = [[] for _ in graph.node_ids]

    def commit(self, indices):
        for index in indices:
            u, v = int(self.graph.sources[index]), int(self.graph.destinations[index])
            timestamp = float(self.graph.times[index])
            for node, neighbor in ((u, v), (v, u)):
                history = self.histories[node]
                history.append((neighbor, int(index) + 1, timestamp))
                if len(history) > self.limit:
                    del history[: -self.limit]

    def get_all_first_hop_neighbors(self, node_ids, node_interact_times):
        neighbors, edges, times = [], [], []
        for node, timestamp in zip(node_ids, node_interact_times):
            rows = [row for row in self.histories[int(node)] if row[2] < timestamp]
            neighbors.append(np.asarray([row[0] for row in rows], dtype=np.int64))
            edges.append(np.asarray([row[1] for row in rows], dtype=np.int64))
            times.append(np.asarray([row[2] for row in rows], dtype=np.float64))
        return neighbors, edges, times


class NativeComparison:
    """A reusable CPU scorer. Callers serialize compare calls on this instance."""

    def __init__(self, artifact, calibration_config=None, base_dir=None):
        self.artifact = Path(artifact).resolve()
        task = json.loads((self.artifact / "manifest.json").read_text()).get("task")
        self.training_mode = (
            "supervised" if task == "temporal-fraud-classification" else "unsupervised"
        )
        if self.training_mode == "supervised":
            self._initialize_supervised()
            self._initialize_caches()
            return
        if calibration_config is None:
            calibration_config = self.artifact / "calibration.config.json"
        if isinstance(calibration_config, (str, Path)):
            filename = Path(calibration_config).resolve()
            config = json.loads(filename.read_text())
            base = filename.parent
        else:
            config = copy.deepcopy(calibration_config)
            base = Path(base_dir or ".").resolve()
        self.metadata = _artifact(self.artifact)
        graph_config = config["dataset"]
        if graph_config.get("loader") != "payment_graph":
            raise ValueError(
                "Native comparison calibration requires the original payment_graph configuration."
            )
        self.original = load_dataset(graph_config, base)
        if fingerprint(self.original) != self.metadata.get("dataset_fingerprint"):
            # File locations are provenance, not model inputs. Permit a cloned
            # repository while still checking every graph array and source byte.
            saved = self.metadata.get("dataset", {})
            relocated = dataclasses.replace(self.original, provenance=saved)
            if (
                not saved.get("source_sha256")
                or self.original.provenance.get("source_sha256")
                != saved["source_sha256"]
                or fingerprint(relocated) != self.metadata.get("dataset_fingerprint")
            ):
                raise ValueError(
                    "Calibration must use the original graph dataset and configuration."
                )
        self.original_events = load_dataset(graph_config["dataset"], base).document
        parts = _partitions(self.metadata, self.original)
        self.reference_indices = _calibration_prefix(
            self.original, parts["test"], config.get("max_rows", 100)
        )
        self.validation_indices = parts["test"][len(self.reference_indices) :]
        self.model, self.descriptor = create_model(
            self.metadata["model_id"], self.original.schema
        )
        if source_hashes(self.descriptor) != self.metadata.get("implementation_sha256"):
            raise ValueError(
                "Model implementation changed since this artifact was created."
            )
        self.model.load_graph(self.artifact / "model.npz", self.original)
        historical_logits = self._shadow_logits(self.original)
        reference = historical_logits[self.reference_indices]
        self.heads = {
            entry["id"]: create_head(entry["id"]).fit(reference)
            for entry in head_manifest()["prediction_heads"]
        }
        truth = self.original_events.get("truth", {})
        labels = [
            int(truth[self.original.ids[index]])
            if self.original.ids[index] in truth
            else -1
            for index in self.validation_indices
        ]
        self.frontiers = {
            identifier: frontier(
                head.score(historical_logits[self.validation_indices])["score"], labels
            )
            for identifier, head in self.heads.items()
        }
        self.calibration = {
            "ids": [self.original.ids[index] for index in self.reference_indices],
            "count": len(reference),
            "partition": "test-prefix",
            "dataset_fingerprint": self.metadata["dataset_fingerprint"],
            "minimum_tail_probability": 1 / (len(reference) + 1),
            "threshold_validation_ids": [
                self.original.ids[index] for index in self.validation_indices
            ],
            "threshold_validation_count": len(self.validation_indices),
            "note": "Frozen rank references and separate later threshold validation from the historical artifact. Selected dataset outcomes are never used for fitting.",
        }
        self._initialize_caches()

    def _initialize_caches(self):
        self._shadow_cache = OrderedDict()
        self._enforce_cache = OrderedDict()
        self.inference_calls = 0

    def describe(self):
        labels = {
            entry["id"]: entry.get("label", entry["id"])
            for entry in head_manifest(kind=None)["prediction_heads"]
        }
        return {
            "id": self.descriptor["id"],
            "label": "DyGFormer + TAMI · native",
            "checkpoint_id": self.metadata["model_sha256"],
            "artifact": {
                "path": str(self.artifact),
                "model_sha256": self.metadata["model_sha256"],
                "training_mode": self.training_mode,
                "encoder_training": self.metadata.get("encoder_training"),
                "best_epoch": self.metadata.get("best_epoch"),
                "epochs_completed": len(self.metadata.get("training_history", [])),
                "training_dataset": self.metadata.get("dataset", {}).get("source_name"),
            },
            "feature_contract": (
                self.metadata.get("feature_schema")
                if self.training_mode == "supervised"
                else self.metadata.get("features")
            ),
            "execution": "python",
            "training_modes": [self.training_mode],
            "supported_modes": ["shadow", "enforce"],
            "supported_policies": ["shared", "manual"]
            + (["tuned", "auto"] if all(self.frontiers.values()) else []),
            "prediction_heads": [
                {"id": identifier, "label": labels.get(identifier, identifier)}
                for identifier in self.heads
            ],
            "default_head": next(iter(self.heads)),
            "calibration": copy.deepcopy(self.calibration),
        }

    def _initialize_supervised(self):
        from .temporal_fraud_experiments import load_artifact
        from .temporal_fraud_data import available_labels
        from prediction_heads.implementations.fraud_linear import FraudLogitScorer

        self.metadata, labeled, self.model = load_artifact(self.artifact)
        self.original = labeled.graph
        self.original_events = labeled.document
        # The task resolver supplies the source-checked implementation descriptor;
        # all tensors and data were already verified by load_artifact.
        _, self.descriptor = create_model(
            self.metadata["model_id"],
            labeled.schema,
            task="temporal-fraud-classification",
        )
        positions = {identifier: i for i, identifier in enumerate(self.original.ids)}
        self.reference_indices = np.empty(0, dtype=np.int64)
        self.validation_indices = np.asarray(
            [
                positions[identifier]
                for identifier in self.metadata["split"]["policy_validation"]
            ],
            dtype=np.int64,
        )
        historical_logits = self._shadow_logits(self.original)
        head = FraudLogitScorer(self.model.head_id)
        if self.metadata["prediction_head"]["id"] != head.id:
            raise ValueError("Unsupported supervised artifact prediction head.")
        self.heads = {head.id: head}
        cutoff = self.metadata.get("label_cutoffs", {}).get("policy_validation")
        labels = available_labels(labeled, cutoff)[self.validation_indices]
        self.frontiers = {
            head.id: frontier(
                head.score(historical_logits[self.validation_indices])["score"], labels
            )
        }
        self.calibration = {
            "ids": [],
            "count": 0,
            "partition": "policy_validation",
            "dataset_fingerprint": self.metadata["dataset_fingerprint"],
            "threshold_validation_ids": [
                self.original.ids[index] for index in self.validation_indices
            ],
            "threshold_validation_count": len(self.validation_indices),
            "label_availability_cutoff": cutoff,
            "note": "Fraud classifier weights are frozen. Thresholds use only the reserved historical policy-validation period; selected dataset outcomes are reporting labels only.",
        }

    def _evaluation(self, graph, document, options):
        if self.training_mode == "supervised":
            if _same_graph(self.original, graph):
                return list(self.metadata["split"]["test"]), True
            reserved = {
                identifier
                for partition in ("train", "model_validation", "policy_validation")
                for identifier in self.metadata["split"][partition]
            }
            used = {
                _payment_signature(self.original_events, event)
                for event in self.original_events["events"]
                if event["kind"] == "payment" and event["id"] in reserved
            }
            if any(
                _payment_signature(document, event) in used
                for event in document["events"]
                if event["kind"] == "payment"
            ):
                raise ValueError(
                    "Target overlaps historical training, model-validation or policy-validation transactions; use the complete original dataset for held-out test selection or a separate dataset."
                )
            return list(graph.ids), False
        evaluation, same_dataset = _evaluation_ids(
            self.metadata,
            self.reference_indices,
            self.original,
            self.original_events,
            graph,
            document,
        )
        if same_dataset and options["decisionPolicy"] in ("tuned", "auto"):
            raise ValueError(
                "The selected dataset supplied historical threshold-validation outcomes. Use a separate comparison dataset for tuned or automatic policies."
            )
        if options["decisionPolicy"] in ("tuned", "auto"):
            validation_ids = {
                self.original.ids[index] for index in self.validation_indices
            }
            used = {
                _payment_signature(self.original_events, event)
                for event in self.original_events["events"]
                if event["kind"] == "payment" and event["id"] in validation_ids
            }
            if any(
                _payment_signature(document, event) in used
                for event in document["events"]
                if event["kind"] == "payment"
            ):
                raise ValueError(
                    "The selected dataset overlaps historical threshold-validation transactions. Use a separate dataset for tuned or automatic policies."
                )
        return evaluation, same_dataset

    def _shadow_logits(self, graph, cancelled=lambda: False):
        self.model._bind(graph)
        self.model.network.eval()
        logits = np.empty(len(graph.ids))
        with torch.no_grad():
            for group in timestamp_groups(graph, np.arange(len(graph.ids))):
                if cancelled():
                    raise InterruptedError("Comparison superseded by newer settings.")
                values, _, proposed = self.model.score_group(graph, group)
                logits[group] = values.cpu().numpy()
                self.model._commit(graph, group, proposed)
        if not np.isfinite(logits).all():
            raise ValueError("Native inference returned nonfinite logits.")
        return logits

    def _graph(self, document):
        events = [event for event in document["events"] if event["kind"] == "payment"]
        names = [account["external_id"] for account in document["accounts"]]
        return graph_dataset(
            [event["id"] for event in events],
            [event["t"] * 60 for event in events],
            [names[event["u"]] for event in events],
            [names[event["v"]] for event in events],
            [[np.log1p(event["amount"])] for event in events],
            ["log1p_amount"],
            {"origin": "comparison-current-dataset"},
            len(self.original.node_feature_names),
        )

    def compare(self, document, options=None, *, cancelled=lambda: False):
        if cancelled():
            raise InterruptedError("Comparison superseded by newer settings.")
        options = _options(options, self.heads, self.training_mode)
        document = normalize(document)
        if (
            len(document["accounts"]) > BROWSER_MAX_ACCOUNTS
            or len(document["events"]) > BROWSER_MAX_EVENTS
        ):
            raise ValueError(
                "Comparison supports at most 256 accounts and 20,000 events."
            )
        graph = self._graph(document)
        evaluation, same_dataset = self._evaluation(graph, document, options)
        eligible = set(evaluation)
        head = self.heads[options["predictionHead"]]
        validation = self.frontiers[head.id]
        fit = (
            select_policy(validation, options)
            if options["decisionPolicy"] in ("tuned", "auto")
            else None
        )
        tau = (
            options["manualTau"]
            if options["decisionPolicy"] == "manual"
            else fit["tau"]
            if fit
            else None
        )
        policy_state = {
            **{
                key: options[key]
                for key in (
                    "decisionPolicy",
                    "alpha",
                    "warmup",
                    "eta",
                    "mode",
                    "predictionHead",
                    "trainingMode",
                )
            },
            "errorCosts": {
                "falseBlock": options["falseBlockCost"],
                "missedFraud": options["missedFraudCost"],
            },
            "policyFit": fit,
            "tau": tau,
        }
        warmup = []
        rows = []

        def decisions(group, logits):
            nonlocal tau
            result = head.score(np.asarray(logits, dtype=float))
            accepted = []
            for offset, index in enumerate(group):
                identifier = graph.ids[index]
                score = float(result["score"][offset])
                context = identifier not in eligible
                before = tau
                decision = (
                    "CONTEXT"
                    if context
                    else "LEARNING"
                    if tau is None
                    else "BLOCK"
                    if score > tau
                    else "ALLOW"
                )
                settled = options["mode"] == "shadow" or decision != "BLOCK"
                if settled:
                    accepted.append(offset)
                if options["decisionPolicy"] == "shared" and not context:
                    if decision == "LEARNING":
                        warmup.append(score)
                        if len(warmup) >= options["warmup"]:
                            position = max(
                                0,
                                min(
                                    len(warmup) - 1,
                                    int(np.ceil((1 - options["alpha"]) * len(warmup)))
                                    - 1,
                                ),
                            )
                            tau = float(sorted(warmup)[position])
                    else:
                        tau += options["eta"] * (
                            int(decision == "BLOCK") - options["alpha"]
                        )
                logit = float(logits[offset])
                evidence = (
                    {
                        "kind": "native-fraud",
                        "version": 1,
                        "fraud_logit": logit,
                        "fraud_probability": float(result["fraud_probability"][offset]),
                        "head": head.id,
                        "alpha": options["alpha"],
                    }
                    if self.training_mode == "supervised"
                    else {
                        "kind": "native-likelihood",
                        "logit": logit,
                        "likelihood_probability": float(
                            np.exp(-np.logaddexp(0, -logit))
                        ),
                        "tail_probability": float(result["tail_probability"][offset]),
                        "head": head.id,
                        "reference_count": head.reference_count,
                        "alpha": options["alpha"],
                    }
                )
                rows.append(
                    {
                        "id": identifier,
                        "logit": logit,
                        "score": score,
                        **(
                            {"fraud_logit": logit}
                            if self.training_mode == "supervised"
                            else {}
                        ),
                        "tauBefore": before,
                        "tauAfter": tau,
                        "decision": decision,
                        "settled": settled,
                        "evaluationEligible": not context,
                        "evidence": evidence,
                    }
                )
            return accepted

        groups = timestamp_groups(graph, np.arange(len(graph.ids)))
        cache_hit = False
        enforce_cache_hit = False
        if options["mode"] == "shadow":
            key = fingerprint(graph)
            if key in self._shadow_cache:
                logits = self._shadow_cache.pop(key)
                cache_hit = True
            else:
                logits = self._shadow_logits(graph, cancelled)
                self.inference_calls += 1
            self._shadow_cache[key] = logits
            while len(self._shadow_cache) > 4:
                self._shadow_cache.popitem(last=False)
            for group in groups:
                if cancelled():
                    raise InterruptedError("Comparison superseded by newer settings.")
                decisions(group, logits[group])
        else:
            # A policy's irrelevant controls cannot alter accepted history. For
            # example, changing the displayed budget under a manual cutoff only
            # changes evidence; it does not require replaying the neural model.
            policy_fields = {
                "shared": ("alpha", "warmup", "eta"),
                "manual": ("manualTau",),
                "tuned": ("falseBlockCost", "missedFraudCost"),
                "auto": ("objective",),
            }[options["decisionPolicy"]]
            identity = (
                fingerprint(graph),
                head.id,
                options["decisionPolicy"],
                *(options[key] for key in policy_fields),
            )
            if identity in self._enforce_cache:
                cached = self._enforce_cache.pop(identity)
                self._enforce_cache[identity] = cached
                enforce_cache_hit = True
                for group in groups:
                    if cancelled():
                        raise InterruptedError(
                            "Comparison superseded by newer settings."
                        )
                    decisions(group, cached[group])
            else:
                self.model._bind(graph, [])
                neighbors = _AcceptedNeighbors(
                    graph, self.model.parameters["max_input_sequence_length"] - 1
                )
                self.model.network[0].neighbor_sampler = neighbors
                self.model.network.eval()
                self.inference_calls += 1
                cached = np.empty(len(graph.ids))
                with torch.no_grad():
                    for group in groups:
                        if cancelled():
                            raise InterruptedError(
                                "Comparison superseded by newer settings."
                            )
                        logits, _, proposed = self.model.score_group(graph, group)
                        values = logits.cpu().numpy()
                        if not np.isfinite(values).all():
                            raise ValueError(
                                "Native inference returned nonfinite logits."
                            )
                        cached[group] = values
                        accepted = decisions(group, values)
                        if accepted:
                            accepted_group = group[accepted]
                            self.model._commit(
                                graph, accepted_group, proposed[accepted]
                            )
                            neighbors.commit(accepted_group)
                self._enforce_cache[identity] = cached
                while len(self._enforce_cache) > 4:
                    self._enforce_cache.popitem(last=False)
        result = {
            "version": 2,
            "schema": "native-fraud-comparison/v2",
            "dataset": document,
            "model": {
                "id": self.descriptor["id"],
                "label": "DyGFormer + TAMI · native",
                "checkpoint_id": self.metadata["model_sha256"],
                "implementation": self.descriptor,
                "parameters": sum(
                    parameter.numel() for parameter in self.model.network.parameters()
                ),
                "policy_validation": {self.training_mode: validation}
                if validation
                else {},
                "training": {
                    "mode": self.training_mode,
                    "task": self.metadata["task"],
                    "encoder_training": self.metadata.get("encoder_training"),
                    "head": head.id,
                    "label_counts": self.metadata.get("label_counts"),
                },
            },
            "predictions": rows,
            "heads": {
                identifier: value.to_dict() for identifier, value in self.heads.items()
            },
            "head": head.id,
            "alpha": options["alpha"],
            "options": options,
            "policy_state": policy_state,
            "policyFit": fit,
            "evaluation_ids": evaluation,
            "calibration": copy.deepcopy(self.calibration),
            "provenance": {
                "artifact_model_sha256": self.metadata["model_sha256"],
                "task": self.metadata["task"],
                "artifact_dataset": self.metadata["dataset"],
                "same_dataset": same_dataset,
                "implementation_sha256": self.metadata["implementation_sha256"],
                "model_parameters": self.metadata.get("parameters", {}),
                "evaluation_scope": (
                    "saved-test"
                    if self.training_mode == "supervised"
                    else "saved-test-after-calibration"
                )
                if same_dataset
                else "separate-target-dataset",
                "inference": "native-cpu",
                "shadow_cache_hit": cache_hit,
                "enforce_cache_hit": enforce_cache_hit,
            },
            "history": {
                "mode": options["mode"],
                "payments": "observed-attempts"
                if options["mode"] == "shadow"
                else "accepted-payments",
                "timestamps": "strictly-before",
                "deposits": False,
                "reports": False,
            },
            "score_meaning": (
                "Fraud probability estimated from temporal interaction context, pair memory and candidate amount. Score = softplus(fraud logit) / ln(2); a score cutoff of 1 corresponds to estimated fraud probability above 0.5. Probabilities are not calibrated."
                if self.training_mode == "supervised"
                else "Low observed-link likelihood indicates unusual endpoints and timing. Current payment amount is not a scoring input. Neither link likelihood nor empirical tail rank is fraud probability."
            ),
        }
        # Check the complete response, including option/metadata fields, before
        # returning it to an HTTP transport or a comparison adapter.
        json.dumps(result, allow_nan=False)
        return result
