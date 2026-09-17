"""Actual Torch scoring, policy parity, independent calibration and settlement."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from framework.experiments import train_experiment

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "optional PyTorch dependency")
class NativeComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from framework.native_comparison import NativeComparison

        cls.temporary = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temporary.name)
        cls.document = {
            "accounts": [
                {"id": i, "external_id": "historical-" + str(i)} for i in range(5)
            ],
            "events": [
                {
                    "id": "deposit",
                    "kind": "deposit",
                    "u": -1,
                    "v": 0,
                    "t": 0,
                    "amount": 1000,
                }
            ],
            "truth": {},
        }
        for i in range(80):
            cls.document["events"].append(
                {
                    "id": "payment-" + str(i),
                    "kind": "payment",
                    "u": i % 2,
                    "v": 2 + (i % 2 if i % 7 else 2),
                    "t": i // 2,
                    "amount": 10 + i % 5,
                }
            )
            cls.document["truth"]["payment-" + str(i)] = i % 4 == 1
        (cls.base / "payments.json").write_text(json.dumps(cls.document))
        graph_config = {
            "loader": "payment_graph",
            "dataset": {"loader": "payment_json", "path": "payments.json"},
            "node_feature_dim": 4,
        }
        cls.graph_config = graph_config
        parameters = dict(
            epochs=1,
            patience=1,
            batch_size=8,
            learning_rate=0.003,
            time_feat_dim=4,
            channel_embedding_dim=4,
            num_layers=1,
            patch_size=2,
            max_input_sequence_length=8,
            dropout=0.0,
            seed=31,
        )
        train_experiment(
            {
                "model": "dyg_tami_native",
                "dataset": graph_config,
                "parameters": parameters,
            },
            cls.base / "artifact",
            cls.base,
        )
        cls.calibration = {"dataset": graph_config, "max_rows": 3}
        (cls.base / "calibration.json").write_text(json.dumps(cls.calibration))
        cls.scorer = NativeComparison(
            cls.base / "artifact", cls.base / "calibration.json"
        )
        cls.target = copy.deepcopy(cls.document)
        for account in cls.target["accounts"]:
            account["external_id"] = "new-" + account["external_id"]

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_true_native_logits_match_model_and_shadow_head_changes_reuse_them(self):
        from framework.registry import create_model

        scorer = self.scorer
        result = scorer.compare(
            self.target, {"mode": "shadow", "decisionPolicy": "manual", "manualTau": 2}
        )
        graph = scorer._graph(result["dataset"])
        native, _ = create_model("dyg_tami_native", graph.schema)
        native.load_graph(self.base / "artifact" / "model.npz", graph)
        expected = native.predict_graph(graph, np.arange(len(graph.ids)))[
            "positive_logits"
        ]
        np.testing.assert_array_equal(
            [row["logit"] for row in result["predictions"]], expected
        )
        calls = scorer.inference_calls
        changed = scorer.compare(
            self.target,
            {
                "mode": "shadow",
                "decisionPolicy": "manual",
                "manualTau": 0.8,
                "predictionHead": "fixed_likelihood",
            },
        )
        self.assertEqual(scorer.inference_calls, calls)
        self.assertTrue(changed["provenance"]["shadow_cache_hit"])
        self.assertEqual(
            [row["logit"] for row in result["predictions"]],
            [row["logit"] for row in changed["predictions"]],
        )
        self.assertNotEqual(
            [row["score"] for row in result["predictions"]],
            [row["score"] for row in changed["predictions"]],
        )
        self.assertEqual(len(result["evaluation_ids"]), 80)
        self.assertEqual(result["calibration"]["count"], 4)
        self.assertEqual(result["calibration"]["threshold_validation_count"], 12)

    def test_selected_truth_cannot_change_any_policy_head_or_prediction(self):
        target = copy.deepcopy(self.target)
        target["truth"] = {key: not value for key, value in target["truth"].items()}
        for mode in ("shadow", "enforce"):
            for policy in ("shared", "manual", "tuned", "auto"):
                options = {
                    "mode": mode,
                    "decisionPolicy": policy,
                    "manualTau": 2,
                    "warmup": 4,
                }
                original = self.scorer.compare(self.target, options)
                changed = self.scorer.compare(target, options)
                self.assertEqual(original["predictions"], changed["predictions"])
                self.assertEqual(original["policy_state"], changed["policy_state"])
                self.assertEqual(original["heads"], changed["heads"])
        self.assertEqual(
            self.scorer.describe()["supported_policies"],
            ["shared", "manual", "tuned", "auto"],
        )

    def test_enforce_cache_tracks_effective_policy_and_ignores_inactive_controls(self):
        target = copy.deepcopy(self.target)
        for account in target["accounts"]:
            account["external_id"] += "-policy-cache"
        for policy in ("manual", "tuned", "auto"):
            options = {"mode": "enforce", "decisionPolicy": policy, "manualTau": 1.1}
            first = self.scorer.compare(target, options)
            calls = self.scorer.inference_calls
            second = self.scorer.compare(
                target, {**options, "alpha": 0.07, "warmup": 10, "eta": 0.5}
            )
            self.assertEqual(self.scorer.inference_calls, calls)
            self.assertTrue(second["provenance"]["enforce_cache_hit"])
            self.assertEqual(
                [row["logit"] for row in first["predictions"]],
                [row["logit"] for row in second["predictions"]],
            )
            self.assertEqual(
                [row["decision"] for row in first["predictions"]],
                [row["decision"] for row in second["predictions"]],
            )
            self.assertTrue(
                all(row["evidence"]["alpha"] == 0.07 for row in second["predictions"])
            )
        for options in (
            {"mode": "enforce", "decisionPolicy": "manual", "manualTau": 1.2},
            {
                "mode": "enforce",
                "decisionPolicy": "manual",
                "manualTau": 1.2,
                "predictionHead": "fixed_likelihood",
            },
            {"mode": "enforce", "decisionPolicy": "shared", "warmup": 4, "alpha": 0.1},
            {"mode": "enforce", "decisionPolicy": "shared", "warmup": 4, "alpha": 0.2},
            {"mode": "shadow", "decisionPolicy": "manual", "manualTau": 1.2},
        ):
            calls = self.scorer.inference_calls
            self.scorer.compare(target, options)
            self.assertEqual(self.scorer.inference_calls, calls + 1)

    def test_enforcement_commits_only_accepted_history(self):
        from framework.registry import create_model
        from framework.temporal_sampling import timestamp_groups
        from models.implementations.temporal_history import NeighborSampler
        import torch

        options = {
            "mode": "enforce",
            "decisionPolicy": "shared",
            "alpha": 0.4,
            "warmup": 4,
        }
        actual = self.scorer.compare(self.target, options)
        graph = self.scorer._graph(actual["dataset"])
        oracle, _ = create_model("dyg_tami_native", graph.schema)
        oracle.load_graph(self.base / "artifact" / "model.npz", graph)
        oracle._bind(graph, [])
        accepted = []
        expected = np.empty(len(graph.ids))
        with torch.no_grad():
            for group in timestamp_groups(graph, np.arange(len(graph.ids))):
                oracle.network[0].neighbor_sampler = NeighborSampler(graph, accepted, 7)
                logits, _, proposed = oracle.score_group(graph, group)
                expected[group] = logits.numpy()
                kept = [
                    offset
                    for offset, index in enumerate(group)
                    if actual["predictions"][index]["settled"]
                ]
                if kept:
                    oracle._commit(graph, group[kept], proposed[kept])
                    accepted.extend(group[kept].tolist())
        self.assertTrue(0 < len(accepted) < len(graph.ids))
        np.testing.assert_array_equal(
            [row["logit"] for row in actual["predictions"]], expected
        )
        allowed = self.scorer.compare(
            self.target,
            {"mode": "enforce", "decisionPolicy": "manual", "manualTau": 100},
        )
        shadow = self.scorer.compare(
            self.target,
            {"mode": "shadow", "decisionPolicy": "manual", "manualTau": 100},
        )
        self.assertEqual(
            [row["logit"] for row in allowed["predictions"]],
            [row["logit"] for row in shadow["predictions"]],
        )
        self.assertNotEqual(
            [row["logit"] for row in actual["predictions"]],
            [row["logit"] for row in shadow["predictions"]],
        )

    def test_blocked_amounts_do_not_enter_future_history_or_same_time_scores(self):
        changed = copy.deepcopy(self.target)
        for event in changed["events"]:
            event["amount"] *= 100
        options = {"mode": "enforce", "decisionPolicy": "manual", "manualTau": -1}
        first = self.scorer.compare(self.target, options)
        second = self.scorer.compare(changed, options)
        self.assertTrue(
            all(
                row["decision"] == "BLOCK" and not row["settled"]
                for row in first["predictions"]
            )
        )
        self.assertEqual(first["predictions"], second["predictions"])
        shadow = self.scorer.compare(self.target, {**options, "mode": "shadow"})
        changed_shadow = self.scorer.compare(changed, {**options, "mode": "shadow"})
        # The first two payments share a timestamp and see no current amounts.
        self.assertEqual(shadow["predictions"][:2], changed_shadow["predictions"][:2])
        self.assertNotEqual(
            shadow["predictions"][2:], changed_shadow["predictions"][2:]
        )

    def test_policies_match_existing_javascript_frontier_and_selection(self):
        from framework.native_comparison import frontier, select_policy, _options

        node = shutil.which("node") or "/tmp/payment-tools-node/bin/node"
        if not Path(node).exists():
            self.skipTest("Node not installed")
        scores, labels = [5, 4, 4, 3, 2, 1, 0], [0, 1, 1, -1, 0, 1, 0]
        validation = frontier(scores, labels)
        script = "const p=require('./shared/runtime/policy'); const x=JSON.parse(process.argv[1]); const v=p.frontier(x.rows); console.log(JSON.stringify({v,tuned:p.select(v,x.options),auto:p.autoTune(v,x.options.objective)}));"
        for objective in ("f1", "f2", "balanced_accuracy"):
            options = _options(
                {
                    "decisionPolicy": "tuned",
                    "falseBlockCost": 3,
                    "missedFraudCost": 5,
                    "objective": objective,
                },
                self.scorer.heads,
            )
            request = {
                "rows": [
                    {"score": s, "label": label} for s, label in zip(scores, labels)
                ],
                "options": options,
            }
            raw = subprocess.check_output(
                [node, "-e", script, json.dumps(request)], cwd=ROOT, text=True
            )
            expected = json.loads(raw)
            self.assertEqual(validation, expected["v"])
            self.assertEqual(select_policy(validation, options), expected["tuned"])
            self.assertEqual(
                select_policy(validation, {**options, "decisionPolicy": "auto"}),
                expected["auto"],
            )

    def test_relocated_artifact_validates_bytes_graph_and_rejects_historical_reuse(
        self,
    ):
        from framework.native_comparison import NativeComparison

        relocated = self.base / "relocated"
        relocated.mkdir(exist_ok=True)
        shutil.copytree(
            self.base / "artifact", relocated / "artifact", dirs_exist_ok=True
        )
        shutil.copyfile(self.base / "payments.json", relocated / "payments.json")
        (relocated / "calibration.json").write_text(json.dumps(self.calibration))
        scorer = NativeComparison(
            relocated / "artifact", relocated / "calibration.json"
        )
        self.assertEqual(scorer.calibration, self.scorer.calibration)
        same = scorer.compare(
            self.document,
            {"mode": "shadow", "decisionPolicy": "manual", "manualTau": 2},
        )
        self.assertTrue(same["provenance"]["same_dataset"])
        self.assertEqual(
            sum(row["decision"] == "CONTEXT" for row in same["predictions"]), 68
        )
        with self.assertRaisesRegex(ValueError, "historical threshold-validation"):
            scorer.compare(self.document, {"decisionPolicy": "auto"})
        subset = copy.deepcopy(self.document)
        subset["events"] = subset["events"][-8:]
        subset["truth"] = {}
        for event in subset["events"]:
            event["id"] = "renamed-" + event["id"]
        with self.assertRaisesRegex(ValueError, "historical threshold-validation"):
            scorer.compare(subset, {"decisionPolicy": "tuned"})
        modified = copy.deepcopy(self.document)
        modified["events"][1]["amount"] += 1
        (relocated / "payments.json").write_text(json.dumps(modified))
        with self.assertRaisesRegex(ValueError, "original graph"):
            NativeComparison(relocated / "artifact", relocated / "calibration.json")

    def test_cancelled_inference_never_caches_partial_results_and_can_restart(self):
        target = copy.deepcopy(self.target)
        for account in target["accounts"]:
            account["external_id"] += "-cancelled"
        graph = self.scorer._graph(target)
        from framework.temporal_experiments import fingerprint

        for mode in ("shadow", "enforce"):
            ticks = [0]

            def cancelled():
                ticks[0] += 1
                return ticks[0] > 3

            with self.assertRaisesRegex(InterruptedError, "superseded"):
                self.scorer.compare(target, {"mode": mode}, cancelled=cancelled)
            self.assertNotIn(fingerprint(graph), self.scorer._shadow_cache)
        result = self.scorer.compare(target, {"mode": "shadow"})
        self.assertEqual(len(result["predictions"]), 80)
        self.assertFalse(result["provenance"]["shadow_cache_hit"])


if __name__ == "__main__":
    unittest.main()
