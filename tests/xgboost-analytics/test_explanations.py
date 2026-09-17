"""Independent leaf-enumeration / permutation Shapley oracle and export checks.

Run with --write-fixture to regenerate the tiny cross-language oracle fixture.
The JS engine instead enumerates feature subsets recursively through each tree.
"""
import itertools
import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.export_xgboost_explanations import (
    count_tree,
    fingerprint,
    model_fingerprint,
)
from training.train import AMOUNT, XGB_FEATURES, xgb_rows


def leaf_paths(tree, counts):
    """Flatten to disjoint leaves with independent conditional branch weights."""
    result = []
    cursor = 0

    def walk(node, path):
        nonlocal cursor
        count = counts[cursor]
        cursor += 1
        if "leaf" in node:
            result.append((node["leaf"], path))
            return
        for direction in ("left", "right"):
            weight = counts[cursor] / count
            walk(
                node[direction],
                path + [(node["feature"], node["threshold"], direction, weight)],
            )

    walk(tree, [])
    return result


def oracle(model, counts, row):
    leaves = [leaf_paths(tree, cover) for tree, cover in zip(model["trees"], counts)]
    # Include a dummy unused feature explicitly in every permutation.
    used = sorted(
        {step[0] for tree in leaves for _, path in tree for step in path} | {3}
    )

    def expectation(known):
        total = model["base_score"]
        for tree in leaves:
            expected = 0
            for output, path in tree:
                weight = 1
                for feature, threshold, branch, prior in path:
                    if feature in known:
                        if (row[feature] <= threshold) != (branch == "left"):
                            weight = 0
                            break
                    else:
                        weight *= prior
                expected += output * weight
            total += model["learning_rate"] * expected
        return total

    baseline = expectation(set())
    contributions = [0.0] * len(row)
    for order in itertools.permutations(used):
        known, previous = set(), baseline
        for feature in order:
            known.add(feature)
            current = expectation(known)
            contributions[feature] += current - previous
            previous = current
    contributions = [value / math.factorial(len(used)) for value in contributions]
    return {
        "baseline": baseline,
        "contributions": contributions,
        "rawMargin": expectation(set(used)),
    }


def fixture():
    model = {
        "features": XGB_FEATURES,
        "amount_bins": AMOUNT,
        "base_score": -0.3,
        "learning_rate": 0.2,
        "checkpoint_id": "independent-oracle",
        "trees": [
            {
                "feature": 0,
                "threshold": 0.5,
                "left": {
                    "feature": 1,
                    "threshold": 0.4,
                    "left": {
                        "feature": 0,
                        "threshold": 0.2,
                        "left": {"leaf": -2.0},
                        "right": {"leaf": 1.0},
                    },
                    "right": {"leaf": 0.5},
                },
                "right": {
                    "feature": 2,
                    "threshold": 0.6,
                    "left": {"leaf": 4.0},
                    "right": {"leaf": -3.0},
                },
            },
            {
                "feature": 0,
                "threshold": 0.5,
                "left": {"leaf": 1.0},
                "right": {
                    "feature": 1,
                    "threshold": 0.5,
                    "left": {"leaf": 4.0},
                    "right": {"leaf": -3.0},
                },
            },
            {"leaf": 0.7},
        ],
    }
    reference_rows = [
        [0, 0, 0],
        [0.4, 0.2, 0.8],
        [0.3, 0.8, 0.2],
        [0.8, 0, 0.2],
        [1, 1, 1],
    ]
    counts = [count_tree(tree, reference_rows) for tree in model["trees"]]
    sidecar = {
        "version": 1,
        "feature_schema_version": 1,
        "features": XGB_FEATURES,
        "checkpoint_id": model["checkpoint_id"],
        "checkpoint_sha256": "0" * 64,
        "model_fingerprint": model_fingerprint(model),
        "reference": {
            "convention": "unweighted_labeled_fitting_rows",
            "row_count": len(reference_rows),
        },
        "node_counts": counts,
    }
    sidecar["reference_id"] = fingerprint(sidecar)
    samples = [
        [0.1, 0.1, 0],
        [0.1, 0.1, 1],
        [0.1, 0.9, 0],
        [0.4, 0.1, 1],
        [0.7, 0.2, 0.3],
        [1, 1, 1],
        [0.5, 0.4, 0.6],
    ]
    cases = []
    for sample in samples:
        values = sample + [0] * (len(XGB_FEATURES) - len(sample))
        cases.append({"values": values, "expected": oracle(model, counts, values)})
    return {
        "method": "Exact permutation Shapley values; independent flattened leaf-path expectations.",
        "model": model,
        "sidecar": sidecar,
        "cases": cases,
    }


class ExplanationExportTests(unittest.TestCase):
    def test_unknown_rows_affect_history_but_are_not_reference_rows(self):
        events = [
            {
                "kind": "payment",
                "id": "unknown",
                "u": 0,
                "v": 1,
                "t": 1,
                "amount": 200,
                "label": -1,
            },
            {
                "kind": "payment",
                "id": "known",
                "u": 0,
                "v": 1,
                "t": 2,
                "amount": 100,
                "label": 1,
            },
        ]
        rows, labels, episode_ids = xgb_rows(
            {"episodes": [{"accounts": 2, "events": events}]}
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(labels.tolist(), [1])
        self.assertEqual(episode_ids.tolist(), [0])
        self.assertAlmostEqual(rows[0][2], math.log1p(1) / 6)
        self.assertEqual(rows[0][16], 1)

    def test_oracle_additivity_and_unused_features(self):
        data = fixture()
        for case in data["cases"]:
            expected = case["expected"]
            self.assertAlmostEqual(
                expected["baseline"] + sum(expected["contributions"]),
                expected["rawMargin"],
            )
            self.assertTrue(
                all(abs(value) < 1e-14 for value in expected["contributions"][3:])
            )
        self.assertEqual(
            data["cases"][0]["expected"]["rawMargin"],
            data["cases"][1]["expected"]["rawMargin"],
        )
        self.assertNotEqual(
            data["cases"][0]["expected"]["contributions"],
            data["cases"][1]["expected"]["contributions"],
        )

    def test_saved_fixture_matches_independent_oracle(self):
        saved = json.loads(
            (Path(__file__).parent / "explanation-oracle.json").read_text()
        )
        self.assertEqual(saved, fixture())


if __name__ == "__main__":
    if "--write-fixture" in sys.argv:
        (Path(__file__).parent / "explanation-oracle.json").write_text(
            json.dumps(fixture(), indent=2) + "\n"
        )
    else:
        unittest.main()
