"""Independent stationary-distribution oracles and graph approximation contracts."""
import itertools
import math
from types import SimpleNamespace
import unittest

import numpy as np

from datasets.graph_features import DAMPING, GRAPH_FEATURE_SPECS, GraphFeatureState


def events(edges, kind="payment"):
    return [
        SimpleNamespace(source=source, destination=target, kind=kind)
        for source, target in edges
    ]


def stationary(state, source=None, directed=False):
    """Dense solve, independent of the production sparse/push implementation."""
    count = len(state.nodes)
    transition = np.zeros((count, count))
    adjacency = state.outgoing if directed else state.contacts
    for node, neighbors in enumerate(adjacency):
        if neighbors:
            for target in neighbors:
                transition[node, target] = 1.0 / len(neighbors)
        else:
            transition[node] = 1.0 / count
    restart = (
        np.full(count, 1.0 / count)
        if source is None
        else np.eye(count)[state.node_lookup[source]]
    )
    return np.linalg.solve(
        np.eye(count) - state.damping * transition.T, (1 - state.damping) * restart
    )


class GraphFeatureTests(unittest.TestCase):
    def test_cold_queries_and_nonpayment_events_never_admit_nodes(self):
        state = GraphFeatureState()
        self.assertTrue(all(value == 0 for value in state.values(10, 20).values()))
        state.observe_batch(
            events([(10, 20)], "deposit")
            + events([(30, 40)], "report")
            + events([(50, 50), (None, 60), (70, None)])
        )
        self.assertEqual(state.nodes, [])
        self.assertEqual(state.edge_count, 0)
        self.assertEqual(state.compute_ppr(10).edge_traversals, 0)
        self.assertEqual(state.pagerank_refreshes, 0)

    def test_directed_pagerank_matches_dangling_node_oracle_and_bound(self):
        state = GraphFeatureState()
        state.observe_batch(events([("a", "b")]))
        values = state.values("a", "b")
        expected = stationary(state, directed=True)
        actual = np.array(
            [values["graph_sender_pagerank"], values["graph_recipient_pagerank"]]
        )
        self.assertLessEqual(
            np.abs(actual - expected).sum(), values["graph_pagerank_error_bound"]
        )
        self.assertAlmostEqual(
            values["graph_sender_pagerank"], 1 / (2 + DAMPING), places=7
        )
        self.assertAlmostEqual(
            values["graph_recipient_pagerank"], (1 + DAMPING) / (2 + DAMPING), places=7
        )
        self.assertEqual(values["graph_pagerank_edge_lag"], 0)

    def test_pagerank_snapshots_show_lag_and_refresh_after_unique_edge_threshold(self):
        state = GraphFeatureState()
        state.observe_batch(events([(node, node + 1) for node in range(11)]))
        previous = state.values(0, 1)
        for _ in range(20):
            state.observe_batch(events([(0, 1)]))
        self.assertEqual(state.edge_count, 11)
        state.observe_batch(events([(11, 12)]))
        stale = state.values(0, 12)
        self.assertEqual(stale["graph_pagerank_edge_lag"], 1)
        self.assertEqual(
            stale["graph_sender_pagerank"], previous["graph_sender_pagerank"]
        )
        self.assertEqual(stale["graph_recipient_pagerank"], 0)
        self.assertEqual(state.pagerank_refreshes, 1)
        state.observe_batch(events([(12, 13)]))
        refreshed = state.values(0, 2)
        self.assertEqual(refreshed["graph_pagerank_edge_lag"], 0)
        self.assertEqual(state.pagerank_refreshes, 2)
        actual = np.array(
            [state.values(node, node)["graph_sender_pagerank"] for node in state.nodes]
        )
        self.assertLessEqual(
            np.abs(actual - stationary(state, directed=True)).sum(),
            refreshed["graph_pagerank_error_bound"],
        )

    def test_ppr_tiny_graphs_are_lower_bounds_with_valid_error_certificates(self):
        graphs = [
            [(0, 1)],
            [(0, 1), (1, 2)],
            [(0, 1), (1, 2), (2, 0), (2, 3)],
            [(0, 1), (0, 2), (0, 3), (3, 4), (4, 5), (5, 6)],
        ]
        for edges, budget in itertools.product(graphs, [1, 3, 16, 512]):
            state = GraphFeatureState(ppr_edge_budget=budget)
            state.observe_batch(events(edges))
            for source in state.nodes:
                expected = stationary(state, source)
                result = state.compute_ppr(source)
                self.assertLessEqual(result.edge_traversals, budget)
                self.assertLessEqual(sum(result.reserve.values()), 1 + 1e-14)
                for target in state.nodes:
                    score = result.estimate(target)
                    error = expected[state.node_lookup[target]] - score
                    self.assertGreaterEqual(error, -1e-14)
                    self.assertLessEqual(error, result.error_bound(target) + 1e-14)

    def test_two_node_ppr_converges_within_target_mass(self):
        state = GraphFeatureState()
        state.observe_batch(events([("s", "t")]))
        result = state.compute_ppr("s")
        self.assertLessEqual(result.remaining_mass, state.ppr_tolerance)
        self.assertAlmostEqual(
            result.estimate("s"), 1 / (1 + DAMPING), delta=result.error_bound("s")
        )
        self.assertAlmostEqual(
            result.estimate("t"), DAMPING / (1 + DAMPING), delta=result.error_bound("t")
        )

    def test_high_degree_partial_push_keeps_hub_and_unvisited_neighbor_signal(self):
        leaves = 2000
        state = GraphFeatureState(ppr_edge_budget=8)
        state.observe_batch(events([("hub", leaf) for leaf in range(leaves)]))
        leaf_result = state.compute_ppr(0)
        self.assertEqual(leaf_result.edge_traversals, 8)
        self.assertGreater(leaf_result.estimate("hub"), 0)
        exact_hub = DAMPING / (1 + DAMPING)
        self.assertLessEqual(leaf_result.estimate("hub"), exact_hub)
        self.assertLessEqual(
            exact_hub - leaf_result.estimate("hub"), leaf_result.error_bound("hub")
        )
        hub_result = state.compute_ppr("hub")
        exact_leaf = DAMPING / (leaves * (1 + DAMPING))
        for leaf in range(leaves):
            self.assertGreater(hub_result.estimate(leaf), 0)
            self.assertLessEqual(hub_result.estimate(leaf), exact_leaf)
            self.assertLessEqual(
                exact_leaf - hub_result.estimate(leaf), hub_result.error_bound(leaf)
            )
        self.assertGreater(hub_result.remaining_mass, state.ppr_tolerance)
        self.assertGreater(hub_result.dropped_mass, 0)
        # Cached results keep their historical meaning even if held after growth.
        state.observe_batch(events([("hub", "new leaf")]))
        self.assertEqual(hub_result.estimate("new leaf"), 0)

    def test_exact_components_degrees_and_impossible_paths(self):
        state = GraphFeatureState()
        state.observe_batch(events([(0, 1), (1, 2), (2, 0), (10, 11)]))
        values = state.values(0, 2)
        self.assertEqual(values["graph_same_component"], 1)
        self.assertEqual(values["graph_sender_component_size"], math.log1p(3))
        self.assertEqual(values["graph_sender_degree"], math.log1p(2))
        calculations = state.ppr_computations
        for target in (10, "new"):
            values = state.values(0, target)
            self.assertEqual(values["graph_same_component"], 0)
            self.assertEqual(values["graph_ppr_sender_to_recipient"], 0)
            self.assertEqual(values["graph_ppr_forward_error_bound"], 0)
        self.assertEqual(state.ppr_computations, calculations)

    def test_cache_is_bounded_and_tracks_only_contact_topology_changes(self):
        state = GraphFeatureState(ppr_cache_size=2)
        state.observe_batch(events([(0, 1), (1, 2)]))
        first = state.compute_ppr(0)
        self.assertIs(state.compute_ppr(0), first)
        state.observe_batch(events([(1, 0)]))  # new directed edge, same contacts
        self.assertIs(state.compute_ppr(0), first)
        state.compute_ppr(1)
        state.compute_ppr(2)
        self.assertEqual(len(state._ppr_cache), 2)
        self.assertIsNot(state.compute_ppr(0), first)
        state.observe_batch(events([(2, 3)]))
        self.assertEqual(len(state._ppr_cache), 0)

    def test_endpoint_admission_does_not_scan_existing_identity_registry(self):
        class NoRegistryScan(dict):
            def keys(self):
                raise AssertionError(
                    "Endpoint admission must not scan all historical identities."
                )

            def __iter__(self):
                raise AssertionError(
                    "Endpoint admission must not scan all historical identities."
                )

        state = GraphFeatureState()
        state.observe_batch(events([(node, node + 1) for node in range(100)]))
        state.node_lookup = NoRegistryScan(state.node_lookup)
        state.observe_batch(events([(100, 101)]))
        self.assertEqual(len(state.nodes), 102)

    def test_batch_permutation_and_query_order_do_not_change_values(self):
        edges = [
            (("customer", "1"), ("terminal", "1")),
            (("customer", "2"), ("terminal", "1")),
            (("customer", "3"), ("terminal", "2")),
            (("customer", "2"), ("terminal", "2")),
        ]
        states = [GraphFeatureState(ppr_edge_budget=3) for _ in range(2)]
        states[0].observe_batch(events(edges))
        states[1].observe_batch(events(list(reversed(edges))))
        pairs = [
            (source, target) for source in states[0].nodes for target in states[0].nodes
        ]
        expected = {pair: states[0].values(*pair) for pair in pairs}
        actual = {pair: states[1].values(*pair) for pair in reversed(pairs)}
        self.assertEqual(expected, actual)
        self.assertEqual(len(states[0].nodes), 5)

    def test_definitions_cover_every_value_and_explain_approximation(self):
        state = GraphFeatureState()
        self.assertEqual(
            {spec["id"] for spec in GRAPH_FEATURE_SPECS}, set(state.values(0, 1))
        )
        for spec in GRAPH_FEATURE_SPECS:
            for field in (
                "method",
                "parameters",
                "approximation",
                "orientation",
                "readout",
                "unit",
                "transform",
                "window",
                "requirements",
                "version",
            ):
                self.assertIn(field, spec)
            self.assertEqual(spec["requirements"], ["identities", "timestamps"])


if __name__ == "__main__":
    unittest.main()
