"""Dataset exploration preserves source semantics while bounding every response."""
from contextlib import closing
import http.client
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from framework.comparison_service import make_server
from framework.dataset_graph import DatasetGraphService, node_id
from framework.pipeline_data import DatasetStore
from framework.pipeline_service import PipelineService


HANDBOOK = (
    "TRANSACTION_ID,TX_TIME_SECONDS,CUSTOMER_ID,TERMINAL_ID,TX_AMOUNT,TX_FRAUD,TX_FRAUD_SCENARIO\n"
    "late,120,1,1,12,1,99\nearly,60,1,2,10,0,0\nunknown,120,2,2,8,,\n"
)
PAYSIM_HEADER = (
    "step,type,amount,nameOrig,oldbalanceOrg,nameDest,oldbalanceDest,isFraud\n"
)


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.store = DatasetStore(self.base / "store")
        self.graph = DatasetGraphService(self.store)

    def tearDown(self):
        self.temporary.cleanup()

    def handbook(self, csv=HANDBOOK):
        return self.store.import_source({"source": "handbook", "csv": csv})["id"]

    def paysim(self, rows):
        return self.store.import_source(
            {"source": "paysim", "csv": PAYSIM_HEADER + rows}
        )["id"]

    def all_pages(self, identifier, **options):
        edges, offset = [], 0
        for _ in range(100):
            result = self.graph.query(identifier, {**options, "offset": offset})
            self.assertLessEqual(len(result["nodes"]), options.get("node_limit", 100))
            self.assertLessEqual(len(result["edges"]), options.get("edge_limit", 300))
            node_ids = {node["id"] for node in result["nodes"]}
            for edge in result["edges"]:
                self.assertIn(edge["source"], node_ids)
                self.assertIn(edge["target"], node_ids)
            edges.extend(result["edges"])
            if not result["page"]["has_more"]:
                return edges
            self.assertGreater(result["page"]["next_offset"], offset)
            offset = result["page"]["next_offset"]
        self.fail("Graph pagination did not finish.")

    def test_typed_topology_original_units_and_separate_truth_without_conversion(self):
        identifier = self.handbook()
        summary = self.graph.summary(identifier)
        self.assertEqual(summary["counts"], {"nodes": 4, "edges": 3})
        self.assertEqual(
            summary["node_types"],
            [{"type": "customer", "count": 2}, {"type": "terminal", "count": 2}],
        )
        self.assertEqual(
            summary["time_range"], {"start": 60.0, "end": 120.0, "unit": "seconds"}
        )
        result = self.graph.query(identifier, {})
        self.assertEqual(
            [edge["id"] for edge in result["edges"]], ["early", "late", "unknown"]
        )
        self.assertEqual([edge["label"] for edge in result["edges"]], [0, 1, -1])
        self.assertEqual(
            [edge["properties"]["features"]["amount"] for edge in result["edges"]],
            [10, 12, 8],
        )
        self.assertNotIn("TX_FRAUD_SCENARIO", json.dumps(result))
        self.assertTrue(
            all(
                set(edge["properties"]["features"]) == {"amount"}
                for edge in result["edges"]
            )
        )
        self.assertNotIn(str(self.base), json.dumps([summary, result]))
        customer = next(
            node for node in result["nodes"] if node["id"] == node_id("customer", "1")
        )
        terminal = next(
            node for node in result["nodes"] if node["id"] == node_id("terminal", "1")
        )
        self.assertNotEqual(customer["id"], terminal["id"])
        self.assertEqual(
            (customer["in_degree"], customer["out_degree"], customer["degree"]),
            (0, 2, 2),
        )
        self.assertEqual((terminal["in_degree"], terminal["out_degree"]), (1, 0))
        self.assertFalse(result["page"]["has_more"])

    def test_deterministic_pages_with_dense_hubs_parallel_edges_and_self_loops(self):
        rows = ["1,TRANSFER,1,hub,100,hub,100,0", "1,TRANSFER,2,hub,100,dest0,100,1"]
        rows += [
            f"{index + 2},PAYMENT,3,hub,100,dest{index},100,0" for index in range(80)
        ]
        rows += ["83,TRANSFER,4,origin,100,hub,100,1"]
        identifier = self.paysim("\n".join(rows) + "\n")
        all_edges = self.all_pages(identifier, node_limit=3, edge_limit=4)
        self.assertEqual(len(all_edges), len(rows))
        self.assertEqual(len({edge["id"] for edge in all_edges}), len(rows))
        self.assertEqual(
            all_edges, self.all_pages(identifier, node_limit=3, edge_limit=4)
        )
        hub = node_id("account", "hub")
        outgoing = self.all_pages(
            identifier,
            mode="neighbors",
            node_id=hub,
            direction="outgoing",
            node_limit=3,
            edge_limit=4,
        )
        incoming = self.all_pages(
            identifier,
            mode="neighbors",
            node_id=hub,
            direction="incoming",
            node_limit=3,
            edge_limit=4,
        )
        both = self.all_pages(
            identifier, mode="neighbors", node_id=hub, node_limit=3, edge_limit=4
        )
        self.assertEqual(len(outgoing), 82)
        self.assertEqual(len(incoming), 2)
        self.assertEqual(
            len(both), 83
        )  # Loop appears once, although both degree directions include it.
        self.assertTrue(all(edge["source"] == hub for edge in outgoing))
        self.assertTrue(all(edge["target"] == hub for edge in incoming))
        found = self.graph.search(identifier, "hub")["nodes"][0]
        self.assertEqual(
            (found["in_degree"], found["out_degree"], found["degree"]), (2, 82, 84)
        )

    def test_time_truth_and_type_filters_keep_half_open_source_clock(self):
        identifier = self.handbook()
        result = self.graph.query(identifier, {"start": 60, "stop": 120})
        self.assertEqual([edge["id"] for edge in result["edges"]], ["early"])
        self.assertEqual(
            [
                edge["id"]
                for edge in self.graph.query(identifier, {"start": 120, "label": -1})[
                    "edges"
                ]
            ],
            ["unknown"],
        )
        self.assertEqual(
            [
                edge["id"]
                for edge in self.graph.query(identifier, {"label": 1})["edges"]
            ],
            ["late"],
        )
        self.assertEqual(
            self.graph.query(identifier, {"start": 120, "stop": 120})["edges"], []
        )
        identifier = self.paysim("1,TRANSFER,1,A,10,B,10,1\n2,PAYMENT,1,B,10,A,10,0\n")
        filtered = self.graph.query(identifier, {"edge_type": "TRANSFER"})
        self.assertEqual([edge["type"] for edge in filtered["edges"]], ["TRANSFER"])
        self.assertEqual(
            filtered["nodes"][0]["degree"], 2
        )  # Dataset degree is deliberately not filtered.

    def test_existing_nodes_reserve_budget_and_saturated_expansion_is_explicit(self):
        identifier = self.handbook()
        existing = [node_id("customer", "2"), node_id("terminal", "2")]
        result = self.graph.query(
            identifier, {"existing_node_ids": existing, "node_limit": 2}
        )
        self.assertEqual({node["id"] for node in result["nodes"]}, set(existing))
        self.assertEqual(result["edges"], [])
        self.assertTrue(result["truncated"])
        self.assertEqual(result["limit_reason"], "node_limit")
        self.assertIsNone(result["page"]["next_offset"])
        result = self.graph.query(
            identifier,
            {"mode": "neighbors", "node_id": existing[1], "direction": "outgoing"},
        )
        self.assertEqual([node["id"] for node in result["nodes"]], [existing[1]])
        self.assertEqual(result["edges"], [])
        with self.assertRaisesRegex(ValueError, "exceed"):
            self.graph.query(
                identifier,
                {
                    "mode": "neighbors",
                    "node_id": node_id("customer", "1"),
                    "existing_node_ids": existing,
                    "node_limit": 2,
                },
            )

    def test_literal_prefix_search_preserves_identity_and_is_bounded(self):
        identifier = self.paysim(
            "1,TRANSFER,1,A%_x,10,Z,10,0\n2,TRANSFER,1,a%_y,10,Z,10,0\n"
            "3,TRANSFER,1,another,10,Z,10,0\n"
        )
        result = self.graph.search(identifier, "A%_", limit=1)
        self.assertEqual(len(result["nodes"]), 1)
        self.assertTrue(result["has_more"])
        self.assertEqual(len(self.graph.search(identifier, "A%_")["nodes"]), 2)
        self.assertEqual(self.graph.search(identifier, "%")["nodes"], [])
        identity = node_id("account", "A%_x")
        self.assertEqual(
            self.graph.search(identifier, identity)["nodes"][0]["id"], identity
        )
        self.assertNotEqual(node_id("a:b", "c"), node_id("a", "b:c"))
        self.assertNotEqual(node_id("customer", "01"), node_id("customer", "1"))

    def test_payment_accounts_isolates_and_report_exclusion(self):
        document = {
            "name": "With isolates",
            "accounts": [
                {
                    "id": 0,
                    "external_id": "a",
                    "name": "<script>alert(1)</script>",
                    "path": "/private/source",
                },
                {"id": 1, "external_id": "b"},
                {"id": 2, "external_id": "alone"},
            ],
            "events": [
                {
                    "id": "deposit",
                    "kind": "deposit",
                    "t": 0,
                    "u": -1,
                    "v": 0,
                    "amount": 20,
                },
                {"id": "pay", "kind": "payment", "t": 1, "u": 0, "v": 1, "amount": 10},
                {
                    "id": "report",
                    "kind": "report",
                    "t": 2,
                    "u": -1,
                    "v": 1,
                    "amount": 10,
                },
            ],
            "truth": {"pay": True},
            "label_available_at": {"pay": 2},
        }
        (self.base / "payment.json").write_text(json.dumps(document))
        (self.base / "payment.config.json").write_text(
            json.dumps({"loader": "payment_json", "path": "payment.json"})
        )
        store = DatasetStore(
            self.base / "payments", [self.base / "payment.config.json"]
        )
        graph = DatasetGraphService(store)
        identifier = next(
            row["id"]
            for row in store.catalog()["datasets"]
            if row["name"] == "With isolates"
        )
        self.assertEqual(graph.summary(identifier)["counts"], {"nodes": 3, "edges": 1})
        edge = graph.query(identifier, {})["edges"][0]
        self.assertEqual((edge["id"], edge["time"], edge["label"]), ("pay", 60, 1))
        self.assertEqual(edge["properties"]["outcome"]["available_at"], 120)
        self.assertEqual(edge["properties"]["currency"], "EUR")
        self.assertNotIn("/private/source", json.dumps(graph.query(identifier, {})))
        alone = graph.search(identifier, "alone")["nodes"][0]
        self.assertEqual(alone["degree"], 0)
        result = graph.query(identifier, {"mode": "neighbors", "node_id": alone["id"]})
        self.assertEqual(result["nodes"], [alone])
        self.assertEqual(result["edges"], [])

    def test_numeric_only_dataset_explicitly_unsupported(self):
        columns = ["Time", "Amount", *[f"V{index}" for index in range(1, 29)], "Class"]
        csv = ",".join(columns) + "\n" + ",".join(["0", "20", *[".2"] * 28, "1"]) + "\n"
        identifier = self.store.import_source({"source": "ulb", "csv": csv})["id"]
        self.assertFalse(self.graph.summary(identifier)["supported"])
        with self.assertRaisesRegex(ValueError, "no entity relationships"):
            self.graph.query(identifier, {})

    def test_canonical_descriptor_roles_support_new_entity_kinds_without_model_changes(
        self,
    ):
        original = self.handbook()
        source = self.base / "new-adapter.sqlite"
        source.write_bytes(Path(self.store._config(original)["path"]).read_bytes())
        with closing(sqlite3.connect(source)) as database:
            descriptor = json.loads(
                database.execute(
                    'SELECT value FROM metadata WHERE key="descriptor"'
                ).fetchone()[0]
            )
            descriptor["graph"] = {"source_role": "payer", "destination_role": "payee"}
            descriptor["entity_types"].append("processor")
            database.execute(
                'UPDATE metadata SET value=? WHERE key="descriptor"',
                (json.dumps(descriptor),),
            )
            for event_id, raw in database.execute(
                "SELECT event_id,entities FROM events"
            ).fetchall():
                entities = json.loads(raw)
                for entity in entities:
                    entity["role"] = {"source": "payer", "destination": "payee"}[
                        entity["role"]
                    ]
                entities.append(
                    {"kind": "processor", "id": "shared", "role": "operator"}
                )
                database.execute(
                    "UPDATE events SET entities=? WHERE event_id=?",
                    (json.dumps(entities), event_id),
                )
            database.commit()
        config = self.base / "new-adapter.config.json"
        config.write_text(
            json.dumps(
                {
                    "loader": "prepared_fraud",
                    "path": source.name,
                    "name": "Extended schema",
                    "selection": {"start": 120, "stop": 180},
                }
            )
        )
        store = DatasetStore(self.base / "extended", [config])
        graph = DatasetGraphService(store)
        identifier = next(
            row["id"]
            for row in store.catalog()["datasets"]
            if row["name"] == "Extended schema"
        )
        self.assertEqual(graph.summary(identifier)["counts"], {"nodes": 5, "edges": 2})
        self.assertEqual(graph.summary(identifier)["time_range"]["start"], 120)
        self.assertEqual(
            graph.search(identifier, "shared")["nodes"][0]["type"], "processor"
        )
        self.assertEqual(graph.search(identifier, "shared")["nodes"][0]["degree"], 0)
        self.assertEqual(
            graph.query(identifier, {})["edges"][0]["source"], node_id("customer", "1")
        )

    def test_index_is_reused_after_restart_and_payload_tampering_is_rejected(self):
        identifier = self.handbook()
        expected = self.graph.summary(identifier)
        with patch(
            "framework.dataset_graph.PreparedStreamProvider.populate",
            side_effect=AssertionError("rebuilt"),
        ):
            self.assertEqual(self.graph.summary(identifier), expected)
            self.assertEqual(
                DatasetGraphService(self.store).summary(identifier), expected
            )
            self.assertEqual(len(self.graph.query(identifier, {})["edges"]), 3)
        source = Path(self.store._config(identifier)["path"])
        source.write_bytes(source.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "payload checksum"):
            self.graph.query(identifier, {})

    def test_index_can_rebuild_corruption_and_prevents_path_traversal(self):
        identifier = self.handbook()
        expected = self.graph.summary(identifier)
        index = self.graph.directory / (identifier + ".sqlite")
        index.write_bytes(b"corrupt cache")
        self.assertEqual(self.graph.summary(identifier), expected)
        self.assertFalse(list(self.graph.directory.glob(".building-*")))
        for invalid in ("../source", "ds-" + "a" * 31, "/etc/passwd"):
            with self.assertRaises(ValueError):
                self.graph.summary(invalid)

    def test_invalid_and_nonfinite_payloads_fail_before_query(self):
        identifier = self.handbook()
        invalid = [
            {"node_limit": 501},
            {"edge_limit": 2001},
            {"node_limit": True},
            {"edge_limit": 0},
            {"offset": -1},
            {"offset": 1.5},
            {"mode": "unknown"},
            {"direction": "reverse"},
            {"start": float("nan")},
            {"stop": float("inf")},
            {"start": True},
            {"start": 10**400},
            {"start": 2, "stop": 1},
            {"label": True},
            {"label": 2},
            {"edge_type": []},
            {"mode": "neighbors"},
            {"node_id": "foo"},
            {"existing_node_ids": "foo"},
            {"existing_node_ids": ["a", "a"]},
            {"existing_node_ids": [[]]},
            {"path": "/etc/passwd"},
        ]
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.graph.query(identifier, payload)
        for payload in (
            {"mode": "neighbors", "node_id": "missing"},
            {"existing_node_ids": ["missing"]},
        ):
            with self.assertRaisesRegex(ValueError, "Unknown graph node"):
                self.graph.query(identifier, payload)
        for limit in (0, 51, True, 2.5):
            with self.assertRaises(ValueError):
                self.graph.search(identifier, "", limit)


class GraphHTTPTests(unittest.TestCase):
    def test_public_graph_endpoints_and_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            pipeline = PipelineService(Path(temporary) / "saved")
            self.addCleanup(pipeline.close)
            server = make_server(
                SimpleNamespace(models=lambda: {"models": []}),
                port=0,
                pipeline_service=pipeline,
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                identifier = pipeline.store.catalog()["datasets"][0]["id"]
                base = "/api/pipeline/datasets/" + identifier + "/graph"

                def request(path, body=None):
                    connection = http.client.HTTPConnection(
                        "127.0.0.1", server.server_port, timeout=30
                    )
                    connection.request(
                        "POST" if body is not None else "GET",
                        path,
                        body=json.dumps(body) if body is not None else None,
                        headers={"Content-Type": "application/json"},
                    )
                    response = connection.getresponse()
                    result = response.status, json.loads(response.read())
                    connection.close()
                    return result

                self.assertEqual(request(base)[0], 200)
                status, result = request(
                    base + "/query", {"node_limit": 3, "edge_limit": 2}
                )
                self.assertEqual(status, 200, result)
                self.assertLessEqual(len(result["nodes"]), 3)
                self.assertEqual(request(base + "/search?q=0&limit=2")[0], 200)
                for invalid in (
                    "?q=0&q=1",
                    "?limit=NaN",
                    "?limit=0",
                    "?path=/private",
                    "?limit=1.5",
                ):
                    self.assertEqual(request(base + "/search" + invalid)[0], 400)
                self.assertEqual(
                    request(base + "/query", {"start": float("nan")})[0], 400
                )
                self.assertEqual(
                    request(base + "/query", {"node_limit": 999999})[0], 400
                )
                self.assertEqual(request(base + "/unknown", {})[0], 404)
                self.assertEqual(
                    request("/api/pipeline/datasets/not-an-id/graph/query", {})[0], 404
                )
                self.assertEqual(
                    request("/api/pipeline/datasets/..%2Fprivate/graph/query", {})[0],
                    404,
                )
                self.assertNotIn(temporary, json.dumps(result))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
                pipeline.close()


if __name__ == "__main__":
    unittest.main()
