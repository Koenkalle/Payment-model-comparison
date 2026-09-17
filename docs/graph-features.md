# Graph features in Data lab

Use **Dataset features → Graph features** to inspect and select graph inputs.
**Add graph features** adds available graph columns to the current selection;
save a feature dataset to make them available to every trainable model. Each
definition exposes its method, parameters, graph orientation, approximation,
and full-population statistics. Individual graph columns and their diagnostic
columns can be disabled independently.

The recipe library is now `payment-features/v2`. Existing saved feature datasets
keep their original columns and recipe catalog. Follow **Original source** to
inspect the new recipes and save another version. Sources without stable entity
identities, including ULB, display an explanation instead of fabricated graph
features.

## Available inputs

| Inputs | Meaning |
| --- | --- |
| `graph_sender_pagerank`, `graph_recipient_pagerank` | Approximate global PageRank on the directed graph of earlier unique payment relationships |
| `graph_ppr_sender_to_recipient`, `graph_ppr_recipient_to_sender` | Lower bounds for personalized PageRank between the endpoints on the undirected contact graph |
| `graph_ppr_forward_error_bound`, `graph_ppr_reverse_error_bound` | Absolute probability error bounds for the corresponding PPR estimates |
| `graph_pagerank_error_bound` | Numerical L1 error bound for the most recently calculated PageRank snapshot |
| `graph_pagerank_edge_lag` | New directed relationships observed since that PageRank snapshot |
| `graph_same_component` | Whether both endpoints already belong to the same weakly connected component |
| `graph_sender_component_size`, `graph_recipient_component_size` | `log1p` of the number of observed entities in each endpoint's component |
| `graph_sender_degree`, `graph_recipient_degree` | Each endpoint's distinct contacts, with the transformation shown in its definition |

PageRank scores are probability mass, not percentiles. PPR is personalized to an
endpoint, so the two directions need not agree even on an undirected graph.
An endpoint absent from the historical graph has zero PageRank, degree, and
component size; pair PPR is also zero. Snapshot-wide diagnostics can still be
nonzero. This represents absent history and does not admit a new node to earlier
calculations.

## History and graph semantics

All rows at a timestamp see exactly the same preceding graph. Only after those
rows have been scored are their relationships inserted. Future transactions,
future account roster entries, outcomes, settlement decisions, amounts,
deposits, and reports do not influence these graph features. The topology uses
distinct relationships; repeated payments retain their existing count and value
features without multiplying graph edges. Self-payments do not create graph
relationships.

Prepared stream selections retain earlier observable source context. Entity
types remain part of identity, so customer `1` and terminal `1` are different
nodes. Graph features are stored by transaction ID alongside the other dataset
columns. They reach numeric models directly, and graph models through edge and
current-transaction inputs. They do not alter labels, splits, or replay events.

Global PageRank follows payment direction. PPR and component features use the
undirected contact graph. This matters for customer–terminal sources: a purely
directed customer→terminal graph cannot represent longer paths, whereas contact
PPR can follow customer→terminal→customer→terminal relationships. It does so
without constructing a potentially dense customer-to-customer projection.

## Cheap approximations and their limits

Global PageRank uses sparse power iteration with damping `0.85`, uniform
teleportation, and uniform redistribution from dangling nodes. These are the
conventional PageRank choices documented by
[NetworkX](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html).
The implementation uses NumPy directly and does not add a NetworkX dependency.

Recomputing the entire graph for every transaction would be expensive. Instead,
the feature provider periodically refreshes a snapshot when enough new directed
relationships have arrived, with a bounded iteration count and warm starts.
The parameters are visible in Data lab. The iteration residual gives a
numerical L1 error bound for that snapshot; `graph_pagerank_edge_lag` separately
describes topology that arrived since it. The numerical bound does **not** bound
the error caused by a stale snapshot. Newly observed nodes absent from the
snapshot receive zero PageRank until a refresh includes them.

PPR uses a local reserve/residual push, following the invariant described by
[Zhang, Lofgren, and Goel](https://www.kdd.org/kdd2016/papers/files/rfp1146-zhangA.pdf).
At each step, the restart fraction of a node's residual becomes reserved score,
and the continuation fraction moves toward its neighbors. Work is capped by
edge traversals. Partial expansions retain the original degree denominator;
they never renormalize a conveniently chosen subset of neighbors. Unexpanded
probability mass supplies an explicit error bound, including when a large hub
uses the budget. The cap and stopping threshold therefore never masquerade as a
guaranteed requested precision.

The capped partial-expansion scheme and its bound are an adaptation of the
published invariant. Small scores with large error bounds are unresolved
estimates, not proof that two nodes are unrelated. Estimates are deterministic,
and a bounded cache reuses a source's PPR result while contact topology remains
unchanged. Components use an incremental union-find structure.

[Bahmani, Chowdhury, and Goel](https://arxiv.org/abs/1006.2880) also demonstrate
incremental PageRank through sampled walks. That is a useful alternative for
future work, but introduces sampling error and maintained walk state. Here,
sparse snapshots and capped local pushes give explicit, inspectable behavior
without recomputing full personalized vectors for every transaction.

## Extending and validating the methods

The state provider and definitions live in `datasets/graph_features.py`.
`datasets/features.py` advances it between timestamp groups and registers its
outputs alongside existing feature recipes. Add a graph definition and its
calculation there; frontend controls and statistics are generated from metadata.
Bump the recipe version when calculation semantics or approximation budgets
change. Saved datasets retain their previous values and catalog.

Validation includes independently solved tiny PageRank/PPR systems, work-cap
and hub cases, strict timestamp and future-data isolation, bipartite three-hop
relationships, source capability checks, and persisted numeric/graph alignment.
Approximation bounds describe computation accuracy; they do not promise improved
fraud detection on a particular dataset. Compare held-out model runs to measure
whether a selected feature helps.
