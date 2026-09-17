# Dataset graph explorer

The Data lab can inspect stored datasets as directed graphs when their source
provides entity identities and relationships. This works for generated customer
and terminal payments, payment scenarios, and graph-capable imported datasets.
Numeric-only data such as ULB has no account identities; the graph action explains
why that dataset cannot be drawn.

Start the workspace from the repository root:

```sh
python build.py
python serve.py
```

Open `http://127.0.0.1:8000/data-lab.html`, generate or select a dataset, then open
its graph from the dataset inspector. Graph records load when the explorer opens.
The stored dataset remains available to the training and comparison tools.

## Explore a small part of a large dataset

The summary reports the whole dataset's entity and relationship counts, entity
types, relationship types, and time range. The visible graph is a bounded subset;
the explorer reports its counts separately. An edge represents a source event,
so several events between the same entities remain separate relationships.

1. Set the node and edge limits, optionally select a time interval, outcome or
   relationship type, then apply the controls to load a sample.
2. Select a node on the canvas or in the node table to inspect its identity,
   source properties, and incoming/outgoing event counts. Select a relationship
   on the canvas or in the edge table to inspect its event time, outcome, and
   source features.
3. Expand a selected node to add its immediate neighbors within the current
   budgets. Focus on a node to replace the viewport with its neighborhood. Choose
   incoming, outgoing, or both directions to follow the relationships of interest.
4. Search for an entity by name or identity to reach a node outside the displayed
   subset. Select a search result to inspect it, then choose **Focus here** to
   open its neighborhood.
5. Request the next sample to advance through source events. Each page replaces
   the viewport. Undo returns to a previous exploration step; reset restores the
   default limits and filters and starts a fresh sample.

Applying filters always starts a fresh sample. **Expand direction** controls
the subsequent expansion or focus operation. Time bounds are inclusive at the
start and exclusive at the end.

Drag the background to pan, zoom to inspect detail, and select a layout to make
the currently loaded structure easier to read. The node and edge tables provide
another way to select graph elements. Export downloads the displayed graph and
its exploration context as JSON.

During loading, **Cancel loading** preserves the previous view. Closing the
explorer also cancels its active request. With the canvas focused, use `+` and `-`
to zoom, arrow keys to pan, `F` to fit the graph, and `Escape` to clear selection.

Outcome labels describe the source data's known fraud, legitimate, or unknown
outcomes. They are not model predictions. A subset's shape or label proportions
need not represent the whole dataset; sampling follows event order rather than a
statistical sampling procedure. A time or outcome filter can correctly produce an
empty view. Source times and amount units are shown with the graph's properties.

## Bounded exploration

The server enforces a maximum of **500 nodes and 2,000 edges per response**. The
browser also enforces its chosen viewport budgets when merging neighborhoods, so
repeated expansion cannot grow the drawing without limit. Every displayed edge
includes both endpoints. Paging returns a continuation cursor and explicitly
indicates when more matching events remain; it does not silently present a
partial graph as the complete dataset.

History keeps the last 12 viewport snapshots. Search matches a name or external
ID prefix, or an exact graph ID, and returns up to 20 results from the stored
graph, including nodes outside the viewport.
Changing datasets invalidates previous graph work, preventing a late response
from displaying records from the previous selection.

The first graph request builds a disposable, versioned SQLite index under the
workspace's `graph-index` directory. Canonical dataset streams are indexed in
batches; payment scenario snapshots already have bounded source documents.
Subsequent requests use the index's event cursor and entity adjacency indexes
instead of materializing the whole graph. The index survives server restarts.
The source snapshot is checksum-verified when first used and after its file
signature changes; index construction completes through an atomic replacement.

Initial index preparation may take longer than a subsequent query for a large
stored dataset. Rendering and browser payloads remain bounded by the selected
limits. Raising the limits makes dense graphs more expensive to lay out and
harder to inspect; focusing on one entity or narrowing the interval usually gives
a clearer view.

## Implementation and extension points

The explorer separates these components:

| Component | Responsibility |
| --- | --- |
| `framework/dataset_graph.py` | Source providers, graph indexing, summaries, bounded queries, and search |
| `framework/pipeline_service.py` | Dataset-scoped graph HTTP endpoints |
| `shared/graph/explorer.js` | Reusable exploration controls, selection, request lifecycle, history, and export |
| `shared/graph/view.js` | Explorer markup, responsive styles, and control labels |
| `shared/graph/model.js` | Pure graph merging with budgets and visible structure calculations |
| `shared/graph/renderer.js` | Reusable graph drawing and pointer interaction |
| `tools/data-lab/ui.js` | Selected dataset and graph availability in the Data lab |
| `tools/registry.json` | Script registration in the generated Data lab page |

The versioned `dataset-graph/v1` payload uses opaque node IDs. Nodes expose
`id`, `label`, `type`, `properties`, and degree counts. Edges expose `id`, `source`,
`target`, `type`, `time`, `label`, and `properties`. The renderer works with this
graph contract and does not interpret dataset loaders, generators, or models.
Keep source-specific names, units, and feature values in provider properties.
Degree counts describe the full dataset, independently of viewport filters.
Amounts retain source values; a currency of `null` means that the adapter does
not declare a currency. Relationship outcomes include their source availability
time when provided.

The endpoint prefix is `/api/pipeline/datasets/{dataset_id}/graph`:

| Request | Result |
| --- | --- |
| `GET` the prefix | Capability, whole-dataset counts, types, time bounds, and enforced limits |
| `POST /query` | Bounded node and event records plus a continuation cursor |
| `GET /search?q=...&limit=20` | Matching entities and an indication of further matches |

Queries use `mode: "sample"` or `mode: "neighbors"` with an opaque `node_id`.
Specify `node_limit`, `edge_limit`, and `offset`; copy the returned `next_offset`
when continuing. Optional filters are `direction`, `start`, `stop`, `label`, and
`edge_type`. Neighborhood expansion supplies `existing_node_ids`, which count
toward the same node budget. A `null` continuation cursor means there is no
advancing next page with that query and budget; a full node budget may require
focusing on one entity or increasing the limit.

The prepared-stream provider uses the dataset descriptor's declared source and
destination roles. A new adapter that emits those roles can use the existing
graph projection. Payment scenarios use the payment-document provider. To
support another storage format, implement a provider and register its loader in
the provider mapping. Preserve stable event identities and endpoint identities;
two entity types may legitimately use the same external ID.

To expose the graph in another tool, reuse the provider endpoints and renderer,
or mount the explorer controller against another panel. The page's selected
dataset wiring can be adapted separately.
Adding a model does not require a graph UI change because the explorer inspects
the dataset independently of training. Edit source templates and scripts, then
run `python build.py` to regenerate `data-lab.html`.

The incremental expansion, directional exploration, history, and provider
separation were informed by
[VibePaWiz](https://github.com/Koenkalle/VibePaWiz), which explores citation
networks with interchangeable data providers. The entity/relationship inspector
also follows the familiar interaction pattern of graph database browsers.

## Verification

Build the pages and run the provider, graph model, renderer, and browser checks.
The renderer and browser tests require Playwright with its Chromium browser:

```sh
python build.py
python tests/pipeline/test_graph.py
node tests/pipeline/test_graph_model.js
node tests/pipeline/test_graph_renderer.js
node tests/pipeline/test_graph_browser.js
```

Set `PLAYWRIGHT_MODULE` to a local Playwright package path if it is installed
outside this repository, and `PYTHON_BINARY` if the server needs a particular
Python environment. The integration test creates a temporary workspace and exercises stored
generated datasets, bounded exploration, canvas painting and selection, search,
filters, cancellation, failed-request recovery, export, mobile layout, provider
switching, and numeric-only availability.
