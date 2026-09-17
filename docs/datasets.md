# Fraud datasets and chronological replay

The data module imports several source schemas through one ordered transaction
stream. Choose a view to use the same source with the existing numeric
experiments, temporal models, or browser comparison. Source preparation and
iteration use the Python standard library; fitted models keep their existing
NumPy, scikit-learn, and optional PyTorch requirements.

The bundled [Handbook-shaped CSV](../examples/datasets/fixtures/handbook.csv) is
an invented, hand-authored smoke fixture: 48 rows, 24 distinct timestamps, and
both classes at every timestamp. It is not an extract from the Handbook, a
realistic fraud simulation, or a benchmark. Its balanced labels and easily
separable amounts make it useful for checking an integration, not assessing
model quality. No external dataset is bundled or downloaded automatically.

## Run the included example

Run these commands from the repository root, using an environment with the
project's model dependencies installed:

```sh
python experiment.py list
python experiment.py inspect-data --config examples/datasets/handbook.config.json
python experiment.py train --config examples/handbook-experiment.json --output artifacts/handbook-logistic
python experiment.py evaluate --config examples/datasets/handbook.config.json --artifact artifacts/handbook-logistic --partition test --output artifacts/handbook-logistic-test.json
```

Paths inside a configuration are relative to that configuration's directory.
Outputs must not already exist. The logistic example fits preprocessing and
model parameters on the training prefix, chooses a threshold on validation,
and reports the chronological test suffix. Equal timestamps stay together.

With `requirements-temporal.txt` installed, the small CPU example exercises the
existing supervised temporal fraud task:

```sh
python experiment.py train --config examples/handbook-fraud-training.json --output artifacts/handbook-fraud
```

It uses four chronological partitions, random encoder initialization, and two
training epochs for a quick integration run. Increase data and training settings
for an actual experiment. This training path uses the repository's existing
temporal training protocol; the independent replay API below controls its own
history and feedback protocol.

## Select a source and view

The `fraud_dataset` loader chooses a source adapter with `dataset`, then a
representation with `view`:

```json
{
  "loader": "fraud_dataset",
  "dataset": "handbook",
  "path": "fixtures/handbook.csv",
  "view": "numeric"
}
```

| View | Existing contract or API | Suitable use |
| --- | --- | --- |
| `stream` | `TransactionStream` | Ordered batches, replay, preparation |
| `numeric` | `numeric-table/v1` | Logistic regression and XGBoost experiments |
| `graph` | `temporal-graph/v1` | Existing temporal link-prediction models |
| `labeled_graph` | Supervised temporal graph | Existing `temporal-fraud-classification` task |
| `payments` | `payment-events/v1` | JSON import in the browser comparison |

Numeric and graph features contain only adapter-declared observable inputs.
Use `features` to choose a subset by name; `log1p_amount` is also available.
Labels, fraud scenario identifiers, and post-event balances cannot be selected
as model features. Entity IDs identify graph nodes rather than becoming numeric
feature values. Graph identity namespaces include the dataset, release version,
and entity type, so a customer and terminal with the same ID remain different
nodes. Set `release` consistently across partitions from the same release, and
use different releases for unrelated simulations. Source content fingerprints
remain in provenance: editing an outcome must not rename an observed entity.
The `payments` and `labeled_graph` views use the existing fixed amount feature
contract and reject configurable `features`.

Data lab additionally supports saved **feature datasets** through the
`dataset_features` loader. These materialize selected source and derived columns
once, reuse them as numeric inputs or graph edge attributes, and preserve the
original payment document for replay. This is a separate wrapper around the
source views described here. See [shared features in Data lab](pipeline.md#choose-and-inspect-shared-features)
for selection, statistics, causal history, and the save API.

Use a half-open time selection to keep a materialized view manageable:

```json
{
  "loader": "fraud_dataset",
  "dataset": "handbook",
  "path": "fixtures/handbook.csv",
  "view": "numeric",
  "features": ["log1p_amount"],
  "selection": {"start": 0, "stop": 43200}
}
```

Selection bounds use the adapter's relative seconds. A stream uses
`iter_events(start=..., stop=...)` instead of configuration selection.
Materialized temporal graphs rebase their clock to the first selected event and
record that origin in provenance; explicit model split boundaries use that
graph clock. Numeric views and direct replay preserve source-relative seconds.
Training views materialize their selected arrays. Preparing and iterating a
stream do not require keeping every transaction in Python memory. These are
distinct scaling properties: choosing `stream` does not make a fitted model or
a growing historical graph bounded in memory. Payment JSON preserves source-relative
minutes across selections, so a selected interval need not start at zero and
historical overlap checks continue to identify the same transactions.

### Supported source schemas

| Adapter | Required source shape | Observable inputs and relationships |
| --- | --- | --- |
| [`handbook`](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/SimulatedDataset.html) | `TRANSACTION_ID`, `TX_TIME_SECONDS`, `CUSTOMER_ID`, `TERMINAL_ID`, `TX_AMOUNT`; optional `TX_FRAUD` | Amount; customer-to-terminal relationships; relative seconds |
| [`ulb`](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | `Time`, `Amount`, `V1`–`V28`; optional `Class` | Amount and anonymized numeric features; no stable relational IDs |
| [`paysim`](https://www.kaggle.com/datasets/ealaxi/paysim1) | `step`, `type`, `nameOrig`, `nameDest`, `amount`, `oldbalanceOrg`, `oldbalanceDest`; optional `isFraud` | Amount and pre-event balances as numeric features; transaction type on events; directed account relationships |

Use either `path` for one CSV or a directory of alphabetically ordered `*.csv`
files, or `paths` for an explicit ordered list of CSV files. These adapters
accept CSV only. Export Handbook notebook pickle partitions to CSV before
using them. `TX_TIME_SECONDS` supplies Handbook event time; `TX_DATETIME` and
any timezone information remain diagnostic source fields. ULB `Time` is already
relative seconds. PaySim targets the documented hourly-step release and converts
`step` to seconds by multiplying by 3,600. A different step interpretation
requires a different adapter assumption.

Set `release`, `name`, and `currency` when known. Currency defaults to unknown;
the module does not infer it from a dataset name. The descriptor exposes clock
semantics, graph capabilities, numeric feature names, label meanings, ordering,
and source references. `source_catalog()` lists source adapters without opening
data. ULB and PaySim lack transaction IDs in these schemas, so event IDs derive
from source file position and row number. Reordering those files changes those
derived identities and the source fingerprint.

ULB supports `numeric` and `stream`. Relational views fail with an explanation
because the source provides no reliable account or merchant identity. Creating
invented customers from row numbers would change the modeling task.

Handbook `TX_FRAUD` is ground truth and `TX_FRAUD_SCENARIO` is diagnostic
information. PaySim `isFraud` is ground truth; `isFlaggedFraud` and new balances
stay outside prediction inputs. Unknown labels remain unknown and are excluded
from supervised fitting and retrospective classification counts rather than
treated as legitimate transactions. Absent labels, blank labels, `-1`,
`unknown`, and `null` represent unknown outcomes. These adapters expose final
labels as retrospective truth; none claims to record investigator feedback
times. The ordinary numeric training path uses available final labels within
each partition. Use replay with an explicit delay assumption when evaluating
learning from delayed feedback.

The [ULB configuration](../examples/datasets/ulb.config.json) and
[PaySim configuration](../examples/datasets/paysim.config.json) point to
`data/creditcard.csv` and `data/paysim.csv` at the repository root. Supply a
release you are permitted to use, check its schema and units, and edit the
configuration as needed. Merely using an adapter does not establish the
provenance, license, or representativeness of a third-party CSV copy.

## Prepare once and reuse

Preparation validates source records and writes an indexed SQLite store with
chronological iteration, event IDs, separate truth, and source provenance.
Global sorting happens on disk; a file's row order need not already be
chronological. A deterministic secondary order resolves timestamp ties for
storage, without claiming a measured causal order between tied events.

```sh
python experiment.py prepare --config examples/datasets/handbook-stream.config.json --output artifacts/handbook.sqlite
```

Use the prepared store through the same views:

```json
{
  "loader": "prepared_fraud",
  "path": "handbook.sqlite",
  "view": "numeric"
}
```

The path in this example is relative to whichever configuration contains it.
The supplied [prepared configuration](../examples/datasets/handbook-prepared.config.json)
uses the SQLite artifact produced by the preceding command.
Preparing a `stream` produces SQLite; preparing a `payments` view produces
the existing browser event JSON. A prepared dataset retains the source
descriptor and fingerprints so later runs can record exactly which data and
transformations were used. Reopening a prepared store also fingerprints the SQLite
file itself; editing it invalidates saved-partition evaluation for that artifact.

The direct Python API separates trusted dataset access from model inputs:

```python
import json
from pathlib import Path
from datasets.stream import open_source, open_prepared

config_path = Path("examples/datasets/handbook-stream.config.json")
config = json.loads(config_path.read_text())
with open_source(config, config_path.parent) as stream:
    print(stream.descriptor)
    print(stream.provenance)
    print(stream.count)
    for batch in stream.iter_batches(batch_size=16):
        event_ids = [event.event_id for event in batch]
        # The evaluator can request outcomes separately from observable events.
        outcomes = stream.truth_for(event_ids)

with open_prepared("artifacts/handbook.sqlite") as stream:
    first_run = [event.event_id for event in stream.iter_events()]
    second_run = [event.event_id for event in stream.iter_events()]
    assert first_run == second_run
```

Every iteration starts afresh. Close streams, preferably with `with`, to release
their database resources and any temporary preparation store. Observable
events expose identity, time, type, typed entities, and numeric inputs;
outcomes and diagnostics remain in the trusted data layer. Do not pass the
stream or separately fetched outcomes into a model's prediction method.

## Replay a history prefix and evaluation suffix

`datasets.replay.replay` accepts a model factory; return a fresh model for each run.
Events before `split_time` supply history. Events at or after that time are
predicted and joined to truth by event ID, independently of prediction order.
An empty prefix is a supported cold start; an empty evaluation suffix is an error.
The default groups equal timestamps atomically: all candidates in a group are
scored before any candidate in that group becomes persistent history or
releases feedback.

```python
import json
from pathlib import Path
from datasets.stream import open_source
from datasets.replay import replay

class AmountDemo:
    def predict(self, events, history):
        # An illustrative score in this fixture's invented amount units.
        return {event.event_id: min(event.features["amount"] / 300, 1.0)
                for event in events}

config_path = Path("examples/datasets/handbook-stream.config.json")
with open_source(json.loads(config_path.read_text()), config_path.parent) as stream:
    result = replay(stream, AmountDemo, split_time=43200,
                    history="growing", learning=False, batch_size=7)
    print(result["manifest"])
    print(result["rows"][0])
```

Choose `history="fixed"` to retain only the prefix, or `"growing"` to add each
observed timestamp group after scoring it. `HistoryView` exposes immutable
events and typed edges, with `neighbors(entity)` for graph-style access. The
candidate already carries its observable endpoints and features; the model
may use those when scoring. Only persistent history is restricted by the
chosen policy.

Models may implement `reset()` and `observe(events, history)`. Parameter
updates require `learning=True` and `learn(feedback, history)`. Supply
`label_delay` in seconds when modeling feedback availability that the source
does not record; a delay of zero still releases a tied group's labels only
after the group is scored. Feedback contains the observed event, its label,
and availability time, without diagnostic fields. With `label_delay=None`,
only recorded availability times can release feedback; the three current CSV
adapters have no such times, so their retrospective labels never reach a
learner unless a delay is explicitly supplied. Pending labels are not released
beyond the final observed timestamp. The manifest reports final evaluation
label maturity as unknown, retrospective, mature, or pending. Unknown labels never become
negative feedback. The run manifest records the feedback assumption so an
invented delay is not mistaken for a measured source field.

Retrospective result labels are for evaluation. Their presence in a report
does not mean the learner received them during replay. Choose `score_kind`
explicitly: `probability` validates values in `[0, 1]`, `risk` permits finite
scores where larger means more suspicious, and `logit` permits finite log odds.
`join_predictions` also supports externally computed event-ID/score rows and
checks exact population alignment; missing, extra, and duplicate predictions
are errors. Pass `event_ids` for an explicitly selected evaluation population
and `dataset_identity` to pin its source and configuration fingerprints or a
previous replay result's `dataset` object. Returned rows contain outcome
labels and diagnostics for trusted evaluation only.

Growing history, returned prediction rows, and model-owned state can consume
memory proportional to the run. `max_events` retains only that many of the most
recent events in the runner's history. It does not truncate the scored
population, returned rows, pending feedback, or model-owned state. With fixed
history, this retains the last events of the prefix. Duplicate-ID validation
also retains seen IDs, and an entire tied timestamp group must fit in memory
to preserve atomic scoring. Time-based windows,
out-of-order arrival, and source-specific investigator
protocols require additional implementations. Replay does not claim to
reproduce the Handbook's daily investigator/delay evaluation protocol.

## Import into the comparison pages

For saved datasets and models, start with the [web pipeline](pipeline.md):
**Data lab → Model trainer → Model comparison**. The data lab stores generated
or fixed dataset snapshots, including numeric-only ULB data. The trainer fits
compatible models with configurable parameters and chronological splits; the
comparison tool evaluates the ready artifacts on common held-out transactions.

The comparison webview has a **Benchmark datasets** picker. Start or restart
`python serve.py`, open `http://127.0.0.1:8000/index.html`, choose the invented
Handbook fixture, and click **Load dataset**. For that 48-row fixture, set shared
warm-up to 8 to leave transactions for evaluation. Every model uses the same
selected dataset and its existing checkpoint.

Choose **Fraud Detection Handbook CSV** or **PaySim CSV** to upload a source file
directly. Specify **EUR per source amount unit** and, optionally, a release name
and time interval. ULB is listed as tabular only because it cannot supply the
payment endpoints required by this page. Errors leave the previous dataset in
place; **Use synthetic scenarios** returns to generated scenarios.

Uploads support CSV files up to 6 MiB. To use a larger local source or a prepared
store, register its config when starting the server:

```sh
python serve.py --dataset-config examples/datasets/handbook-prepared.config.json
```

Repeat `--dataset-config` for additional datasets. The picker lists these local
entries, and uses their conversion and selection settings as defaults. Set a
`name` in each config for a useful display label. The selected interval must fit
the existing 256-account and 20,000-event limits; rows are never silently dropped.
Only startup configurations choose server files. Browser requests select catalog
IDs or upload CSV contents. CSV loading requires the local Python app; opening
the HTML directly still supports synthetic scenarios and payment JSON imports.

The command-line export remains available for both comparison pages:

```sh
python experiment.py prepare --config examples/datasets/handbook-payments.config.json --output artifacts/handbook-payments.json
```

Choose **Import event JSON** in either browser page and select the result.
The browser contract expresses elapsed minutes and EUR. The demo configuration
sets `amount_to_eur: 1.0` explicitly to interpret one invented fixture unit as
one EUR. This is a demonstration assumption, not a claim that the Handbook
source specifies EUR or a real exchange rate. For an unknown or non-EUR source
currency, the payment bridge requires an explicit positive conversion factor.
Numeric and plain graph views retain source amount units.
The `labeled_graph` view passes through the existing supervised fraud model's
EUR payment contract, so its example makes the same explicit demo conversion.

The payment bridge projects source transaction types to payment attempts and
preserves typed external identities. Browser imports remain limited to 256
accounts and 20,000 events; select a smaller time interval when necessary.
Importing data uses the pages' existing model checkpoints and policies. Training
an appropriate model and calibration remains a separate experiment step.

## Extend the module

Add a source adapter at the normalization boundary rather than branching on
dataset names inside models or experiment runners. Reuse the existing views
unless the new source requires a genuinely different prediction target.

The implementations and `ADAPTERS` registry live in
[`datasets/adapters.py`](../datasets/adapters.py); immutable contracts and
ordered storage access live in [`datasets/stream.py`](../datasets/stream.py).
A registry entry provides `parse`, `required`, `feature_names`, `entity_types`,
`source_unit`, `source_refs`, and `event_identity`. Its
`parse(row, generated_id)` callable returns `(TransactionEvent, Outcome)`.
`EntityRef(kind, id, role)` describes observable participants, using `source`
and `destination` roles for the current graph projection. For source schemas
with no stable entities, return an empty entity tuple and declare an empty
`entity_types` list. The preparation layer handles file iteration, duplicate
IDs, sorting, fingerprints, and persistence for every adapter.

An adapter should declare source identity and version, required columns, numeric
inputs, clock semantics, currency, available graph roles, and source references.
Normalize a row into an observable event plus a separate truth record. Preserve
source event IDs where available; declare deterministic derived IDs when a
source has none. Validate duplicate IDs, finite times and amounts, label values,
and source-specific missing values. Keep labels, post-event fields, and
diagnostics outside observable features. Do not fabricate relations when stable
entities are absent.

Register the adapter in the source adapter registry and add a small fixture,
configuration, and tests of its actual assumptions. Existing
`fraud_dataset`/`prepared_fraud` loader registrations and model adapters remain
reusable. A different output contract, such as account-level risk rather than
transaction fraud, warrants a separate task and capability declaration.

Useful adapter validation checks include repeatable ordering after shuffled
input, preservation of timestamp groups across different batch sizes, duplicate
ID rejection, unknown-label handling, leakage-prone field exclusion, and honest
failure for unsupported views. Replay checks should show that changing a
future suffix cannot alter earlier predictions, learning waits for released
labels, and shuffled prediction rows still align by event ID.

For benchmark results, record the source release and fingerprint, adapter
version, configuration and transformations, target and label availability,
chronological boundaries, graph/history policy, model settings, seed, and metric
definitions. Report results per dataset before combining results across tasks
or different fraud prevalences.
