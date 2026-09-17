# Dataset preparation, training, and comparison

Complete the [README setup](../README.md#setup), then start the app from the
repository root with that environment active:

```sh
python serve.py
```

Open the printed **Data lab** address, normally
`http://127.0.0.1:8000/data-lab.html`. The top navigation connects the data lab,
model trainer, and comparison tool. These preparation tools require the Python
server; opening their HTML files directly shows how to start it.

Hover over model parameters or feature names for extended explanations, or use
their info buttons with a keyboard or touch. See [parameter and feature help](info-windows.md)
for coverage and how to add descriptions alongside new definitions.

## Prepare and inspect a dataset

The data lab defaults to **Configurable customer and terminal payments**. Adjust
customer and terminal counts, transaction count, duration, fraud fraction, and
seed, then choose **Generate and store dataset**. The generator creates invented
EUR transactions in the Handbook CSV schema. It is intended for simulations and
integration checks; it does not reproduce the published Handbook benchmark.

The **Payment scenarios** generator reuses the comparison tool's existing
scenarios. Its settings include scenario, network size, random seed, reporting
delay, and forwarding delay. Explicit fraud report times control when an outcome
is available to training. Some scenario splits contain only one outcome class
or no confirmed frauds; choose an appropriate dataset and split for supervised
training and validation.

To use fixed data, select **Import a fixed CSV dataset**, choose Handbook, ULB,
or PaySim, and upload its CSV. Set a release identifier where appropriate.
Handbook and PaySim require the amount conversion in EUR for payment views;
ULB has anonymized numeric features and no graph account identities.
Uploads are limited to 6 MiB. Larger local datasets can be registered when
starting the server:

```sh
python serve.py --dataset-config examples/datasets/handbook-payments.config.json
```

Each configured or uploaded source becomes a stored snapshot. The server seeds
the bundled 48-row invented Handbook fixture automatically. No external
benchmark is downloaded automatically. See the [dataset module guide](datasets.md)
for source formats, preparation, feature selection, and dataset extensions.

Select a row in **Dataset library** to inspect its row count, label coverage,
supported views, source settings, provenance, content fingerprint, and a sample
of records in chronological order. Unknown labels are displayed explicitly.
Use **Train on this dataset** to carry its selection into the model trainer.
Generated datasets retain their parameters and source snapshots for reuse.

For graph-capable datasets, open the graph explorer from the inspector to browse
a bounded sample, inspect entities and relationships, and explore neighborhoods
or search beyond the current view. See the [graph explorer guide](graph-explorer.md)
for controls, scalability limits, and provider extension points.

## Choose and inspect shared features

The dataset's **Features** view lists source columns and derived payment
features, including the 27 amount, account activity, recent activity, pair
history, and historical-mean features previously calculated inside the demo's
XGBoost implementation. Toggle each feature independently, inspect its definition
and statistics, then save the selection as a feature dataset. Source columns
can also be disabled. At least one available feature must remain selected.

The **Graph features** filter also exposes global PageRank, personalized
PageRank in both directions, connected-component features, and distinct contact
degrees. **Add graph features** selects them in one action. Their definitions
include approximation methods, budgets, error bounds, and graph orientation.
See [graph feature methods and limitations](graph-features.md) for causal
history, inexpensive approximations, and bipartite source handling.

Saving creates a new dataset ID with its own name, ordered feature list,
calculated values, recipe version, checksums, and parent dataset identity.
The original dataset and models trained from it stay usable. Selecting another
set of features creates another version; it does not change an existing run.
Record previews show the saved input columns. The trainer also lists the
dataset's numeric inputs and records the features used by each new run.

Native XGBoost, logistic regression, and TabFM consume the same stored numeric
columns. Supervised graph models receive those columns as edge features on
the same transactions and as current-transaction inputs to their fraud head,
with per-column normalization fitted only on training records. Labels, label availability, account identities, and
chronological partitions retain their existing roles. The browser's bundled
checkpoint replay continues to use its checkpoint-compatible feature contract.

Derived history uses **strictly earlier timestamps**: simultaneous transactions
do not affect one another. Counts and amounts describe observed payment
attempts, independently of fraud labels and simulated blocking decisions.
Deposits affect activity counts and gaps; outcome reports never affect input
features. Recent windows include the lower boundary of the preceding 60 minutes.
History uses earlier events retained in the stored source, including context
before a prepared stream's selected interval, and continues across chronological
partitions using observed inputs only. Payment JSON snapshots contain only their
stored interval. No label-based aggregation
or whole-dataset normalization is used to construct these features.

Amounts retain the source numeric view's units (EUR for payment JSON). Amount
bucket boundaries and transformations are inspectable in each definition.
Datasets without stable account identities, such as ULB, can use amount-derived
features; account-history features explain why they are unavailable. The
statistics describe the full saved population and transformed input values,
including disabled columns that are available for selection. They are
descriptive inspection data and are not fitted preprocessing parameters.

The HTTP API exposes the same workflow:

```text
GET  /api/pipeline/datasets/{dataset_id}/features
POST /api/pipeline/datasets/{dataset_id}/features
     {"name":"Payment history experiment",
      "features":["amount","log_amount","sender_out_count","pair_total_count"]}
```

The POST response contains the new dataset metadata. Use that dataset ID when
training models. Calculated input matrices are stored with the dataset and
reused after server restarts; models do not implement these recipes themselves.

## Configure and train models

In the model trainer, choose a dataset and model. The model menu describes
availability for that dataset and the dependencies installed in the server's
Python environment:

| Model | Dataset view | Main settings |
| --- | --- | --- |
| Logistic regression | Numeric | Regularization, iterations, class weighting, seed |
| XGBoost (native) | Numeric | Boosting rounds, tree depth, learning rate, sampling, class weight, seed |
| TabFM (Google foundation model) | Numeric | Context row limit, ensemble members, prediction batch size, seed |
| DyGFormer + TAMI (supervised fraud) | Labeled graph | Epochs, patience, batch size, learning rate, dimensions, attention heads, history length, seed |

TabFM uses frozen pretrained weights and labelled examples from the training
partition as context. Its action is **Prepare and save model**. The same
validation cutoff, held-out test evaluation, and saved-model comparisons apply.
Install its optional dependencies with Python 3.11+; the classification download
is approximately 6.1 GiB. See [TabFM setup and artifacts](tabfm.md), including the
weights' non-commercial, non-production license.

Every model also offers **Fixed fraud probability cutoff (optional)**. Leave it
blank to select the cutoff from validation outcomes, or deliberately supply a
fixed probability such as `0.5`. A fixed cutoff can be useful when a validation
window has only one class. Supervised fitting still needs usable training
labels; changing a cutoff cannot supply missing training examples.

The default split is 60% training, 20% validation, and 20% test. Adjust training
and validation percentages; the test share is the remainder. Records are ordered
by time, and equal timestamps stay together, so the UI's estimated row counts can
differ from the final counts. The saved run records its exact partitions.
Native temporal training divides the validation share equally between model
selection and policy cutoff selection. Explicit delayed outcomes are masked at
the corresponding training and validation boundaries.

Choose **Train and save model** to submit a server job. **Training activity**
shows its state and retained log messages. You can change settings and submit
another run while a previous job is working. Returning to the page reloads
jobs and completed models from the server. Failed jobs display their reason and
do not become ready models.

Each ready model includes the saved parameters, split, exact partition counts,
test metrics, decision cutoff and its source, dataset identity, and model
checksum. Select one or more ready models, then choose **Compare selected
models** to pass their IDs to the comparison tool.

## Compare saved models

The comparison tool's prepared-model workflow selects a stored evaluation dataset
and ready model runs. For a training dataset, use the held-out test partition.
If selected models used different splits, evaluation uses the intersection of
their test records. All selected models score the same eligible rows, with
predictions joined by transaction ID.

For a new compatible dataset, explicitly select evaluation of all its rows.
The server checks dataset fingerprints and observed records to reject overlap
with a selected model's training or validation data. It also checks model and
artifact identity before evaluating saved models.

Thresholds come from each saved run. Comparison reports include per-model
metrics and the shared transaction population; the existing browser payment
replay remains available alongside this prepared-model workflow.

## Storage and extending the workspace

Datasets, job records, training artifacts, and comparison reports live under
`artifacts/web-pipeline` by default. Use a separate directory for another
workspace:

```sh
python serve.py --port 8001 --pipeline-dir artifacts/my-benchmark-workspace
```

Reloading a page preserves stored work. Restarting the server preserves completed
artifacts; interrupted jobs are marked failed with an explanation on startup.
The selected dataset in a URL takes precedence over the browser's remembered
dataset selection.

Dataset preparation is implemented in `framework/pipeline_data.py`, which
reuses the dataset adapter and view contracts. Training and comparison jobs are
implemented in `framework/pipeline_training.py`, which delegates fitting to the
existing experiment runners. `framework/pipeline_service.py` exposes these
capabilities to HTTP. The frontend renders parameter controls from server
schemas, so generator and model parameter changes have one authoritative source.

To add a source, first implement and register its dataset adapter as described
in the dataset module guide, then expose its import configuration and supported
views in the store. To add a generator, expose its parameter schema and validated
generation branch in the store. To add a trainable model, register its experiment
implementation and add a bounded parameter schema and dataset view to
`MODEL_SETTINGS`. New entries then appear in the appropriate web selector.

To add a derived feature, register a `FeatureRecipe` in
`datasets/features.py::FEATURE_RECIPES`. Supply a unique ID, label, description,
group, unit, transformation, history window, observable requirements, version,
and `calculate(context)` function. Requirements determine availability for each
source, and the Data lab controls and statistics are generated from the registry.
The context contains the current observable event and strictly earlier payment
history. Keep outcomes out of recipe inputs. Increment `RECIPE_VERSION` whenever
calculations change so new materializations receive a distinct identity; saved
feature datasets keep their stored values.

The common payment definitions and legacy replay adapters live in
`datasets/payment_features.py` and `datasets/payment_features.js`. The replay
adapters preserve the existing checkpoints' ordered, settled-payment history;
dataset recipes explicitly use observed attempts and timestamp groups. New
dataset recipes do not require edits to model implementations or frontend code.

Saved feature variants retain their original recipe catalog as well as their
selected values. To use newly added recipes, follow **Original source** in the
inspector and create a new feature dataset from that source. This keeps older
experiments reproducible while allowing the feature library to grow.
