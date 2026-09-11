# Payment model tools

Open **index.html** for the offline model comparison, or
**xgboost-analytics.html** for the completed-scenario XGBoost dashboard. Shared
navigation links the two tools. Both pages contain their own scripts and model
data and work directly from disk, without a server or network connection.

The comparison replays synthetic payment streams and scores each proposed
payment before the event is committed.

## Model and dataset architecture

Start with [the redesign plan](docs/plans/model-dataset-architecture.md).
Implementations live separately from the runners and UI:

| Location | Responsibility |
|---|---|
| `models/registry.json` | Model IDs, implementation paths, supported input schemas and runtimes |
| `models/implementations/<model>.py` | Each model's training implementation or explicit architecture declaration |
| `models/implementations/<model>.js` | Each browser model's inference adapter |
| `models/implementations/_neural.py`, `_temporal.py` | Legacy neural primitives and fixed-encoder temporal prototypes |
| `models/implementations/dygformer.py`, `tami.py`, `dyg_tami.py` | Published DyGFormer backbone, TAMI LTE/TRC, and native fitted model |
| `models/<model>.json` | Existing trained demo checkpoints, separate from code |
| `datasets/registry.json` | Dataset loaders and their schemas |
| `datasets/implementations/` | Separate numeric CSV, payment CSV, payment JSON and synthetic providers |
| `framework/contracts.py` | Dataset and fitted-model interfaces |
| `framework/experiments.py` | Shared chronological splitting, fitting, evaluation and artifact handling |
| `examples/` | Editable dataset and experiment configurations |

**The new `xgboost_native` uses the official XGBoost library.**
`logistic_regression` uses scikit-learn with training-only standardization.
Both support fitting, saving and reloading on user-supplied numeric CSV data.
The existing browser `xgboost` remains the custom NumPy tree implementation.
GRU/statistics entries use custom trained neural components; the four
browser DyGFormer/TAMI-inspired entries use fixed random encoders and optional trained
logistic heads. Their registry labels explicitly identify them as prototypes.
The separate `dyg_tami_native` entry trains the published DyGFormer + TAMI
architecture end to end in PyTorch.

### Train and evaluate a real model

Install optional dependencies in your Python environment:

```sh
python -m pip install -r requirements-models.txt
python experiment.py list
python experiment.py train --config examples/xgboost-experiment.json --output artifacts/my-xgboost
python experiment.py evaluate --config examples/datasets/numeric-table.config.json --artifact artifacts/my-xgboost --partition test --output artifacts/test-evaluation.json
```

First edit the example configurations to point to your own CSV and its columns.
Paths are relative to each configuration file. These examples refer to the same
`examples/datasets/transactions.csv`; no real dataset is bundled or downloaded.
Choose numeric feature columns explicitly, with an ID, timestamp and optional
binary outcome column. Unknown outcomes stay unknown. ISO timestamps require a
timezone. Features must already be available at decision time; the framework
cannot detect future information embedded in a supplied feature column.

Training uses chronological train/validation/test partitions, keeping equal
timestamps together. Only known training outcomes are fitted; validation F1
selects the cutoff unless `decision_threshold` is provided. The final test report
contains predictions, metrics and, for native XGBoost, additive feature
contributions in raw-margin units. This is completed-dataset analysis without
step-by-step replay. The model, feature order, implementation hash, input and
configuration hashes, library version and split IDs are saved together.
Evaluation rejects incompatible feature order and modified model artifacts;
saved partitions require the original data and loader configuration. Evaluating
`--partition all` includes every supplied row and reports known training overlap;
it is not automatically a held-out test.

Switch models by changing `model` to `logistic_regression` and replacing the
XGBoost-specific parameters with, for example, `{"C": 1.0}`. Switch datasets by
changing the loader configuration. Add a model by implementing `fit`, `predict`,
`save` and `load` in its own module and registering its supported schema. Add a
dataset by implementing `load(config, base_dir)` in its own module and registering
it. Incompatible model/dataset combinations fail explicitly.

Native models run through the Python numeric-table or temporal-graph runner. The browser
dashboards still use their bundled event-model checkpoints; a native JSON model
is not a browser checkpoint. Loaders currently operate in memory. Categorical
preprocessing and automatic causal feature generation remain separate extensions.

### Train DyGFormer + TAMI

The authors' [TAMI repository](https://github.com/Alleinx/TAMI_temporal_graph)
already combines DyGFormer with logarithmic time encoding and directed-pair
historical memory. This implementation uses that complete backbone and decoder,
including learned co-occurrence encoding, four feature channels, patching and
transformer attention. Inspect [source provenance and adaptations](models/implementations/UPSTREAM.md)
and the [integration plan](docs/plans/dygformer-tami.md).

The following example runs on the existing synthetic takeover fixture:

```sh
python -m pip install -r requirements-temporal.txt
python experiment.py train --config examples/dyg-tami-experiment.json --output artifacts/dyg-tami
python experiment.py evaluate --config examples/datasets/dyg-tami-payments.config.json --artifact artifacts/dyg-tami --partition test --output artifacts/dyg-tami-evaluation.json
```

Outputs include `model.npz`, the experiment manifest and `test-report.json`.
Training logs one line per epoch to stderr, including training and validation
loss, best-epoch or patience status, epoch duration and total elapsed time. Fraud
experiments also report partition label counts and the transitions from link
pretraining to fraud fitting, policy-threshold selection and final evaluation.
The checkpoint stores every learned tensor and a detached pair-memory snapshot,
without pickle. Evaluation resets state and replays the observed prefix using
frozen model weights, so rerunning validation after test is safe. The report
contains final AP, ROC AUC, log loss, accuracy and per-link scores. Validation
loss selects the epoch; validation F1 selects the cutoff. Test outcomes never
select either. Device, dimensions, patch size, history length, gamma, epochs and
optimizer settings are explicit in the experiment JSON. The small CPU defaults
are not the authors' benchmark-specific best configurations.

**This is dynamic link prediction, not fraud classification.** Observed payments
are positives, including observed attempts that did not settle. Fraud flags are
ignored. Deposit and report events are omitted. With the payment bridge, the
historical edge feature is log-transformed amount, and missing static node
features are explicit zero vectors. The current payment amount cannot enter its
own historical features. A high score means the link resembles observed links
relative to sampled counterexamples; it is not a calibrated fraud probability.

The native runner scores all events at a timestamp before committing their pair
memories. Neighbor histories contain only strictly earlier events. Negatives
sample destinations observed by the query time and exclude self-links and all
same-time positive links for that source. A row without a valid negative still
updates history but is excluded from paired loss/metrics; reports give the count.
Repeated pairs at the same timestamp average their proposed memory updates.
These chronology and sampling choices are documented adaptations of upstream's
minibatch protocol, so local scores are not paper benchmark reproductions.

For real temporal graph data, replace the experiment's `dataset` block with
the contents of `examples/datasets/dyglib.config.json`. It supports DyGLib's
`ml_<dataset>.csv`, `ml_<dataset>.npy` edge attributes and
`ml_<dataset>_node.npy` static node attributes. CSV node/edge IDs index the source
feature matrices; internal IDs and zero padding are normalized by the loader.
For other CSV files, map `id`, `source`, `destination`, and `time`, and list numeric
`features` explicitly. Features must be available at the relevant historical
interaction, and static node features must not encode future outcomes.
No benchmark data is downloaded automatically.

Graph models implement `fit_graph`, `predict_graph`, `save`, and `load_graph`;
providers return `TemporalGraphDataset`. Registration declares the compatible
schema. Existing browser IDs resolve to `*_prototype.py` compatibility entries;
the native PyTorch checkpoint is not silently substituted into browser replay.

To verify the implementation:

```sh
python tests/temporal/test_dyg_tami.py
python tests/temporal/test_upstream_parity.py
```

The second test compares embeddings, logits and gradients to a checked-in
reference generated by the pinned upstream classes, independent of the local
model implementation. To regenerate it from an upstream checkout, run
`python scripts/export_tami_oracle.py --upstream /path/to/TAMI_temporal_graph`.

### Inspect your own payment events in the browser

Both pages offer **Import event JSON**. Convert a mapped payment CSV first:

```sh
python experiment.py prepare --config examples/datasets/payment-events.config.json --output artifacts/payment-events.json
```

Edit the configuration's column mapping and input path first, then select the
resulting JSON in either page. The loader preserves external account identities,
normalizes timestamps to elapsed minutes, validates chronological events and
keeps outcomes outside model inputs. This initial payment schema accepts EUR;
it does not silently convert currencies. Browser imports are bounded to 256
accounts and 20,000 events; use a bounded subset or the Python numeric runner for
larger inputs. Importing data does not retrain the synthetic demo checkpoints or
recalibrate their historical policies. The UI states that limitation explicitly.

## XGBoost analytics

The analytics tool runs the selected scenario to completion using the existing
supervised tree checkpoint. It presents final metrics, a confusion matrix, score
distributions, all 27 features' contributions and a searchable payment table.
Select a payment to inspect its contribution waterfall, readable feature values,
transformed model inputs and the path it took through any of the 32 trees.
Feature charts follow the filtered payment population. CSV and JSON exports
include that population's predictions and contributions.

There are no replay controls. Each payment's features are captured before its
event is applied, and each decision retains the threshold used at that time.
Observed-history execution records blocks while allowing all transfers to
settle. Changing the decision policy reuses completed predictions; changing the
scenario produces a new completed report. Only supervised tree scoring is used:
the comparison's no-label XGBoost mode is a separate rarity formula.

The feature explanations are exact path-dependent Shapley values for the
declared reference: unweighted labeled rows from the historical fitting
partition. `models/xgboost-explanations.json` records node counts, feature schema
and checkpoint/history/source hashes. The builder regenerates it when stale,
without retraining. Unknown outcomes contribute to preceding history but are
excluded from reference rows; validation episodes are excluded from the
reference population. This count convention differs from native XGBoost's
Hessian cover convention.

The reference expected margin plus all feature contributions equals the raw
prediction in log-odds. Probability and score in bits are transformations of
that total; per-feature contributions are not probability percentages or causal
effects. Unused features remain visible with zero contribution. The checkpoint
is a NumPy implementation of XGBoost-compatible trees, not a native model import.

Classification metrics exclude warm-up and unknown outcomes. Block rate excludes
warm-up and includes all remaining payments. The dashboard shows those
denominators and marks unavailable metrics accordingly. Probabilities and
synthetic scenario metrics are illustrative, uncalibrated model outputs.

## Adding tools

`tools/registry.json` declares each tool's label, template, scripts, model IDs,
embedded data ID and output filename. Add a tool directory and registry entry to
build another page. Shared runtime code lives in `shared/runtime/`; shared page
styles/navigation live in `shared/ui/`; bundled libraries and licenses live in
`shared/vendor/`. Tool code depends on shared code, which is independent of UI.

Edit source templates and scripts, then run `python build.py`. Generated HTML
pages are deliverables. `python build.py --tool comparison` or
`python build.py --tool xgboost-analytics` builds a selected page. Root build and
training commands remain entry points to `scripts/` and `training/`.

In the comparison tool, setting changes keep the controls responsive. New replays run in short chunks;
choosing another setting cancels the outdated result. In same-history mode,
changes to costs, the cutoff policy, alpha or warm-up reuse existing model
scores and account states. Switching the inspected model/account only redraws
the view. Recently used setups and rewind checkpoints are cached with fixed
limits. Changing an enforced blocking policy uses its own history.

For the 64-account mixed example at event 530, previous local replay benchmarks measured
policy changes at a median 0.99 ms (previous version: 3,065 ms). A fresh replay
took 1.28 seconds, yielding 135 times; the longest measured work slice was
10.7 ms. These historical figures exclude DOM rendering and vary by device and
checkpoint. Run `node scripts/benchmark.js` to measure the current setup.

## What is swappable

The comparison uses the same request stream and evaluation fixtures for every
design. The selected policy is applied per model, and each checkpoint records
its own offline training recipe:

| Design | Learned state | Graph readout |
|---|---|---|
| **History statistics** | None | None; pair and activity statistics only |
| **GRU memory** | One 8-value account state, updated by typed events | None |
| **GRU + graph mean** | Same GRU state | Two layers with uniform neighbor averaging |
| **GRU + graph attention** *(default)* | Same GRU state | Two learned attention layers over up to four recent incidents |
| **XGBoost trees** | No learned account state | Offline boosted trees over causal statistics |
| **DyGFormer history** | Retrieved temporal event history | Sinusoidal time-encoded history readout |
| **TAMI pair history** | Bounded directed pair state | Pair-keyed elapsed-time readout |
| **DyGFormer + TAMI** | Temporal history plus pair state | No graph aggregation |
| **DyGFormer + TAMI + GNN** | Temporal history plus pair state | Two-hop time-respecting graph readout |

The default therefore **does use graph structure**. Its readout sees account
memories plus directed completed-payment incidents with amount and age. The
displayed network is a bounded explanation view; the replay itself processes
the full event stream.

Every design supports the same **Training mode** toggle:

- **No fraud labels:** temporal models use self-supervised likelihood surprise;
  XGBoost uses a causal rarity score.
- **Use flagged history:** temporal models use an offline logistic fraud head;
  XGBoost uses its offline boosted fraud checkpoint.

The toggle changes the offline scoring head. A separate **Decision policy**
control chooses how the cutoff `τ` is fitted, in either model training mode.
`--fraud-flags N` controls the flagged historical pool used when producing the
supervised checkpoints; changing it requires retraining and rebuilding.

The four temporal-family entries are compact prototypes of the earlier
DyGFormer + TAMI + GNN proposal. They intentionally share one causal adapter
boundary so the comparison isolates the added history, pair, and graph
mechanisms instead of mixing unrelated training and serving code.
Their bounded event buffers and sinusoidal projections are faithful to the
mechanisms being explored, but are not reproductions of the papers' complete
architectures.

The XGBoost checkpoint is trained offline from the flagged historical examples
requested by `--fraud-flags`. In flagged-history mode it uses
`−log₂(1 − estimated flagged-fraud probability)`; in no-label mode it falls
back to a causal feature-rarity score. Both use the same chronological replay
and cutoff policy.

Model implementations are behind a small adapter boundary:

```js
FraudAdapters.register('my_design', {
  initialize(model, accountCount) { /* return account states */ },
  update(model, state, event, settled) { /* return next states */ },
  readout(model, state) { /* return embeddings/evidence */ },
  predict(model, state, request) { /* return score and parts */ }
});
```

The runner and decision policy are shared, so a new adapter can be compared
without changing replay, cutoff, evidence, or evaluation code.

## Reading the score and memory

In no-label mode, the displayed **bits** for temporal designs are negative
log-likelihood:

```text
bits = -log2(probability of the observed recipient, amount bin, and timing bin)
```

Higher bits means the request is less expected under the model. It is an
anomaly score, not a fraud probability. Flagged-history mode instead displays
`−log₂(1 − p_flagged)` from the selected offline fraud head, also uncalibrated
by this demo. The cutoff `τ` is learned using the selected decision policy.

## Decision policies

| Policy | What is learned | Use of outcome labels |
|---|---|---|
| **Shared budget** (default) | Each model learns its own `τ`, targeting a common chosen `α` | None for the policy; quantile initialization and online quantile-loss updates |
| **Individual settings** | Each model can use its own unlabeled budget, fixed `τ`, validation costs, or automatic objective | Depends on that model’s selected strategy |
| **Auto-tune every model** | Each model independently selects `τ` for F1, F2, or balanced accuracy | Held-out historical outcomes only; each cutoff is frozen throughout replay |

In **Individual model settings**, each model has its own target α, warm-up, fixed τ,
false-block cost, missed-fraud cost, and auto-tune objective. Changing one model
does not change the others. Costs are relative per-request costs, not currency
amounts. Changing a model’s costs selects a different threshold from its
precomputed validation scores; it does not retrain the encoder. Equal-cost ties
prefer fewer blocks, and equal scores receive the same decision (`score > τ`).

An unsupervised model can use a policy tuned with outcomes. The demo labels
these controls separately: **No labels for the model + Shared budget** is the
fully unsupervised combination. The blocking budget or cost tradeoff is an
operational choice; minimizing blocks alone would simply allow everything.

Policy validation uses the final 20% of episodes in `xgb-training.json`, excluded
from supervised weight fitting. In this export that is four held-out episodes:
2,385 requests, 24 fraud flags, 2,346 legitimate outcomes, and 15 unknowns.
Unknowns remain in chronological history and in the denominator for the
implied blocking rate, but are excluded from the error-cost objective. The
historical export contains payment events; deposits and live report events
are absent. Episodes restart their account state independently.

This is synthetic episode separation, not evidence of real-world chronological
generalization. The limited, selectively flagged validation sample affects the
fitted policy; the implied `α` is neither a forecast nor a false-positive rate.
These validation episodes were also used to report supervised-head log loss;
their fitted costs are not independent test results. Evaluate replay scenarios
separately, using seeds distinct from the historical generator. The UI's
**Block rate** and **Error cost** columns describe processed replay requests.
Error cost is populated only for cost-tuned policies and uses that model's own
costs;
**Recall at equal budget** always uses the same ranking budget for all models,
independent of the different fitted rates.

For a GRU design, each account has eight learned state dimensions. The cells
are coordinates of one compressed behavioral state; they do not have fixed
human meanings such as “fraud” or “velocity”. Their interpretation comes from
the events, graph readout, score parts, and retrieved evidence. The statistics
design intentionally has no learned account memory.

## Controls

- **Scenario / network size:** 32, 64, or 112 registered accounts and roughly
  450, 1,000, or 2,100 background events, plus the scenario activity.
- **Training mode:** swap all nine designs between no-label scoring and
  flagged-history scoring.
- **Policy scope:** shared unlabeled budget, individual model settings, or auto-tune every model.
- **Individual settings:** select a model and choose its strategy and parameters.
- **Auto objective:** choose F1, F2, or balanced accuracy; it is applied separately to every model.
- **Warm-up:** choose how many initial request scores (default 128) are allowed
  under the shared-budget policy; tuned policies start with their fitted cutoff.
- **Comparison protocol:**
  - *Same history* records every model's decision while all transfers execute.
  - *Independent blocking* lets each model's blocks change its own later state.
- **Process event:** score the pending payment, apply the selected policy, then
  update memory and the cutoff.
- **Evaluate:** reveals replay outcomes only for inspection; they never enter
  scoring or policy tuning.

Reports arrive as delayed evidence and never rewrite an earlier decision.
Blocked attempts update behavioral memory where present, but do not create a
completed-payment edge or transferred value.

## Reproduce

For the existing synthetic browser models, Python 3.10+ with NumPy and Node.js are sufficient. Native model tests additionally use `requirements-models.txt`:

```bash
python train.py --design all --steps 140 --fraud-flags 96 --warmup 128
python build.py
python tests/framework/test_framework.py
python tests/shared/test_build.py
python tests/shared/test_gradients.py
python tests/xgboost-analytics/test_explanations.py
node tests/shared/test_policy.js
node tests/comparison/test_compare.js
node tests/shared/test_logic.js
node tests/comparison/test_ui.js
node tests/comparison/test_responsiveness.js
node tests/xgboost-analytics/test_explanations.js
node tests/xgboost-analytics/test_analytics.js
node scripts/benchmark.js
```

For real-browser checks, make Playwright available in your development
environment and run `node tests/xgboost-analytics/test_browser.js` and
`node tests/framework/test_browser_import.js` after building.
If Playwright is installed elsewhere, set `PLAYWRIGHT_MODULE` to its module path.
The browser suite runs offline and covers selection, filtering, exports,
cancellation, keyboard navigation and desktop/mobile layout. Set
`ANALYTICS_SCREENSHOTS` to a directory to save screenshots during the checks.

`--steps` controls neural optimizer iterations. `--fraud-flags` controls the
number of flagged historical events used by every supervised head and by the
XGBoost training partition; a small held-out flagged sample is kept for
policy fitting and reporting validation loss. Unflagged fraud-like events are
retained as unknown and excluded from supervised loss. `--warmup` records the
default online cutoff warm-up and can also be changed interactively in the
demo. `--xgb-trees` controls the exported tree count.

`build.py` automatically regenerates `policy-validation.json` after checkpoint,
score implementation or historical data changes. It hashes the complete model
files (including optional supervised heads), replay/scoring code, fitting code,
and history export. Run `node scripts/tune_policy.js` to regenerate explicitly. No model
is retrained during policy fitting or live replay. Missing labelled validation
data disables cost/automatic policy selection; shared-budget learning remains
available.

`models/` contains one checkpoint per design. `xgb-training.json` records the
offline flagged/unknown training stream. `parity/` and `parity-supervised/`
contain independent episodes used to verify both JavaScript inference modes.
`datasets/` contains larger saved fixtures; `datasets/implementations/synthetic_payments.js` can generate more
seeds and sizes.

`performance-results.json` records local replay timings and event-loop yields;
these exclude browser layout and rendering. Cached datasets/checkpoints are
treated as immutable. Editing them requires rebuilding or a fresh comparison.
The replay API also offers `seekAsync`, cancellation and `ComparisonCache`;
the page exposes `demo.whenIdle()` for deterministic asynchronous UI checks.

The predictive training task is self-supervised by default: recipient,
amount-bin, and sender-gap-bin likelihoods are learned from observed event
history. The optional flagged-history mode adds a separately trained fraud head
for every design. Neither mode is calibrated production blocking accuracy;
synthetic metrics are illustrative.

## Files

`shared/runtime/model-registry.js` defines the browser model boundary;
`model-adapters.js` is its compatibility bootstrap. `shared/runtime/model.js`
implements chronological replay, `policy.js` selects thresholds and `metrics.js`
evaluates decisions. Tree inference and feature attribution live in
`models/implementations/xgboost_numpy_core.js`; synthetic event generation lives
in `datasets/implementations/synthetic_payments.js`. The old runtime paths remain
compatibility wrappers.

`scripts/tune_policy.js` exports held-out historical score frontiers.
`tools/comparison/` runs models side by side and `tools/xgboost-analytics/`
collects and displays completed-run analytics. `training/train.py` preserves the
legacy synthetic training CLI and imports; actual architectures and training
routines live in `models/implementations/`, using `training/autodiff.py` for
neural differentiation. Legacy training retains its historical file conventions;
new native plugins use the independent dataset and experiment contracts.
`scripts/build.py` assembles standalone pages;
`scripts/export_xgboost_explanations.py` exports reference counts. Run it with
`--check` to verify freshness without writing files.

The two-layer memory/readout structure is TGN-inspired, but this demo is not a
reproduction of the official TGN or PRISM implementations.
