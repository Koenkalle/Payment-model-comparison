# Payment model comparison

Open **index.html** for the offline demo. It replays larger synthetic payment
streams and scores each proposed payment before the event is committed.

Setting changes keep the controls responsive. New replays run in short chunks;
choosing another setting cancels the outdated result. In same-history mode,
changes to costs, the cutoff policy, alpha or warm-up reuse existing model
scores and account states. Switching the inspected model/account only redraws
the view. Recently used setups and rewind checkpoints are cached with fixed
limits. Changing an enforced blocking policy uses its own history.

For the 64-account mixed example at event 530, local replay benchmarks measured
policy changes at a median 0.99 ms (previous version: 3,065 ms). A fresh replay
took 1.28 seconds, yielding 135 times; the longest measured work slice was
10.7 ms. These figures exclude DOM rendering and vary by device. Model outputs
and all 18 historical threshold-fit frontiers remain unchanged.

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

Python 3.10+ with NumPy and Node.js are sufficient:

```bash
python train.py --design all --steps 140 --fraud-flags 96 --warmup 128
python build.py
python test_gradients.py
node test_policy.js
node test_compare.js
node test_logic.js
node test_ui.js
node test_responsiveness.js
node benchmark.js
```

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
and history export. Run `node tune_policy.js` to regenerate explicitly. No model
is retrained during policy fitting or live replay. Missing labelled validation
data disables cost/automatic policy selection; shared-budget learning remains
available.

`models/` contains one checkpoint per design. `xgb-training.json` records the
offline flagged/unknown training stream. `parity/` and `parity-supervised/`
contain independent episodes used to verify both JavaScript inference modes.
`datasets/` contains larger saved fixtures; `scenarios.js` can generate more
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

`model-adapters.js` defines the swappable model boundary; `model.js` implements
chronological replay; `policy.js` selects thresholds and `tune_policy.js` exports
held-out historical score frontiers; `comparison.js` runs models
side by side; `scenarios.js` generates event streams and offline labels;
`train.py` and `autodiff.py` train the temporal checkpoints and export the
XGBoost-compatible trees; `build.py` assembles the standalone offline page.

The two-layer memory/readout structure is TGN-inspired, but this demo is not a
reproduction of the official TGN or PRISM implementations.
