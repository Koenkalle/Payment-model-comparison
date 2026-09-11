# Fraud-label training for DyGFormer + TAMI

Status: implemented. This document retains the design and acceptance criteria;
usage and training commands are in [README.md](../../README.md#train-dygformer--tami-with-fraud-labels).
The default trained artifact is `models/native-dyg-tami-fraud/`.

The six-episode synthetic example reserves 1242 / 634 / 409 / 425 payments for
training, model selection, policy selection, and test, containing 34 / 21 / 13 /
16 fraud outcomes respectively. Both encoder modes were trained with the same
initialization and reserved data. Their results are recorded in
`models/native-dyg-tami-fraud/encoder-comparison.json`; the fine-tuned test average
precision is 0.2012 and the frozen baseline is 0.0573. These are fixture results.

## Intended behavior

Selecting **Fraud-label training** in the existing comparison activates a native
DyGFormer + TAMI checkpoint trained to distinguish confirmed fraud from legitimate
payments. Training happens offline through `experiment.py`; changing the comparison
dataset runs inference with the saved weights.

The default training configuration fine-tunes the temporal encoder and a small,
swappable fraud classifier using fraud labels. A `frozen` encoder option trains
only the classifier, providing a cheaper baseline and a way to measure the benefit
of fine-tuning. Both options belong to the initial implementation.

Scenario, size, seed, timing, ordinary dataset import, prediction-head selection,
shared or individual policies, and shadow or enforce mode use the existing flow.
The public comparison model remains `dyg_tami_native`.

## Starting point

`models/implementations/dyg_tami.py` currently trains the real temporal network on
observed links versus sampled alternative destinations. Its binary targets mean
“observed link,” not “fraud.” `prediction_heads/implementations/` converts these
link logits into suspicion scores; those heads do not learn fraud classification.

`TemporalGraphDataset` deliberately contains no outcome labels. Payment JSON and
CSV loaders retain outcomes separately in `EventDataset.document.truth`, which
`payment_graph.py` omits when creating link-training inputs.

The shipped historical artifact cannot provide the required supervised example:

| Current partition | Payments | Fraud labels |
| --- | ---: | ---: |
| Training | 472 | 0 |
| Validation | 189 | 0 |
| Test | 1229 | 5 |

These counts come from `models/native-dyg-tami/manifest.json` and
`models/native-comparison-history.json`. Build a new labeled training fixture;
the existing test outcomes must not be repurposed as its training set.

## 1. Separate the training task from the model and data format

Add an explicit experiment task, `temporal-fraud-classification`, alongside
`dynamic-link-prediction`. Extend model task registration so `dyg_tami_native`
can resolve to a dedicated implementation for each task. Existing configurations
without a task continue to resolve exactly as they do today.

The supervised implementation lives in
`models/implementations/dyg_tami_fraud.py`. It composes the existing DyGFormer,
TAMI time encoder, pair-memory mechanisms, and a registered fraud head. Task
dispatch belongs in the framework; dataset parsing and head selection must not
be embedded in the model implementation.

Add a `LabeledTemporalGraphDataset` task view in `framework/contracts.py`:

- The existing graph, preserving payment IDs, chronological order, and features.
- Aligned outcomes: `1` confirmed fraud, `0` confirmed legitimate, `-1` unknown.
- Optional label-availability times and explicit outcome provenance.

Build this view in `framework/temporal_fraud_data.py` from ordinary payment JSON
or mapped CSV providers. Extract reusable event-to-graph conversion from
`datasets/implementations/payment_graph.py`; keep its current unlabeled output
compatible. No additional comparison dataset loader is needed.

Outcomes and their availability times are targets and metadata only. Never pass
them into encoder features. Unknown payments still supply temporal context but
contribute no supervised loss. An absent fraud report does not establish a
legitimate label.

For datasets with outcome timestamps, enforce the declared label-availability
cutoff for each fitting stage. Add optional JSON outcome metadata and a mapped
CSV column without changing the meaning of existing `truth` values. If a dataset
provides only final labels, record a retrospective-label assumption explicitly;
do not claim to simulate delayed label availability.

## 2. Expose causal transaction representations and swappable heads

The first learned head is `fraud_linear`, implemented in
`prediction_heads/implementations/fraud_linear.py` as a trainable PyTorch linear
layer producing one fraud logit. This provides a small, inspectable baseline;
another registered head can later use an MLP without changing the experiment or UI.

Use these inputs, computed before any current-timestamp history updates:

| Input | Purpose |
| --- | --- |
| TAMI current interaction embedding, `ReLU(fc1([z_source, z_destination]))` | Learned representation of the proposed pair and its past context |
| Previous directed-pair memory | How this interaction relates to earlier interactions between the pair |
| Candidate `log1p(amount)` | Lets the classifier inspect the amount of the payment being decided |

The existing native link score does not directly inspect its candidate amount.
The supervised head should do so. Keep this feature separate from historical
edge features until the payment is committed. Fit its normalization on training
payments only and save the exact feature order, units, and normalization state.

Introduce a typed representation result containing head inputs and proposed memory
updates. Scoring remains pure; a separate commit applies the allowed history
updates. Reuse the existing TAMI equations through composition, retaining the
same-timestamp behavior and detached pair memory. The link decoder's final scalar
layer is not needed by the fraud classifier.

Extend the head registry with explicit input and output contracts. Existing
`empirical_tail` and `fixed_likelihood` heads accept link logits; `fraud_linear`
accepts transaction representations and owns trainable parameters. Do not overload
the current `fit(reference_logits)` interface to mean supervised neural training.
The trainer owns optimization; the learned-head contract supplies construction,
forward scoring, parameters, and safe state serialization.

Use the same score direction and units as existing supervised comparison models:

```text
fraud_logit = head(transaction_representation)
p_fraud = sigmoid(fraud_logit)
score = -log2(1 - p_fraud) = softplus(fraud_logit) / ln(2)
BLOCK when score > policy_threshold
```

Compute the score with the stable softplus expression. A manual threshold of
`1` means blocking estimated fraud probabilities above `0.5`; shared alpha
continues to mean a blocking target, not a fraud-probability cutoff. The output
is an estimated probability, with no claim of probability calibration.

## 3. Train on confirmed payment outcomes

Implement a dedicated runner in `framework/temporal_fraud_experiments.py` and a
supervised temporal model contract with `fit_fraud` and `predict_fraud` methods.
The runner must not reuse the link evaluator's sampled negatives or link metrics.

Training procedure:

1. Create chronological partitions and validate label coverage before allocating
   a training run. Fit feature normalization from the training partition.
2. Initialize with link pretraining restricted to the permitted training history,
   or load a compatible checkpoint with verified training/selection provenance.
   A checkpoint exposed to policy-validation or test data is not eligible for an
   independent experiment, even if that exposure was unlabeled.
3. Attach a new fraud head. In `finetune` mode, optimize the head, DyGFormer, LTE,
   and the current-interaction TAMI projection. In `frozen` mode, keep all encoder
   weights fixed and the encoder in evaluation mode, including dropout.
4. Replay training payments chronologically. Score a complete timestamp group
   against strictly earlier history, apply BCE-with-logits to known fraud labels,
   and then commit the group's observed interactions. Accumulate gradients without
   shuffling history or splitting the group's history updates.
5. Keep pair memories detached, as in the existing implementation. Fine-tuning
   propagates through the current computation, without backpropagating through
   the entire stream. Unknown-only groups advance context without optimizer steps.
6. At each epoch boundary, reset state and replay the prefix with fixed epoch
   weights before validation. Select the epoch by supervised validation log loss.
7. Restore the best weights, fit policy cutoffs on their reserved history, and
   evaluate the final test partition once.

Use ordinary binary cross-entropy initially. Offer an explicit class-weight
option computed solely from known training labels; record it in the artifact.
Do not rebalance validation or test data. Training requires both known classes
and must fail with counts and an actionable explanation if either is missing.

Historical training uses observed transaction attempts, including fraudulent and
unknown payments, consistently with current shadow semantics. Ground truth must
never decide which payments enter graph history. During enforced inference,
the actual policy decisions determine graph and pair-memory updates.

## 4. Reserve independent periods for selection and evaluation

Use four ordered windows, with entire timestamp groups kept together. Start with
60% / 15% / 10% / 15% of distinct timestamps; allow explicit boundaries for real
datasets. These fractions are configuration defaults, not guarantees of class
coverage.

| Window | Permitted uses |
| --- | --- |
| Train | Link pretraining, fraud gradients, normalization, class weights |
| Model validation | Epoch and declared model-configuration selection |
| Policy validation | Historical cost/F1/F2/balanced-accuracy cutoff selection |
| Test | Final frozen-model evaluation only |

Apply label-availability cutoffs as well as transaction-time boundaries. Prefix
replay supplies causal context, not additional gradients. Reset state between
independent runs; never carry a training memory snapshot into an unrelated scenario.

The example fixture must have fraud and legitimate examples in every reserved
window. For user datasets, require both classes in training and model validation;
disable tuned policies when their validation window lacks coverage. Undefined
test metrics should be reported as unavailable with class counts.

Create a reproducible history containing multiple fraud episodes, legitimate
traffic, and independent reserved periods, plus separate evaluation scenarios.
Keep generation in a dataset implementation/configuration and mark its synthetic
origin. Include a mapped real-payment CSV example using the same task contract;
no real labeled dataset has been supplied yet. Do not equate fixture performance
with performance on real payments or repeatedly choose seeds to improve test results.

## 5. Save complete artifacts and expose capabilities

Save supervised artifacts separately, for example `models/native-dyg-tami-fraud/`.
Use a versioned manifest with the task, public model ID, head ID and contract,
encoder training mode, architecture, all tensor checksums, implementation hashes,
feature schema/normalization, initialization lineage, label semantics/availability,
partition IDs and time bounds, per-class counts, training history, selected epoch,
and threshold provenance. Store weights as safe NPZ arrays, as the native model
does today; loading must not silently initialize missing head weights.

Add explicit experiment task dispatch to `framework/experiments.py` and the
existing `experiment.py train` and `evaluate` commands. A proposed configuration
shape is:

```json
{
  "model": "dyg_tami_native",
  "task": "temporal-fraud-classification",
  "dataset": {"loader": "payment_json", "path": "labeled-history.json"},
  "graph": {"node_feature_dim": 32},
  "initialization": {"method": "link_pretrain_train_partition"},
  "encoder_training": "finetune",
  "prediction_head": {"id": "fraud_linear"},
  "split": {"train": 0.60, "model_validation": 0.15, "policy_validation": 0.10},
  "parameters": {"epochs": 10, "patience": 3, "seed": 42}
}
```

These options are supported. The shipped example uses explicit whole-episode
boundaries because the generic fractions do not guarantee fraud-label coverage.

Extend local service configuration to map training modes to artifacts. Discover
mode/head/policy capabilities from successfully loaded artifacts, including
per-mode unavailable reasons. A missing supervised checkpoint must not disable
the unsupervised checkpoint or silently fall back to it.

Keep the existing unsupervised artifacts, configuration defaults, link-training
task, and batch `export-fraud` behavior compatible. Prefer new composition modules
over changing hashed native implementation sources. If a shared source must
change, explicitly regenerate affected artifacts and validate parity; never
bypass the current source-integrity checks.

## 6. Integrate with the existing comparison

Update `framework/native_comparison.py` and `framework/comparison_service.py` to
select the artifact and head by `trainingMode` using their common scoring and
history interfaces. Route both tasks through the same request, cancellation,
policy, and settlement machinery.

`GET /api/models` advertises the valid heads and policies for each training mode.
The existing model/head controls in `tools/comparison/ui.js` consume those
capabilities. Persist head and policy settings per model and training mode, and
choose valid defaults when switching between them.

Supervised results include explicit `fraud_logit`, `fraud_probability`, head ID,
and `native-fraud` evidence. Do not label them link likelihood or tail probability.
Version the evidence contract and update the native result adapter, inspector,
exports, and training-source text, which currently assumes a logistic head on a
fixed encoder for non-XGBoost supervised models.

Shared, manual, cost-tuned, and automatic policies consume the same monotone
score. Derive supervised policy frontiers from its own reserved history. Preserve
the existing observed-history interpretation of historical threshold tuning;
show enforce-mode performance separately because blocks change later histories.

Bind caches to artifact/weights, feature contract, head, training mode, dataset,
and effective history/policy settings. Preserve cancellation between timestamp
groups, common evaluation populations, and outcome-overlap checks. Comparison
labels are only for reporting; flipping them must never retrain weights, refit
cutoffs, or change predictions.

## Delivery order and acceptance

| Step | Main files | Completion evidence |
| --- | --- | --- |
| 1. Data and task contracts | `framework/contracts.py`, `framework/temporal_fraud_data.py`, dataset adapters, model registry | Label masks, availability, and four chronological windows validated through ordinary JSON/CSV inputs |
| 2. Representation and head | `models/implementations/dyg_tami_fraud.py`, `prediction_heads/implementations/fraud_linear.py`, head registry | Pure causal scoring; finite, correctly oriented fraud scores; head state round-trip |
| 3. Supervised experiments | `framework/temporal_fraud_experiments.py`, task dispatch, examples | Real Torch training in frozen and fine-tuned modes; complete independent test artifact |
| 4. Service and comparison | Native service/scorer, native adapter, existing comparison controls | Supervised native model works across all scenarios, normal imports, all policies, and both history modes |
| 5. Reproducible delivery | Labeled fixture configuration, trained artifact, README, generated pages | Documented training command reproduces the mode; existing unsupervised workflows remain usable |

Required checks should verify behavior, not just configuration flags:

- Fraud labels change learned weights. Gradients and parameter changes reach the
  encoder in fine-tuning mode; encoder tensors stay identical in frozen mode.
- Unknown labels produce no loss, explicit legitimate examples do, and class
  counts agree with the saved manifests. No negative-sampled links become fraud targets.
- Future events and future labels cannot affect an earlier prediction under
  fixed weights. Equal-time groups share prior state. Current amounts can affect
  the supervised score before settlement; outcome fields cannot.
- Model/policy validation and test targets never enter the training loss or
  preprocessing. Only the designated validation labels may affect selection;
  test-label flips affect metrics alone and leave the fitted artifact unchanged.
- Reloaded artifacts reproduce logits and decisions. Incompatible head inputs,
  source hashes, normalization, or initialization lineage are rejected clearly.
- Real-service browser checks cover ordinary settings, multiple scenarios/sizes,
  JSON imports, head/mode switching, rapid cancellation, shadow/enforce histories,
  and missing-checkpoint behavior. Existing native and browser tests still pass.

Report fraud average precision (with its metric definition), ROC AUC where
defined, log loss, precision/recall/F1, confusion counts, legitimate false-block
rate, and known/unknown label counts on the final test window. Report the selected
threshold and its validation source alongside these metrics. Compare frozen and
fine-tuned variants using the same reserved data and evaluation population.
