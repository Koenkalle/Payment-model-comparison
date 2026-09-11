# Native fraud models in the shared comparison

Native models are ordinary entries in the model selector. Scenario, size, seed,
report delay, relay timing and imported payment JSON select inputs for every
compatible model. There is no native-specific dataset loader in the UI.

## Execution boundary

`python serve.py` serves the existing pages and a loopback model API. The page
discovers capabilities through `GET /api/models` and sends its current dataset
and effective model settings to `POST /api/compare`. Model code lives in
`framework/native_comparison.py`; HTTP handling and bounded request caches live
in `framework/comparison_service.py`. Clients supply event data, never paths or
executable code.

Versioned results contain each payment's logit, anomaly score, pre/post cutoff,
decision and settlement. The adapter binds them to the exact dataset and policy.
Changing settings requests updated inference; stale requests cannot replace the
current comparison. Per-page revisions and disconnects cancel obsolete Python
work between timestamp groups.

## Models, heads and policies

The shipped checkpoint is trained independently of selected scenarios. Inspect
its source, tensor checksum and historical dataset in `models/native-dyg-tami/`.
The historical fixture uses seed 271828: 472 payments for training, 189 for epoch
selection, 100 for rank references and 1129 for threshold validation. Moving the
repository preserves validity when source bytes and graph content still match.
Selecting a dataset never retrains the model.

Heads are independent files in `prediction_heads/implementations/`. Historical
rank is `(1 + count(reference <= logit)) / (N + 1)`; fixed likelihood is
`sigmoid(logit)`. Both output negative log2 as an anomaly score. Neither is a
calibrated fraud probability.

The normal shared policy learns a cutoff from unlabeled warm-up and tracks the
same alpha target as browser models. Individual fixed thresholds, historical
cost tuning and automatic F1/F2/balanced-accuracy objectives also apply. Historical
threshold validation follows rank calibration and does not consume the selected
dataset's labels. Head selection lives beside the main model selector; settings
persist independently per model.

## History and evaluation

Shadow inference uses observed payment attempts. Enforced blocking updates the
native neighbor history and directed-pair memory only for allowed payments. All
same-timestamp queries are scored before their graph updates. Changing blocking
policy therefore changes subsequent native scores where appropriate.

Metrics share a common payment population, excluding every model's warm-up and
context. Unknown outcomes are excluded from classification. Source-overlap checks
avoid treating training/calibration transactions as independent evaluation.

The link-likelihood score uses past transactions and proposed endpoints/time.
The supervised `fraud_linear` head also uses the candidate amount. Deposits and
reports are not native inputs. The supervised checkpoint trains on confirmed
fraud and legitimate outcomes, with either fine-tuned or frozen encoder weights.
Both training modes use these same comparison controls.

The supervised architecture and implementation stages are described in
[Fraud-label training for DyGFormer + TAMI](dyg-tami-fraud-supervised.md), including
encoder fine-tuning, replaceable fraud heads, and integration through these same
comparison controls.

## Verification and delivery

Tests cover genuine Torch scoring, head/policy parity, separate calibration,
relocation, outcome/future isolation, blocking histories, cancellation, bounded
caches, all 30 generated scenario/size/seed combinations and stored dataset
validity. Browser tests exercise ordinary settings and imports against the
actual local Python service, plus offline browser-only operation.

HTML remains self-contained for browser models. Full native inference requires
the local app; unavailable model entries explain that requirement. Batch
`export-fraud` remains a separate programmatic facility.
