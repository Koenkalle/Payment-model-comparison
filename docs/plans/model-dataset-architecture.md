Model and dataset architecture

Implement a registry-driven framework with inspectable model modules, separate
checkpoint artifacts, independent dataset loaders, and explicit execution
capabilities. Existing checkpoints and numerical behavior are migration inputs;
prototype implementations must be labeled as prototypes.

1. Model implementations

   `models/registry.json` identifies each model's Python implementation, browser
   implementation when available, checkpoint, supported input schema, training
   behavior and maturity. `models/implementations/<model>.py` and `.js` are the
   inspection entry points. Shared neural/history primitives live in explicitly
   named support modules in that directory. Registration and experiment code
   contain no model-family dispatch or model mathematics.

   Move existing Python training and JavaScript inference out of their monolithic
   files. Preserve compatibility imports and model IDs for saved fixtures. Mark
   the four temporal-history/pair variants as fixed-encoder prototypes and the
   existing tree implementation as a custom NumPy booster. Add separate official
   library implementations for XGBoost and logistic regression, with lazy optional
   dependencies and explicit Python-only capability declarations.

2. Dataset implementations

   `datasets/registry.json` identifies independent loaders. Support numeric CSV
   feature tables, CSV payment events, normalized payment-event JSON, and the
   existing synthetic generator through separate files. Dataset configurations
   map source columns explicitly. New fitted-model plugins never open dataset files or
   assumes a generator, file path, label-column name or split.

   Tabular data has ordered feature names, numeric rows, event IDs, timestamps,
   optional outcomes and provenance. Event data has stable external account IDs
   mapped to internal indices, chronological payment/deposit/report events,
   separate evaluation outcomes, time units and provenance. Reject duplicate IDs,
   invalid timestamps, nonfinite inputs, invalid outcomes and ambiguous schemas.
   Unknown outcomes remain unknown. Never infer missing account identities or
   label missing outcomes as legitimate payments.

3. Framework contracts and experiment runner

   `framework/contracts.py` defines dataset and fitted-model interfaces.
   `framework/registry.py` resolves implementations from manifests and checks
   capabilities. `framework/experiments.py` owns chronological train/validation/
   test partitioning, training on labeled training rows only, validation-only
   threshold selection, held-out evaluation, and versioned artifact metadata.
   A model implements fit, predict, save and load; explanations are optional.

   The real-model runner accepts numeric tabular datasets. Payment-event datasets
   are supported by browser replay models; requesting an incompatible combination
   must fail explicitly. This is the extension boundary for later temporal model
   training and feature pipelines, not an implicit conversion to a placeholder.
   Native Python models are not silently substituted into browser adapters.

   Store native model files separately from experiment metadata and reports.
   Metadata includes implementation ID, feature order, source/configuration hashes,
   split membership, parameters and fitted cutoff. Loading validates the artifact
   and feature schema. Never deserialize arbitrary pickle payloads.

4. Tools and user flow

   Build browser model scripts from the model manifest. Both tools obtain datasets
   through a dataset registry. Add local payment JSON import to comparison and
   analytics; provide a CLI to normalize mapped event CSV into that JSON format.
   Imported data is not uploaded. Display its identity and provenance, and clearly
   distinguish synthetic-trained demo checkpoints from a model trained for the
   imported dataset.

   A new experiment CLI lists capabilities, prepares event data, trains a selected
   real model on a selected dataset configuration, and evaluates a saved artifact.
   Supply copyable configuration examples, not a fabricated claim of evaluation
   on real customer data. Existing offline pages remain self-contained.

5. Verification

   Compare extracted browser inference against existing Python parity fixtures
   across all model modes. Keep the existing comparison and analytics tests.
   Add dataset validation, label isolation, deterministic timestamp-safe splits,
   plugin isolation and unsupported-combination tests. Train both native-library
   models against a local CSV fixture, save/reload them, verify predictions and
   XGBoost attribution additivity, and check feature-order rejection. Exercise
   local dataset import in the actual offline pages.

The official XGBoost plugin uses native DMatrix/Booster training, JSON model
persistence and native contribution output as documented in the
[XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html).

Full DyGFormer/TAMI research implementations, graph minibatching, categorical
feature pipelines, remote storage, distributed training and production data
validation are follow-on implementations behind these contracts. They will not
be represented by the existing fixed-encoder prototypes.

Implementation status: completed for this scope. The two native-library plugins,
four dataset providers, registry-based browser build, compatibility extraction,
CLI and local browser import are implemented. Native save/reload and attribution
tests, legacy parity checks and offline browser import/layout checks pass.
Legacy synthetic training retains its old historical-file conventions behind
`train.py`; it is not presented as a native experiment plugin.
