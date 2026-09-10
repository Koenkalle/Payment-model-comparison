# DyGFormer + TAMI integration

Use TAMI's published DyGFormer combination: the four-channel patch transformer
and neighbor co-occurrence encoder, logarithmic time encoding (LTE), and the
directed-pair historical decoder (TRC). Train encoder and decoder end to end for
dynamic link prediction with observed interactions and sampled negative links.

- Put the actual backbone in `models/implementations/dygformer.py`, LTE/TRC in
  `tami.py`, and the fitted model in `dyg_tami.py`. Move the old inspection entries
  to explicitly named `*_prototype.py` files, retaining the existing browser IDs
  and checkpoints. Register the real model separately as `dyg_tami_native`.
- Pin and attribute the MIT-licensed upstream implementations. Record local
  adaptations, including chronological memory handling and safe persistence.
- Add a temporal graph dataset contract and independent CSV provider, including
  DyGLib's preprocessed CSV/NumPy feature format and a payment-event bridge.
- Extend the experiment CLI by input schema, with chronological splits, seeded
  negative sampling, validation-only model selection and a final test report.
  Link-existence scores must never be labeled fraud probabilities.
- Score all interactions at a timestamp before updating pair memory. Histories
  contain strictly earlier interactions. Negative examples never enter memory;
  exclude every observed positive at the same timestamp from negative sampling.
- Verify upstream numerical parity, end-to-end gradients and parameter updates,
  chronological and outcome isolation, directed/repeated-pair memory, equal-time
  behavior, unseen nodes, and saved-model prediction parity on local fixtures.
- Provide runnable examples and documentation. Native PyTorch models run in
  Python; the existing offline browser prototypes keep explicit prototype labels.

Pinned sources:

- TAMI: `b64bfa53731c623d6fa46e44ebb5aa8b09d90d1e`
  <https://github.com/Alleinx/TAMI_temporal_graph>
- DyGLib: `3aacc36b94b8d2d8293d70a74fdf6d39089b4163`
  <https://github.com/yule-BUAA/DyGLib>

This implements the published architecture and an inspectable local experiment
workflow. Benchmark score reproduction requires the authors' datasets, splits,
negative-sampling protocol and hyperparameters; fixture results are not paper
benchmark results.

Implementation completed: the native registry entry, temporal graph providers,
chronological runner, safe checkpoints, examples and provenance are in place.
Seven integration/state/data tests and the independent upstream reference test
pass. The reference compares embeddings, logits and gradients. Legacy model
parity, native tabular model tests and offline build checks also pass.

A local CPU run on the existing 912-payment takeover fixture completed ten
epochs. Validation selected epoch nine, using 547 training links, 182 validation
links and 183 test links. The held-out report has AP 0.87319 and ROC AUC 0.87312
against one sampled negative per positive. These are synthetic link-prediction
results under the documented local protocol, not fraud or paper benchmark results.
