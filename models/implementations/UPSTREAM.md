# DyGFormer + TAMI source provenance

`dygformer.py` is the complete DyGFormer backbone from
[TAMI](https://github.com/Alleinx/TAMI_temporal_graph/tree/b64bfa53731c623d6fa46e44ebb5aa8b09d90d1e),
commit `b64bfa53731c623d6fa46e44ebb5aa8b09d90d1e`, derived from
[DyGLib](https://github.com/yule-BUAA/DyGLib/tree/3aacc36b94b8d2d8293d70a74fdf6d39089b4163).
Both repositories use MIT licenses, retained in `licenses/`.

The backbone retains all four channels (node features, edge features, learned
time encodings, learned neighbor co-occurrence encodings), patch projections,
joint source/destination transformer attention and output projections. The
TAMI variant uses a trainable logarithmic time encoder.

`tami.py` adapts `TimeEncoder`, `HistEmbAggregatorWeightedSum`, `HistoricalDecoder`
and `TRCMemory` from TAMI's `models/modules.py`. It preserves the LTE and TRC
equations and trainable parameter names. The current interaction embedding is
concatenated with the **previous** directed-pair memory for scoring. Only after
scoring is memory updated as `gamma * current + (1 - gamma) * previous`, with
the stored tensor detached from the gradient graph, as upstream.

Local adaptations:

- Package imports point to isolated modules in this repository.
- DyGFormer timestamp padding uses float64; loaders express times as elapsed
  seconds. This prevents loss of subsecond information around large Unix times.
- Histories use strictly earlier timestamps. The sampler bounds retrieval to the
  most recent `max_input_sequence_length - 1` records before upstream patching.
- The decoder scores positives and negatives before committing any same-time
  positives. Duplicate directed pairs at one timestamp average their proposed
  updates. Upstream uses minibatch updates, whose repeated-key result depends on
  row order; its training and evaluation also differ in score/update ordering.
- Training accumulates gradients over timestamp groups. Each group sees the
  chronological past, including earlier groups in the same optimizer batch.
- Negatives replace the destination using destinations observed by the query
  time, excluding self-links and every same-time observed positive for that
  source. Upstream samples from a global destination pool and can draw positives.
  A positive with no valid negative still enters history but contributes no
  paired loss or metric. The report gives the excluded count.
- Each evaluation resets memories and replays earlier observations with frozen
  weights. This avoids carrying training-time embeddings or test state into
  subsequent evaluations. Validation loss chooses the epoch; test loss does not.
- Checkpoints use NumPy tensor arrays plus JSON metadata (`allow_pickle=False`),
  including the pair-memory snapshot, instead of Python/dill pickle. Evaluation
  reconstructs history, so the snapshot cannot accidentally leak future state.

These are explicit experiment-protocol adaptations. Default dimensions and
training duration are modest local defaults, not the authors' per-dataset best
configurations. Link prediction evaluates observed links against sampled links;
it is not supervised fraud detection, calibrated probability estimation, or a
claim to reproduce the papers' benchmark scores.

The `*_prototype.py` entries preserve old browser checkpoint compatibility.
They are separate from this native implementation and remain labeled prototypes.
