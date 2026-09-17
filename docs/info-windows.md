# Parameter and feature information

Hover over a parameter or feature to read its extended information. Keyboard
focus also opens help; the info button pins a window open on click or touch.
The window can be scrolled, stays inside the viewport, and closes with Escape,
its close button, or an outside click. Opening help does not change a field,
toggle a feature, or open its separate statistics disclosure.

The visible info symbol is 14 pixels across, with a larger tap target on touch
screens. Mouse help opens 32 pixels to the right of the cursor when space allows,
leaving nearby labels visible. It stays in place while being read or scrolled,
uses the left side near the right screen edge, and uses space above or below
on narrow screens. Keyboard and touch help are anchored to the relevant field.

The same component is used in Data lab, Model trainer, comparison, and XGBoost
analytics. It covers editable settings, selected and unavailable features,
record-preview column headers, saved-run configuration, and feature explanations.
Parameter help shows behavior, tradeoffs, defaults and supported choices.
Feature help shows interpretation, formulas, units, history, approximation
limits, and available summary statistics. Full distributions remain available
in Data lab's expanded feature view.
The introduction describes the individual measure, including its endpoint and
payment direction. Saved custom definitions keep their supplied meaning;
unknown columns identify the missing source definition explicitly.

## Where descriptions belong

- **Model parameters:** `framework/pipeline_training.py` declares `MODEL_SETTINGS`.
  Every `_field` requires an `info` argument. Keep behavior and tuning guidance
  there; defaults, limits and choices are read from the same fields used by
  server validation. `default_label` explains special values such as automatic
  threshold selection or unweighted classes without changing the stored value.
- **Feature calculations:** recipe metadata in `datasets/features.py`,
  `datasets/payment_features.py` and `datasets/graph_features.py` remains
  authoritative for formulas and history. Presentation notes live in
  `datasets/feature_info.py`. A new recipe can supply its own `info` object to
  describe interpretation and limitations without changing frontend code.
- **Replay and analytics controls:** `shared/runtime/info-metadata.js` owns
  shared UI explanations, while model-specific trainer settings come from
  the server's model catalog. Checkpoint feature explanations use the
  checkpoint's original feature definitions.

Help is enriched at read time. Saved datasets keep their pinned recipe manifest,
values and fingerprint; help reads do not recalculate features or load feature
arrays. Source columns receive descriptions without calculating derived inputs.
Unknown source semantics are stated explicitly rather than guessed.

## Reusing the component

`shared/ui/info-windows.js` provides `PaymentInfo.attach(element, info, options)`.
The information object contains `title`, `description`, optional `sections`
(`{title, text}`), and `facts` (`[label, value]` pairs). All content is rendered
as text. `PaymentInfo.parameter(field)` and `PaymentInfo.feature(definition)`
build these objects from metadata and merge an optional `info` override.

Use `buttonParent` to place an icon beside a label while keeping the entire
field as the hover/focus target. For an existing interactive feature name or
SVG mark, use `button: false`. Attach accepts a resolver function for changing
values. Reattaching updates content and restores the existing icon; discarded
elements need no manual cleanup. `PaymentPipelineUI.featureList` and
`parameterList` render saved selections using the same definitions.

The component is bundled before tool scripts in every standalone page and
coexists with the brief tooltips used for other visual marks. Rebuild all pages
with `python build.py` after changing shared UI sources.

Validation covers the complete parameter catalog and feature collection,
immutable saved definitions, real-service data/training workflows, keyboard
and touch interaction, control isolation, safe text rendering, responsive
placement, and comparison/analytics help surfaces.
