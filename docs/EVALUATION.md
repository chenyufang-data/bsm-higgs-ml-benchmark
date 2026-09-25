# Evaluation methodology

How the benchmark in the [README](../README.md) is measured, and why. The same rules apply to every
model, so differences between methods come from the methods.

## Data and weights

- **Simulated events, weighted.** Every simulated event carries a *physical weight*: how many real
  events it represents at the target dataset size (production rate x luminosity / events generated).
  Metrics use physical weights. Training uses a separate *fit weight*, so physical normalization never
  leaks into what a model is asked to learn.
- **Tagging as a probability, not a coin flip.** The selection needs specific jet tags (two bottom-like,
  one charm-like). Applying simulated tag decisions keeps only a small fraction of events. Instead,
  every event is kept with the probability that it would pass the tags, an unbiased importance weight
  that yields about 9-19x more effective training and validation events. A closure test checked it against
  the direct decisions before it was adopted.
- **Data validation before modelling.** Event weights are reconciled with the event generator's own
  bookkeeping, events that two simulated samples both describe are counted once, and each step is
  checked by a gate that stops the pipeline when it fails.
- **Effective sample size.** With weights spanning orders of magnitude, the effective number of events
  `(sum w)^2 / sum w^2` is far smaller than the row count. It is reported for every selection.

## Splits

Events are assigned to training, validation and test (80/10/10) by a deterministic hash of their
identity, recorded once in a registry. Every model uses the same assignment; adding samples never
reshuffles existing events. The test split is not read by any study in this benchmark.

## Training and model selection

- Each model is trained with three seeds. Networks use their published configurations and one shared
  protocol (AdamW, cosine schedule, at most 60 epochs); nothing is tuned on the comparison metric.
- Early stopping and checkpoints use the fit-weighted validation AUC, the BDT's own early-stopping
  metric.

## Metrics

- **Weighted AUC** (physical weights): threshold-free ranking quality. Its seed-to-seed spread is small,
  so it is the primary comparison metric.
- **Expected significance** at an operating point, `Z = sqrt(2((S+B) ln(1+S/B) - S))` for expected
  signal S and background B, statistics only. The threshold is chosen on validation; an operating
  point must keep at least 100 effective background events and at least one simulated event of every
  background process, so it never rests on a handful of heavily weighted events. It is reported
  relative to the cut-based selection chosen the same way.
- **Background efficiency at fixed signal efficiency** and ROC curves, descriptive.

## Comparing models

- **Decision rules are fixed before training.** Each study's configuration records the question, the
  models, the metric and what counts as an improvement, and is committed before any model is trained.
- **Paired and resampled.** Model A is compared with model B of the same seed on the same validation
  events. A bootstrap (1,000 resamples of the validation events, the same resamples for every pair)
  gives a 95% interval for the seed-averaged AUC difference. An improvement needs a positive mean,
  an interval above zero and a positive difference for every seed.
- **Controls.** A Deep Sets network on the same inputs but without pairwise terms separates "a neural
  network" from "a network that models relations between jets".

## Interpretability

- **Permutation importance** on the twelve raw inputs shared by all models (momentum, direction and
  mass of each jet): the drop in AUC when one input is shuffled across events.
- **SHAP values** of the BDT's engineered features (tree explainer, class-balanced validation sample).

## Reproducibility and checks

Every study freezes its settings, archives the source code, hashes all inputs and outputs, and
re-verifies input and source hashes before it finishes. Reports check their own consistency: reloaded
models reproduce their saved scores exactly, features rebuilt from raw inputs reproduce the model
inputs, and values recomputed from saved scores reproduce earlier reports.

## Limitations

- Results are on validation data; the test split is reserved for one final evaluation.
- Significances are statistics-only and relative: no systematic uncertainties, no absolute values.
- The limited number of simulated background events caps the achievable operating points, most
  visibly at m = 400 GeV.
- Two benchmark hypotheses are shown.
