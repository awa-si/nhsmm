# NHSMM State

Current package readiness and next package-level research boundary. Detailed semantics: [`model.md`](model.md). Controlled evidence: [`package-validation.md`](package-validation.md). Verification policy: [`testing.md`](testing.md).

## Current status

**Phase:** duration-context, latent-state, transition-context, and multi-component identifiability are verified. **Multi-component identifiability is PACKAGE-CORE PASS across the full frozen 15-seed A/B/C/D gate.** General-context detect→refine package confirmation is **PARTIAL / NOT PASS**: strong and null pass, while moderate-strength 2D transition-function recovery misses only the frozen median-correlation gate.

Package-core confirmation uses a SHA-pinned sparse local workspace reconstructed from `develop` sources. The exercised paths are canonical training, causal likelihood recursion, duration/transition/emission distributions, transition-only refinement, and causal filtering. External context is supplied explicitly, so encoder/import glue is inert for these validators. No CI is used for this research loop.

## Verified training baseline

- [x] `NHSMM.optimize()` covers all trainable parameters, including the causal encoder.
- [x] Default joint-training maximum is `40` iterations.
- [x] `n_init` uses independent restarts; best-run snapshots include `duration_logits_bias`.
- [x] Explicit `kmeans` emission initialization is available for state-identifiability-sensitive training.
- [x] Scalar external context retains a distribution hidden width >=16.
- [x] Transition context capacity is independently configurable.
- [x] Optional transition refinement updates only transition context modulation against exact sequence likelihood.

## Accepted / provisional evidence

### Duration context — seeds 201..215

Strong, moderate, and null controlled duration-context gates: **PASS**.

### Latent state — seeds 301..315

Strong and moderate state recovery plus null descriptive control: **PASS**.

### Transition context — seeds 401..415

Binary external-context transition recovery at `max_duration=1`: **PASS** for strong, moderate, and null.

### Multi-component identifiability — PACKAGE-CORE PASS

The initial failed joint benchmark is not accepted as evidence because it used a `semi` topology against cyclic ground truth and contained duration/transition scoring errors. A later isolated-C diagnostic was also invalid because duration-context ground truth remained active while duration context was disabled in the model. The corrected component-matched package gate nulls omitted mechanisms in both generator and model while leaving full-joint D unchanged.

Package source pinned for this confirmation: `efd4826ddc48586c47debf075d1c6332596cb0de`; frozen seeds `501..515`; `K=3`, `D=24`, ergodic transition support.

| Scenario | A | B | C | D |
|---|---|---|---|---|
| strong | PASS | PASS | PASS | PASS |
| moderate | PASS | PASS | PASS | PASS |
| null | PASS | PASS | PASS | PASS |

Representative medians:

- strong C: accuracy/ARI `1.000/1.000`, transition MAE `0.184`, corr `0.996`, noncollapsed `15/15`;
- strong D: accuracy/ARI `1.000/1.000`, duration gap `4.383`, transition MAE `0.169`, corr `0.997`, noncollapsed `15/15`;
- moderate C: transition MAE `0.162`, corr `0.963`, noncollapsed `15/15`;
- moderate D: duration gap `2.273`, transition MAE `0.132`, corr `0.938`, noncollapsed `15/15`;
- null C: null transition delta `0.044`, noncollapsed `13/15`;
- null D: null transition delta `0.040`, noncollapsed `13/15`.

See [`validation/multicomponent-component-matched-package-15seed-2026-09-27.md`](validation/multicomponent-component-matched-package-15seed-2026-09-27.md).

**Status: PACKAGE-CORE PASS.** The multi-component research question is closed unless a future package change invalidates the frozen gate.

### General-context robustness — package-core confirmation

Independent-reference work had provisionally accepted the frozen detect→refine mechanism on seeds `731..735` with selection threshold `-0.07` nats/validation-boundary and 20 transition-context-only refinement steps at lr `0.03`.

For package transfer, Initial/Duration/Emission context modulation is disabled so the package harness isolates the same transition-context question. The frozen reference generator has no self-boundary transitions, while package `ergodic` permits them; recovery metrics therefore condition learned destination probabilities on an actual state change (remove diagonal and renormalize off-diagonal). Selection evidence remains the canonical package likelihood without metric transformation.

Package source: `efd4826ddc48586c47debf075d1c6332596cb0de`. Frozen reference spec SHA256: `632d120c60847ffbe180e5f62e0d1c3820a5332b2b69830b7c5f55a712426389`.

| Scenario | Selected | Accuracy | ARI | Transition corr | Transition MAE | Context amp | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| strong | 5/5 | 1.000 | 1.000 | 0.919 | 0.092 | 0.706 | PASS |
| moderate | 4/5 | 1.000 | 1.000 | 0.730 | 0.098 | 0.474 | FAIL |
| null | 0/5 | 1.000 | 1.000 | 0.000 | 0.018 | 0.000 | PASS |

The moderate package run satisfies the frozen selection-count gate (`4/5`), the frozen median-MAE gate (`0.098 <= 0.100`), and the per-seed recovery requirement (`4/5` seeds). It fails only median transition-function correlation (`0.730 < 0.800`). The selected-only median correlation is also below `0.800`, so the result is not explained solely by the one unselected seed.

See [`validation/general-context-detect-refine-package-core-confirmation-2026-09-27.md`](validation/general-context-detect-refine-package-core-confirmation-2026-09-27.md).

**Status: PACKAGE-CORE PARTIAL / NOT PASS.** The open deficiency is narrowly localized to moderate-strength 2D transition-function shape fidelity. State recovery, null rejection, strong recovery, selection count, and moderate MAE all pass. No frozen gate, threshold, or refinement hyperparameter was changed.

## Verification limits

The full discovered pytest suite was not rerun because the local workspace is sparse. No GitHub Actions run is used for this research loop. Package-core evidence is scoped to the exact external-context dependency closure exercised by the validators.

## Package boundary

These controlled synthetic results establish mechanism recovery and identifiability. They do not establish domain semantics or downstream predictive value for Nautilus or any other consumer.

## Next research slices

1. **Moderate 2D context fidelity diagnostic:** without retuning the frozen gate, compare pre-vs-post refinement functional correlation, package context-network parameterization, and duration/self-boundary identifiability to locate the `0.730 < 0.800` gap.
2. Change package semantics only if a controlled diagnostic identifies a reproducible mechanism deficiency; any candidate fix must preserve strong and null gates.
3. **API/design translation** remains deferred until the general-context package gate passes; do not promote evidence selection blindly into core model semantics.
