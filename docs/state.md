# NHSMM State

Current package readiness and next package-level research boundary. Detailed semantics: [`model.md`](model.md). Controlled evidence: [`package-validation.md`](package-validation.md). Verification policy: [`testing.md`](testing.md).

## Current status

**Phase:** duration-context, latent-state, transition-context, and multi-component identifiability are verified. **Multi-component identifiability is now PACKAGE-CORE PASS across the full frozen 15-seed A/B/C/D gate.** General-context robustness remains **provisionally accepted at independent-reference level** using the frozen detect→refine mechanism; package confirmation of that mechanism is now the active research boundary.

Package-core confirmation uses a SHA-pinned sparse local workspace reconstructed from `develop` sources. The exercised paths are canonical training, causal likelihood recursion, duration/transition/emission distributions, and causal filtering. External context is supplied explicitly, so encoder/import glue is inert for these validators. No CI is used for this research loop.

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

### General-context robustness — current boundary

Independent-reference work established:

- scalar strong/moderate/null controlled recovery acceptable;
- ungated 2D null spurious context modulation reproduced;
- global L2 rescue rejected;
- shared-duration/tied-context rescue rejected;
- independent full-vs-null evidence gates rejected;
- **detect→refine paired cross-fit gate: PROVISIONAL PASS**;
- confirmatory seeds `731..735`: strong selected `5/5`, moderate `5/5`, null `0/5`;
- strong median corr `0.960`, MAE `0.075`;
- moderate median corr `0.836`, MAE `0.074`;
- null context amplitude `0.000`, median MAE `0.034`.

**Status:** PROVISIONAL PASS at independent-reference level. Package-level confirmation remains outstanding.

## Verification limits

The full discovered pytest suite was not rerun because the local workspace is sparse. No GitHub Actions run is used for this research loop. Package-core evidence is scoped to the exact external-context dependency closure exercised by the validators.

## Package boundary

These controlled synthetic results establish mechanism recovery and identifiability. They do not establish domain semantics or downstream predictive value for Nautilus or any other consumer.

## Next research slices

1. **General-context package confirmation:** translate the frozen detect→refine reference gate to the actual package core and run it locally without retuning thresholds.
2. **API/design translation:** only after package confirmation, decide whether evidence selection belongs inside package training, a validation utility, or consumer orchestration.
3. Change objectives or parameterizations only if a controlled package-level gate exposes a reproducible deficiency.
