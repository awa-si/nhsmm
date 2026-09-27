# NHSMM State

Current package readiness and next package-level research boundary. Detailed semantics: [`model.md`](model.md). Controlled evidence: [`package-validation.md`](package-validation.md). Verification policy: [`testing.md`](testing.md).

## Current status

**Phase:** duration-context, latent-state, transition-context, and multi-component identifiability are verified. **Multi-component identifiability is PACKAGE-CORE PASS across the full frozen 15-seed A/B/C/D gate.** General-context package confirmation remains **PARTIAL / NOT PASS**.

Package-core confirmation uses a SHA-pinned sparse local workspace reconstructed from `develop` sources. The exercised paths are canonical training, causal likelihood recursion, duration/transition/emission distributions, transition-only refinement, and causal filtering. External context is supplied explicitly, so encoder/import glue is inert for these validators. No CI is used for this research loop.

## Accepted evidence

### Duration context — seeds 201..215

Strong, moderate, and null controlled duration-context gates: **PASS**.

### Latent state — seeds 301..315

Strong and moderate state recovery plus null descriptive control: **PASS**.

### Transition context — seeds 401..415

Binary external-context transition recovery at `max_duration=1`: **PASS** for strong, moderate, and null.

### Multi-component identifiability — PACKAGE-CORE PASS

Package source pinned for this confirmation: `efd4826ddc48586c47debf075d1c6332596cb0de`; frozen seeds `501..515`; `K=3`, `D=24`, ergodic transition support.

| Scenario | A | B | C | D |
|---|---|---|---|---|
| strong | PASS | PASS | PASS | PASS |
| moderate | PASS | PASS | PASS | PASS |
| null | PASS | PASS | PASS | PASS |

Representative medians:
- strong C: transition MAE `0.184`, corr `0.996`, noncollapsed `15/15`;
- strong D: duration gap `4.383`, transition MAE `0.169`, corr `0.997`, noncollapsed `15/15`;
- moderate C: transition MAE `0.162`, corr `0.963`, noncollapsed `15/15`;
- moderate D: duration gap `2.273`, transition MAE `0.132`, corr `0.938`, noncollapsed `15/15`;
- null C: null transition delta `0.044`, noncollapsed `13/15`;
- null D: null transition delta `0.040`, noncollapsed `13/15`.

See [`validation/multicomponent-component-matched-package-15seed-2026-09-27.md`](validation/multicomponent-component-matched-package-15seed-2026-09-27.md).

**Status: PACKAGE-CORE PASS.** The multi-component research question is closed unless a future package change invalidates the frozen gate.

## General-context robustness — active package boundary

Frozen reference spec SHA256: `632d120c60847ffbe180e5f62e0d1c3820a5332b2b69830b7c5f55a712426389`.

Canonical package detect→refine result on seeds `731..735`:

| Scenario | Selected | Accuracy | ARI | Transition corr | Transition MAE | Context amp | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| strong | 5/5 | 1.000 | 1.000 | 0.919 | 0.092 | 0.706 | PASS |
| moderate | 4/5 | 1.000 | 1.000 | 0.730 | 0.098 | 0.474 | FAIL |
| null | 0/5 | 1.000 | 1.000 | 0.000 | 0.018 | 0.000 | PASS |

The moderate package run passes selection count, MAE, and per-seed recovery, but misses only the frozen median-correlation gate (`0.730 < 0.800`). Topology-equivalent recovery scoring conditions package `ergodic` destinations on an actual state change; selection evidence remains canonical package likelihood.

### Rejected candidates

No frozen gate, threshold, or refinement hyperparameter was retuned after confirmatory runs.

- pre-vs-post refinement: moderate corr ~`0.754` before and ~`0.730` after; refinement is not the primary cause;
- no-self topology: rejected (`3/5` moderate selected, corr ~`0.600`);
- shared-duration context: rejected (moderate corr ~`0.644`, MAE ~`0.134`);
- shared-linear estimator alone: excellent active recovery but null selected `5/5`; rejected as detector;
- shared-linear + effect-floor (`>=0.25`) confirmatory `751..755`: strong `5/5`, moderate `3/5`, null `0/5`; **REJECTED**;
- fold-stability / effect-floor candidate confirmatory `771..775`: strong `5/5`, moderate `3/5`, null `0/5`; **REJECTED**;
- single-permutation evidence dev `781..785`: strong `5/5`, moderate `5/5`, null `3/5`; **REJECTED on development**;
- permutation evidence AND effect-floor confirmatory `791..795`: moderate `2/5`; **REJECTED early**;
- permutation-rank exact test (5 folds × 7 cyclic permutations, one-sided exact Binomial vs 50/50, `alpha=0.05`):
  - dev `801..805`: strong `5/5`, moderate `5/5`, null `0/5`;
  - confirmatory `811..815`: strong `5/5`, moderate `5/5`, median corr `0.890`, median MAE `0.082`, but null `1/5` false-positive (seed `813`, exact `p=0.0083`);
  - **REJECTED**. Do not tighten `alpha` after observing the confirmatory block.

See [`validation/general-context-detect-refine-package-core-confirmation-2026-09-27.md`](validation/general-context-detect-refine-package-core-confirmation-2026-09-27.md), [`validation/general-context-package-diagnostics-2026-09-27.md`](validation/general-context-package-diagnostics-2026-09-27.md), [`validation/general-context-shared-linear-effectfloor-confirm-2026-09-27.md`](validation/general-context-shared-linear-effectfloor-confirm-2026-09-27.md), and [`validation/general-context-detector-rapid-2026-09-27.md`](validation/general-context-detector-rapid-2026-09-27.md).

**Current interpretation:** estimator expressiveness is not the remaining blocker. The permutation-rank detector restored full moderate power and good shape fidelity but still missed strict null control by one confirmatory seed. The open problem is replication-calibrated null control.

**Status: PACKAGE-CORE PARTIAL / NOT PASS.**

## Verification limits

The full discovered pytest suite was not rerun because the local workspace is sparse. No GitHub Actions run is used for this research loop. Package-core evidence is scoped to the exact external-context dependency closure exercised by the validators.

## Package boundary

These controlled synthetic results establish mechanism recovery and identifiability. They do not establish domain semantics or downstream predictive value for Nautilus or any other consumer.

## Next research slice

**Split-fit direction replication** is the active candidate. Two independent fits on disjoint training halves must recover the same context-delta direction after state alignment. This adds an independent reproducibility condition rather than another magnitude threshold.

- development seeds: `821..825`;
- if viable, freeze once on that block;
- fresh confirmatory seeds: `831..835`;
- no result has been accepted yet;
- API/design translation remains deferred until the general-context package gate passes.
