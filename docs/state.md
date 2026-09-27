# NHSMM State

Current package readiness and next package-level research boundary. Detailed semantics: [`model.md`](model.md). Controlled evidence: [`package-validation.md`](package-validation.md). Verification policy: [`testing.md`](testing.md).

## Current status

**Phase:** duration-context, latent-state, transition-context, multi-component identifiability, and general-context robustness are verified at package-core level.

- **Multi-component identifiability: PACKAGE-CORE PASS** across the frozen 15-seed A/B/C/D gate.
- **General-context robustness: PACKAGE-CORE PASS** using split-fit direction replication.
- No CI is used for this research loop; confirmation was run locally against the SHA-pinned package core with explicit external context.
- The active boundary is now **API/design translation**, not further detector tuning.

## Accepted evidence

### Duration context — seeds 201..215

Strong, moderate, and null controlled duration-context gates: **PASS**.

### Latent state — seeds 301..315

Strong and moderate state recovery plus null descriptive control: **PASS**.

### Transition context — seeds 401..415

Binary external-context transition recovery at `max_duration=1`: **PASS** for strong, moderate, and null.

### Multi-component identifiability — PACKAGE-CORE PASS

Package source pinned for confirmation: `efd4826ddc48586c47debf075d1c6332596cb0de`; frozen seeds `501..515`; `K=3`, `D=24`, ergodic transition support.

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

## General-context robustness — PACKAGE-CORE PASS

The original package detect→refine gate on seeds `731..735` was only partial: strong and null passed, but moderate transition-shape correlation was `0.730 < 0.800`. A sequence of detector candidates was tested with strict dev→fresh-confirmatory separation and no post-hoc threshold relaxation.

Rejected candidates include:

- no-self topology;
- shared-duration context;
- shared-linear estimator as detector;
- shared-linear + effect-floor;
- fold-sign / fold-stability rules;
- single-permutation evidence;
- permutation evidence + effect-floor;
- permutation-rank/binomial detector;
- Fisher fold-permutation detector.

These remain rejected and must not be revived by retuning on their confirmatory blocks.

### Accepted detector: split-fit direction replication

The accepted detector requires the context effect to reproduce across **two independent fits on disjoint training halves**. Detection is truth-free: latent-state alignment uses learned emission centers only, then the learned context-induced transition-delta functions are compared on a fixed context grid.

Frozen on development seeds `841..845`:

- `replication_corr >= 0.55`;
- `min(amplitude_half1, amplitude_half2) >= 0.50`.

Fresh confirmatory seeds: `851..855`.

| Scenario | Selected | Accuracy | ARI | Transition corr | Transition MAE | Replication corr | Min half-amplitude | Result |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| strong | 5/5 | 1.000 | 1.000 | 0.976 | 0.051 | 0.933 | 0.800 | PASS |
| moderate | 4/5 | 1.000 | 1.000 | 0.914 | 0.071 | 0.771 | 0.516 | PASS |
| null | 0/5 | 1.000 | 1.000 | 0.000 | 0.028 | 0.426 | 0.363 | PASS |

The full frozen gate is satisfied: strong detection is preserved, moderate selection reaches the required `>=4/5`, null remains `0/5`, and selected active cases retain high functional transition recovery with perfect state recovery.

See [`validation/general-context-split-replication-confirm-2026-09-27.md`](validation/general-context-split-replication-confirm-2026-09-27.md) and the machine-readable companion JSON.

**Status: PACKAGE-CORE PASS.** The detector-stability research boundary is closed unless a future package change invalidates the frozen gate.

## Verification limits

The full discovered pytest suite was not rerun because the local workspace is sparse. No GitHub Actions run is used for this research loop. Package-core evidence is scoped to the exact external-context dependency closure exercised by the validators.

## Package boundary

These controlled synthetic results establish mechanism recovery and identifiability. They do not establish domain semantics or downstream predictive value for Nautilus or any other consumer.

## Next research slice

1. **API/design translation:** decide where split-fit evidence selection belongs: core training API, validation utility, or consumer orchestration.
2. Preserve the accepted semantics: independent split fits, emission-based state alignment, direction replication, and minimum replicated effect size.
3. Do not promote rejected detector variants or relax frozen gates.
4. After API placement is decided, add focused package tests for the accepted detector semantics and then run the local test suite relevant to the changed surface.
5. Downstream Nautilus integration remains a separate empirical validation boundary.
