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

### Diagnostics after the partial gate

No frozen gate, threshold, or refinement hyperparameter was retuned.

- pre-vs-post refinement: moderate corr is already ~`0.754` before refinement and ~`0.730` after; refinement is not the primary cause;
- no-self topology: rejected (`3/5` moderate selected, corr ~`0.600`);
- shared-duration context: rejected (moderate corr ~`0.644`, MAE ~`0.134`);
- reference-equivalent shared-linear estimator: strong `corr~0.985/MAE~0.039`, moderate `corr~0.877/MAE~0.084`, but when used as its own detector it selects null `5/5` and is rejected;
- dual-stage canonical-MLP detector → shared-linear estimator on new dev seeds `741..745`:
  - strong: selected `5/5`, median corr `0.984`, MAE `0.044`, 5/5 per-seed recovery PASS;
  - moderate: selected `5/5`, median corr `0.923`, MAE `0.076`, 5/5 per-seed recovery PASS;
  - null: selected `3/5`, median context amplitude `0.169`, median MAE `0.038`, only 2/5 null per-seed PASS;
  - **REJECTED** because detector null-generalization failed on the new dev seeds.

### Shared-linear + effect-floor candidate — fresh confirmatory FAIL

A second candidate used the shared-linear estimator plus a pre-refine functional-amplitude floor. The extra floor was chosen only on development seeds `741..745`, where `0.25` separated active moderate effects from null controls while preserving the existing held-out evidence threshold `delta_ll_per_boundary > -0.07`. The rule was then frozen and tested on fresh seeds `751..755`.

| Scenario | Selected | Accuracy | ARI | Transition corr | Transition MAE | Result |
|---|---:|---:|---:|---:|---:|---|
| strong | 5/5 | 1.000 | 1.000 | 0.960 | 0.061 | PASS |
| moderate | 3/5 | 1.000 | 1.000 | 0.814 | 0.103 | **FAIL** |
| null | 0/5 | 1.000 | 1.000 | 0.000 | 0.017 | PASS |

The candidate restores null rejection (`0/5`) and preserves excellent strong recovery, but moderate selection falls to `3/5`, below the frozen `>=4/5` requirement. It is therefore **REJECTED**. Do not lower the evidence threshold or the `0.25` effect floor after seeing the confirmatory set.

### Rapid-mode detector candidate — checkpoint only

The next candidate targets **detector stability/power**, not estimator shrinkage. Planned statistic: fold-direction agreement / variance-aware evidence over the existing shared-linear context estimator. The intent is to reject null fits whose learned transition-context directions are unstable across cross-fit folds while retaining genuine moderate effects.

- development seeds: `761..765`
- if a rule is chosen, it must be frozen on that development block only;
- confirmatory seeds: `771..775`
- no result has been accepted yet;
- no thresholds have been changed yet;
- this is a checkpoint only, not evidence of PASS.

See [`validation/general-context-detect-refine-package-core-confirmation-2026-09-27.md`](validation/general-context-detect-refine-package-core-confirmation-2026-09-27.md), [`validation/general-context-package-diagnostics-2026-09-27.md`](validation/general-context-package-diagnostics-2026-09-27.md), and [`validation/general-context-shared-linear-effectfloor-confirm-2026-09-27.md`](validation/general-context-shared-linear-effectfloor-confirm-2026-09-27.md).

**Current interpretation:** estimator expressiveness is not the remaining blocker; shared-linear estimation recovers active strong/moderate functional shape well. The unresolved deficiency is robust evidence-based detection/generalization under multidimensional context. Canonical detection can false-positive on null blocks; the effect-floor variant restores null control but loses moderate selection power.

**Status: PACKAGE-CORE PARTIAL / NOT PASS.**

## Verification limits

The full discovered pytest suite was not rerun because the local workspace is sparse. No GitHub Actions run is used for this research loop. Package-core evidence is scoped to the exact external-context dependency closure exercised by the validators.

## Package boundary

These controlled synthetic results establish mechanism recovery and identifiability. They do not establish domain semantics or downstream predictive value for Nautilus or any other consumer.

## Next research slices

1. **Detector stability / power:** run fold-direction-agreement / variance-aware evidence candidate on dev seeds `761..765`; if viable, freeze once and confirm on `771..775`.
2. Preserve strong detection, recover moderate `>=4/5`, and keep null at `0/5`.
3. Prefer mechanism-level evidence uncertainty/calibration over another global shrinkage or lower post-hoc threshold.
4. Keep the shared-linear estimator as diagnostic evidence only until detector stability is solved.
5. **API/design translation** remains deferred until the general-context package gate passes.
