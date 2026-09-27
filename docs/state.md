# NHSMM State

Current package readiness and next package-level boundary. Detailed semantics: [`model.md`](model.md). Controlled evidence: [`package-validation.md`](package-validation.md). Verification policy: [`testing.md`](testing.md).

## Current status

**Phase:** package-core mechanism research and universal validation API translation are complete.

**Saved snapshot:** 2026-09-28 after normalized-public-API functional and stress validation.

- **Duration context:** PASS.
- **Latent-state recovery:** PASS.
- **Transition context:** PASS.
- **Multi-component identifiability:** PACKAGE-CORE PASS across the frozen 15-seed A/B/C/D gate.
- **General-context robustness:** PACKAGE-CORE PASS using split-fit direction replication.
- **Universal validation API:** initial, duration, emission, transition, and arbitrary state-indexed `[K,...]` context effects are supported without Nautilus/trading dependencies and without changing `NHSMM.optimize()` semantics.
- **Normalized public facade:** canonical `component=` API implemented for extraction, replica evaluation, and split-fit evidence; component-specific helpers remain behavior-compatible.
- **Focused validation/API tests:** 23/23 PASS locally in the sparse package workspace.
- **Normalized public API stress:** PASS — 4,800 effect extractions, 160 replication evaluations, and 16 real optimize-based split-fits across all four components without shape, finiteness, normalization, mode-restoration, determinism, or split-contract failures.
- No CI is used for this research loop.

The next package boundary is a broad local regression run from a complete checkout plus any documentation/export cleanup exposed by that run. Downstream Nautilus integration remains a separate empirical validation boundary.

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

### General-context robustness — PACKAGE-CORE PASS

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

Rejected detector variants remain rejected and must not be revived by retuning on their confirmatory blocks: no-self topology, shared-duration context, shared-linear as detector, effect-floor variants, fold-sign/stability rules, permutation variants, permutation-rank/binomial, and Fisher fold-permutation.

See [`validation/general-context-split-replication-confirm-2026-09-27.md`](validation/general-context-split-replication-confirm-2026-09-27.md) and the machine-readable companion JSON.

## Universal validation API

The accepted semantics are exposed as a domain-neutral validation layer under `nhsmm.validation` and via public package exports.

### Canonical facade

Normal callers should use:

- `ContextComponent = Literal["initial", "duration", "emission", "transition"]`
- `context_effect(model, context, component=...)`
- `evaluate_context_effect_replication(model_a, model_b, contexts, component=..., config=...)`
- `split_fit_context_effect_evidence(observations, context, component=..., fit_model=..., evidence_contexts=..., config=...)`

Shared result/policy types remain:

- `ContextEvidenceConfig`
- `ContextEvidence`
- `SplitFitEvidence`

Canonical semantic shapes:

- `initial` -> `[K]` normalized state prior;
- `duration` -> `[K,D]` normalized duration law;
- `emission` -> `[K,F]` state-conditioned means;
- `transition` -> `[K,K]` effective boundary-transition law.

### Compatibility/advanced helpers

Existing component-specific helpers remain available without behavior changes:

- `initial_context_tensor`, `duration_context_tensor`, `emission_context_tensor`, `transition_context_matrix`;
- `evaluate_*_context_replication` component helpers;
- `split_fit_*_context_evidence` component helpers.

Advanced generic helpers remain under `nhsmm.validation` for custom adapters:

- `align_state_centers`
- `evaluate_state_context_replication`
- `split_fit_state_context_evidence`
- `evaluate_context_replication`
- `split_fit_context_evidence`

Design rules:

- validation does not call or modify `NHSMM.optimize()`;
- training policy remains consumer-owned through `fit_model(...)`;
- no trading/domain assumptions;
- research thresholds are explicit policy configuration, not universal package defaults;
- amplitude units remain component-specific;
- initial-state context evidence uses normalized state priors, not internal logits;
- duration-dependent transition evidence integrates over the current duration law including `duration_logits_bias`;
- duration-context evidence evaluates the normalized duration distribution including `duration_logits_bias`;
- emission-context evidence evaluates state-conditioned emission means;
- model training/eval mode is preserved by NHSMM adapters;
- state alignment is semantic via learned emission centers rather than latent truth labels.

### API hardening

Focused semantic and contract coverage verifies:

- latent-state permutation invariance;
- inconsistent replicated direction rejection;
- null-effect rejection;
- disjoint split-fit training units;
- duration-weighted effective transition matrices;
- generic `[K,...]` state-effect permutation invariance;
- normalized initial-state priors;
- normalized duration context tensors;
- emission state-feature tensor extraction;
- model mode preservation;
- explicit context-dimension errors;
- non-finite context rejection;
- non-finite evidence-policy threshold rejection;
- non-finite effect-matrix rejection;
- initialized-distribution requirement;
- normalized facade dispatch equivalence against all component-specific helpers;
- normalized split-fit facade preserves the two-replica contract;
- invalid component names fail explicitly;
- root exports expose the canonical facade.

Focused local result: **23/23 PASS**.

### Public API stress

A dedicated local stress run against the normalized facade completed successfully:

- 40 initialized NHSMM models across seeds `100..139`;
- 15 context points per model;
- 4,800 repeated semantic effect extractions across all four components;
- 160 replica-evidence evaluations;
- 16 real `NHSMM.optimize()` split-fit evaluations across all four components;
- exact repeated-call determinism checked for effect extraction;
- train/eval mode restoration checked throughout;
- initial/duration/transition normalization checked throughout;
- no NaN/Inf or shape-contract failure observed.

Observed local runtime: `22.421 s`; Python-tracked peak memory: `53.47 MiB`.

See [`validation/public-api-stress-2026-09-28.md`](validation/public-api-stress-2026-09-28.md).

Documentation: [`validation-api.md`](validation-api.md).

## Verification limits

The full discovered pytest suite has not yet been rerun after the validation API work because the active local workspace is a sparse source reconstruction and lacks several regular package modules imported by the root package. Temporary local import stubs were used only to isolate focused validation tests; they are not repository changes.

The GitHub connector exposes the complete repository tree but does not provide a mountable source archive in this workflow. Repo transport remains connector-only; no shell Git/curl checkout was used. A complete local checkout is therefore still the required environment for the broad regression run.

No GitHub Actions run is used for this research loop.

## Package boundary

These controlled synthetic results establish mechanism recovery, identifiability, and validation semantics. They do not establish domain semantics or downstream predictive value for Nautilus or any other consumer.

## Next slice

1. Use a complete local checkout of `develop` and run the broad local test suite when that checkout is available in the runtime.
2. Fix only regressions exposed by that complete-package run; preserve accepted probabilistic and validation semantics.
3. Continue behavior-preserving code audit on remaining core modules where useful.
4. Keep all validation thresholds explicit and domain/profile-owned.
5. Treat downstream Nautilus integration as a separate empirical validation task.
