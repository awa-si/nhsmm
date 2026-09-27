# NHSMM handoff

This file is the short operational handoff for the next work session. `docs/state.md` remains the canonical readiness/status owner.

## Current continuation point

Repository: `awa-si/nhsmm`

Branch: `develop`

Snapshot date: `2026-09-28`

The package-core mechanism research and the universal validation/API translation are complete. Do not reopen detector tuning or redesign the accepted validation semantics unless a later package change invalidates them.

## Current verified state

Package-core evidence:

- duration context: PASS;
- latent-state recovery: PASS;
- transition context: PASS;
- multi-component identifiability: PACKAGE-CORE PASS;
- general-context robustness: PACKAGE-CORE PASS using split-fit direction replication.

Universal validation now covers:

- initial-state context effects `[K]`;
- duration context effects `[K,D]`;
- emission context effects `[K,F]`;
- transition context effects `[K,K]`;
- arbitrary state-indexed custom effects `[K,...]` through the advanced validation layer.

The normalized public facade is the preferred API:

```python
context_effect(model, context, component=...)
evaluate_context_effect_replication(
    model_a,
    model_b,
    contexts,
    component=...,
    config=...,
)
split_fit_context_effect_evidence(
    observations,
    context,
    component=...,
    fit_model=...,
    evidence_contexts=...,
    config=...,
)
```

`ContextComponent` is restricted to:

```text
initial | duration | emission | transition
```

Component-specific helper functions remain available as compatibility/advanced API and must remain behavior-compatible.

## Verification status

Focused semantic/API regression:

```text
23/23 PASS
```

Functional end-to-end smoke includes:

- real NHSMM initialization;
- all four normalized `component=` extraction paths;
- normalized-facade equality to component-specific helpers;
- replica-evidence dispatch;
- split-fit dispatch;
- a real split-fit callback using `initialize_distributions()` + `NHSMM.optimize()` on disjoint halves.

Stress result:

```text
STRESS_PASS
effects=4800
replications=160
optimize_splitfits=16
runtime=22.421 s
Python-tracked peak memory=53.47 MiB
```

Stress coverage included:

- 40 initialized models, seeds `100..139`;
- 15 context points per model;
- all four public components;
- exact repeated-call determinism for extraction;
- train/eval mode restoration;
- shape contracts;
- finite values;
- initial/duration/transition normalization;
- disjoint split-fit contract.

Persistent stress record:

`docs/validation/public-api-stress-2026-09-28.md`

## Important semantic constraints

Preserve these unless explicitly justified by new research:

- validation remains separate from `NHSMM.optimize()`;
- training policy is consumer-owned through `fit_model(...)`;
- no trading/Nautilus assumptions inside NHSMM validation;
- validation thresholds remain explicit policy, not package-core defaults;
- state alignment uses learned emission centers, not latent truth;
- initial-state evidence uses normalized probabilities, not internal logits;
- duration evidence includes `duration_logits_bias`;
- effective transition evidence integrates duration-dependent transition laws over the current duration distribution;
- adapters restore the previous train/eval mode;
- accepted split-fit transition detector thresholds remain frozen for the recorded research profile and must not be retuned on prior confirmatory seeds.

## Rejected research directions

Do not revive without materially new evidence:

- no-self topology as package fix;
- shared-duration context;
- shared-linear estimator as detector;
- effect-floor detector variants;
- fold-sign/fold-stability rules;
- single-permutation and permutation-plus-effect-floor variants;
- permutation-rank/binomial detector;
- Fisher fold-permutation detector.

## Current package boundary

The next clean package task is **not more detector/API expansion**.

Proceed in this order:

1. Use a complete local checkout of current `develop` when available.
2. Run the full discovered pytest suite locally.
3. Fix only regressions actually exposed by that run; preserve accepted model and validation semantics.
4. Continue behavior-preserving audit only where there is a concrete correctness/maintenance benefit.
5. Keep downstream Nautilus integration as a separate empirical boundary.

The current runtime used during validation was a sparse/local reconstruction, so focused and stress validation are authoritative for the new validation API, but they are not a substitute for a full-repository pytest pass.

## Source of truth

For detailed current status and evidence, read:

- `docs/state.md`
- `docs/validation-api.md`
- `docs/validation/public-api-stress-2026-09-28.md`
- `docs/validation/general-context-split-replication-confirm-2026-09-27.md`
- `docs/validation/multicomponent-component-matched-package-15seed-2026-09-27.md`
