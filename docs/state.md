# NHSMM State

Current package readiness and next package-level research boundary. Detailed semantics: [`model.md`](model.md). Controlled evidence: [`package-validation.md`](package-validation.md). Verification policy: [`testing.md`](testing.md).

## Current status

**Phase:** duration-context, latent-state, and transition-context recovery baselines verified. **Multi-component identifiability is provisionally accepted** on the basis of the corrected local independent causal-HSMM reference deep dive recorded in [`validation/multicomponent-reference-deep-dive-2026-09-27.md`](validation/multicomponent-reference-deep-dive-2026-09-27.md).

This provisional acceptance is intentionally narrower than package-level acceptance: the corrected v2 validator still needs to run locally against the actual `nhsmm` package implementation. Until that run is available, no package objective or parameterization change is justified by the original failed CI benchmark.

Package validation remains independent of Nautilus, market data, trading labels, and downstream supervised models.

## Verified training baseline

- [x] `NHSMM.optimize()` covers all trainable parameters, including the causal encoder.
- [x] Default joint-training maximum is `40` iterations.
- [x] `n_init` uses independent restarts; best-run snapshots are deep and include `duration_logits_bias`.
- [x] Default emission initialization remains `spread`; explicit `kmeans` is available for state-identifiability-sensitive training.
- [x] Scalar external context retains a distribution hidden width >=16, avoiding `LayerNorm(1)` collapse.
- [x] Transition context capacity is independently configurable.
- [x] Optional transition refinement updates only `transition.context_net + transition.delta_scale` against exact sequence likelihood; it is disabled by default.
- [x] No auxiliary supervised transition loss or alternative duration parameterization is required by the current controlled gates.

## Latest local evidence

Focused training regression:

```text
PYTHONPATH=. pytest -q tests/test_training.py
10 passed
```

Syntax compilation:

```text
python -m compileall -q nhsmm scripts/validate_duration_context.py scripts/validate_state_recovery.py scripts/validate_transition_context.py tests/test_training.py
PASS
```

### Duration context — seeds 201..215

| Scenario | Median gap | Mean gap | Non-collapsed | Result |
| --- | ---: | ---: | ---: | --- |
| strong | 0.09954 | 0.09166 | 15/15 | PASS |
| moderate | 0.02434 | 0.03218 | 15/15 | PASS |
| null | 0.01370 | 0.01184 | 15/15 | PASS |

### Latent state — seeds 301..315

| Scenario | Median accuracy | Median ARI | Non-collapsed | Result |
| --- | ---: | ---: | ---: | --- |
| strong | 0.99667 | 0.99018 | 15/15 | PASS |
| moderate | 0.93000 | 0.80182 | 15/15 | PASS |
| null | 0.36333 | 0.00000 | descriptive | PASS |

### Transition context — seeds 401..415

Validation config: `max_duration=1`, explicit binary external context, `transition_context_max_delta=1.5`, `transition_refine_steps=20`, `transition_refine_lr=0.03`.

| Scenario | Median MAE | Median KL | Delta correlation | Direction | Abs context delta | Non-collapsed | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| strong | 0.01782 | 0.00370 | 0.99999 | 1.000 | 0.47720 | 15/15 | PASS |
| moderate | 0.01833 | 0.00260 | 0.99194 | 1.000 | 0.16448 | 15/15 | PASS |
| null | 0.00524 | 0.00032 | 0.00000 | n/a | 3.04e-7 | 15/15 | PASS |

### Multi-component identifiability — provisional acceptance

The original joint benchmark is not accepted as evidence because it was structurally misspecified: it used `transition_type="semi"` against a cyclic generator that required broader support, scored duration-dependent transition tensors incorrectly, and omitted `duration_logits_bias` from duration scoring.

The corrected independent local causal-HSMM reference used `K=3`, `D=24`, seeds `501..503`, explicit A/B/C/D ablations, and an ergodic topology. It preserved state recovery across strong, moderate, and null scenarios. Strong full-joint D reached median matched accuracy `1.000`, median ARI `1.000`, median effective states `2.968`, median duration gap `2.699`, and median transition-delta correlation `0.987`. Moderate full-joint D likewise retained accuracy/ARI `1.000` with median transition correlation `0.861`. Null A-D retained state identity without a material spurious context effect.

A counterfactual rerun with the old `semi` support reproduced the collapse pattern: strong full-joint D fell to median matched accuracy `0.603`, median ARI `0.234`, median effective states `2.121`, and median transition correlation `0.571`.

**Status:** PROVISIONAL PASS for the research question "is joint state + duration-context + transition-context identification structurally possible under the corrected benchmark?" This does **not** yet certify the actual package training implementation. Package-level acceptance requires one corrected local v2 run against `nhsmm` itself.

## Verification limits

The full discovered pytest suite was not rerun because the local workspace is sparse and only directly affected tests were materialized. `ruff` and `black` are not installed in the available local Python runtime. No GitHub Actions run was started for the deep-dive acceptance.

## Package boundary

These controlled results establish mechanism recovery under known synthetic ground truth. They do not establish domain semantics or downstream predictive value for any consumer.

## Next research slices

1. **Package confirmation:** run the corrected v2 multi-component validator locally against the actual `nhsmm` package and either promote provisional acceptance to package-level PASS or reopen the deficiency if the package diverges from the corrected reference.
2. **General-context robustness:** transition recovery with continuous and multi-dimensional causal contexts.
3. Change objectives or parameterizations only if the corrected package-level gate exposes a reproducible package deficiency.
