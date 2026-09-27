# NHSMM State

Current package readiness and next package-level research boundary. Detailed semantics: [`model.md`](model.md). Controlled evidence: [`package-validation.md`](package-validation.md). Verification policy: [`testing.md`](testing.md).

## Current status

**Phase:** duration-context, latent-state, and transition-context recovery baselines verified. Next research target: **multi-component identifiability** under controlled ground truth.

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

## Verification limits

The full discovered pytest suite was not rerun because the local workspace is sparse and only directly affected tests were materialized. `ruff` and `black` are not installed in the available local Python runtime. No GitHub Actions run was started.

## Package boundary

These controlled results establish mechanism recovery under known synthetic ground truth. They do not establish domain semantics or downstream predictive value for any consumer.

## Next research slices

1. **Multi-component identifiability:** combine state-, transition-, and duration-context effects and test whether each mechanism remains separately recoverable.
2. **General-context robustness:** transition recovery with continuous and multi-dimensional causal contexts.
3. Change objectives or parameterizations only if those controlled gates expose a reproducible package deficiency.
