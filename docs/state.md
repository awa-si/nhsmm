# NHSMM State

Current package readiness and the next package-level research boundary. Detailed model semantics live in [`model.md`](model.md); controlled empirical evidence lives in [`package-validation.md`](package-validation.md); verification policy lives in [`testing.md`](testing.md).

## Current status

**Phase:** duration-context and latent-state recovery baselines verified; next research target is transition-context recovery under controlled ground truth.

**Verified code baseline:** `89bf5aea144cde5e3331971d00df76191bfb088f` (`training: add robust state-recovery initialization`).

Package validation is independent of Nautilus, market data, trading labels, and downstream supervised models.

## Verified training baseline

- [x] `NHSMM.optimize()` covers all trainable parameters, including the causal encoder.
- [x] Default maximum training budget is `40` iterations.
- [x] `n_init` no longer warm-starts later runs from the previous best run.
- [x] Best-run snapshots are independent deep copies and include `duration_logits_bias`.
- [x] Default emission initialization remains `spread`.
- [x] Explicit `emission_init_mode="kmeans"` is available for state-identifiability-sensitive training.
- [x] No new auxiliary loss or duration parameterization is required by the current controlled benchmarks.

## Latest local evidence

Focused training regression:

```text
PYTHONPATH=. pytest -q tests/test_training.py
6 passed
```

Syntax compilation:

```text
python -m compileall -q nhsmm scripts/validate_state_recovery.py tests/test_training.py
PASS
```

Duration-context regression, seeds 201-215, default `spread`:

| Scenario | Median gap | Mean gap | Non-collapsed | Result |
| --- | ---: | ---: | ---: | --- |
| strong | 0.09954 | 0.09166 | 15/15 | PASS |
| moderate | 0.02434 | 0.03218 | 15/15 | PASS |
| null | 0.01370 | 0.01184 | 15/15 | PASS |

Latent-state recovery, seeds 301-315, explicit `kmeans`:

| Scenario | Median accuracy | Median ARI | Non-collapsed | Result |
| --- | ---: | ---: | ---: | --- |
| strong | 0.99667 | 0.99018 | 15/15 | PASS |
| moderate | 0.93000 | 0.80182 | 15/15 | PASS |
| null | 0.36333 | 0.00000 | descriptive | PASS |

## Verification limits

The full discovered pytest suite was not rerun in the current sparse local workspace because only the directly affected tests were materialized. `ruff` and `black` are not installed in the available local Python runtime. No GitHub Actions run was started.

## Package boundary

The package-level claims remain mechanism-specific. The controlled benchmarks establish duration-context recovery and permutation-invariant latent-state recovery under known synthetic ground truth. They do not establish domain semantics, useful transition forecasts, or downstream predictive value for any consumer.

## Next research slices

1. **Transition-context recovery:** generate known context-dependent transition laws and verify causal transition forecasts recover direction/effect size, including a null-context control.
2. **Multi-component identifiability:** combine state-, transition-, and duration-context effects and test whether each mechanism is recoverable without another component absorbing it.
3. Change objectives or parameterizations only if those controlled gates reveal a reproducible package deficiency.
