# NHSMM State

Current package readiness and the next package-level research boundary. Detailed model semantics live in [`model.md`](model.md); controlled duration-context evidence lives in [`package-validation.md`](package-validation.md); verification policy lives in [`testing.md`](testing.md).

## Current status

**Phase:** package baseline verified; next research target is latent-state and transition-context recovery under controlled ground truth.

The current `develop` baseline includes the causal runtime/artifact contracts, full encoder participation in `NHSMM.optimize()`, and a default `max_iter=40` training budget. Package-level validation is intentionally independent of Nautilus, market data, trading labels, and downstream supervised models.

The current controlled duration-context baseline is accepted. On synthetic sequences with known duration ground truth, the committed training/runtime path recovers context-conditioned duration/hazard structure across independent seeds while remaining non-collapsed and while the null-context control stays below the configured spurious-separation threshold.

## Verified baseline

### Training

- [x] `NHSMM.optimize()` optimizes every trainable `model.parameters()` entry, including the causal context encoder.
- [x] Regression coverage verifies encoder parameters are present in the optimizer and at least one encoder parameter changes after optimization.
- [x] Default `ModelConfig.max_iter == 40`; convergence stopping and scheduling remain configurable.
- [x] No auxiliary duration objective or alternative duration parameterization is required by the current controlled benchmark.

### Causal model/runtime

- [x] Dynamic causal boundary-time hazard semantics use only information available at `F_t` for boundary `t -> t+1`.
- [x] Filtering represents normalized `P(z_t, age_t | F_t)` and preserves duration/age indexing semantics.
- [x] Survival/end-within-horizon outputs are canonical duration forecasts.
- [x] State-change probability remains distinct from episode-end probability.
- [x] Prefix causality, probability normalization, survival monotonicity, one-step episode-end consistency, and bounded streaming state are maintained.

### Artifact/inference

- [x] Artifact v1 persists resolved `ModelConfig`, canonical encoder metadata, causal mode, schema metadata, and full model/distribution state.
- [x] Loading is strict and fail-closed for incompatible artifact/config/schema state.
- [x] Production inference preparation freezes parameters by default and preserves artifact/state-dict compatibility.

## Current package evidence

Latest local verification was performed against source blobs matching commit `fac6bc46bcdf193e8f289f336d9b668247a9f350`.

Focused training regression:

```text
PYTHONPATH=. pytest -q tests/test_training.py
2 passed
```

Syntax/import compilation:

```text
python -m compileall -q nhsmm scripts/validate_duration_context.py tests/test_training.py
PASS
```

Controlled 15-seed acceptance, using the committed `_train`, `_generate`, `_evaluate`, and `_summary` functions with CPU intra/inter-op threads limited to one only to fit the local execution budget:

| Scenario | Positive gap | Gap >= 0.02 | Median gap | Mean gap | Non-collapsed |
| --- | ---: | ---: | ---: | ---: | ---: |
| strong context, duration means 6 vs 18 | 15/15 | 15/15 | 0.09723 | 0.09305 | 15/15 |
| moderate context, duration means 9 vs 15 | 14/15 | 12/15 | 0.03588 | 0.03668 | 15/15 |
| null context, duration means 12 vs 12 | descriptive | 5/15 | 0.00907 | 0.01064 | 15/15 |

Acceptance result: **PASS** for strong context, moderate context, and null-context no-spurious-separation gates.

The multi-seed run was chunked only because the local execution tool terminates foreground processes after roughly 30 seconds. Model configuration, seeds, synthetic generator, training logic, evaluation functions, and acceptance thresholds were unchanged. The final aggregate contains exactly seeds `201..215` for each of the three scenarios.

## Verification limits

The full discovered pytest suite was **not** rerun on this commit in the current session. The Git-backed AWA workspace was materialized at the correct commit, but both currently allowlisted execution images (`python:3.14-slim` and `alpine:3.22`) were unavailable on the host. The executable `/tmp` package source was therefore hash-checked against the current Git blobs before focused execution.

`ruff` and `black` were not installed in the available local Python runtime, so the repository static-format checks remain unverified for this session. No GitHub Actions run was started; CI is not the local edit/test loop.

## Package boundary

The package-level claim is deliberately narrow: NHSMM can recover known context-conditioned episode-duration structure under controlled synthetic ground truth with the current causal training/runtime contract.

This does **not** establish useful state semantics, transition forecasts, or downstream predictive value in any external domain. Nautilus or any other consumer must evaluate those properties independently with its own causally valid out-of-sample contract.

## Next research slices

1. **Latent-state recovery:** controlled synthetic benchmark with known state identity up to permutation; measure recovery, occupancy robustness, and seed stability without assigning domain semantics.
2. **Transition-context recovery:** generate known context-dependent transition laws and verify that causal transition forecasts recover direction and effect size, including a null-context control.
3. **Multi-component identifiability:** combine state-, transition-, and duration-context effects and determine when the package can recover each component without one mechanism absorbing another.
4. Only if those controlled gates expose a reproducible package deficiency, change the objective or parameterization; otherwise keep the current baseline stable.
