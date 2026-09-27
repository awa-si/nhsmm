# Public API stress test — 2026-09-28

Status: **PASS**.

## Scope

Local stress run against the normalized public validation facade on `develop` at source head `aa6e1d0268481c040915f0691f4c96f840c61a63` before this report commit.

Canonical API exercised:

- `context_effect(..., component=...)`
- `evaluate_context_effect_replication(..., component=...)`
- `split_fit_context_effect_evidence(..., component=...)`

Components:

- `initial`
- `duration`
- `emission`
- `transition`

No model, threshold, training, or validation semantics were changed for this run. No CI was used.

## Workload

### Broad inference sweep

- model seeds: `100..139` (`40` initialized NHSMM instances)
- evidence grid: `15` two-dimensional context points per model
- components: `4`
- every extraction repeated twice to test exact repeat determinism
- total semantic effect extractions: **4,800**

Checks on every extraction:

- finite output;
- canonical component shape;
- probability normalization for initial, duration, and transition effects;
- exact repeated-call equality;
- preservation of the model's prior train/eval mode.

### Replication sweep

- seeds: `200..219`
- identical-model replica comparisons plus independently initialized replica comparisons
- all four components
- total replication evaluations: **160**

Checks:

- finite replication correlation;
- finite minimum replica amplitude;
- correct context-grid cardinality;
- no dispatch or alignment failure.

### Optimize split-fit sweep

- seeds: `300..303`
- observations per run: `6 x 7 x 2`
- context per run: `6 x 7 x 2`
- two disjoint replicas of `3` training units each
- `n_init=1`, `max_iter=2`, scheduler/convergence stopping disabled for the stress harness
- all four components
- total real `NHSMM.optimize()` split-fit evaluations: **16**

Checks:

- exactly two disjoint fit callbacks per split-fit call;
- replica sizes `(3, 3)`;
- finite evidence metrics;
- correct evidence-grid cardinality;
- no component-specific dispatch failure.

## Result

```text
STRESS_PASS effects=4800 replications=160 optimize_splitfits=16 elapsed_s=22.421 peak_py_mb=53.47
```

Observed Python-tracked peak memory: **53.47 MiB**.
Observed wall time in the active local runtime: **22.421 s**.

## Conclusion

**PASS.** The normalized public validation facade remained stable across repeated extraction, replica evaluation, and real optimize-based split-fit use for all four supported components. No NaN/Inf, shape, normalization, mode-restoration, deterministic-repeat, or split-contract failure was observed.

This is an API/runtime stress result, not new statistical validation evidence and not downstream-domain evidence.
