# Validation API

`nhsmm.validation` provides domain-neutral evidence checks for learned context effects. Validation is deliberately separate from `NHSMM.optimize()`: training policy stays with the caller, while validation tests whether a fitted context effect reproduces across independent fits.

## Canonical public API

For normal application code, use the component-normalized facade exported directly from `nhsmm`:

```python
from nhsmm import (
    ContextEvidenceConfig,
    context_effect,
    evaluate_context_effect_replication,
    split_fit_context_effect_evidence,
)
```

All three operations take the same explicit component selector:

```text
component = "initial" | "duration" | "emission" | "transition"
```

The canonical operations are:

- `context_effect(model, context, component=...)`
- `evaluate_context_effect_replication(model_a, model_b, contexts, component=..., config=...)`
- `split_fit_context_effect_evidence(observations, context, component=..., fit_model=..., evidence_contexts=..., config=...)`

`ContextComponent` is exported as the corresponding typing alias.

Semantic effect shapes are stable by component:

- `initial` -> normalized initial-state probabilities `[K]`;
- `duration` -> normalized duration probabilities `[K,D]`, including `duration_logits_bias`;
- `emission` -> state-conditioned emission means `[K,F]`;
- `transition` -> effective boundary-transition probabilities `[K,K]`, duration-integrated where applicable.

Example:

```python
from nhsmm import ContextEvidenceConfig, evaluate_context_effect_replication

policy = ContextEvidenceConfig(
    replication_corr_min=0.55,
    min_replica_amplitude=0.10,
)

evidence = evaluate_context_effect_replication(
    model_a,
    model_b,
    contexts=evidence_grid,
    component="duration",
    config=policy,
)
```

Thresholds are always explicit policy. Amplitude units are component-specific and therefore are not universal package defaults.

## Split-fit validation

`split_fit_context_effect_evidence(...)` creates two disjoint training partitions and delegates fitting back to the caller. The callback owns initialization, optimization, refinement, seeds, stopping rules, and application-specific training choices. Validation never calls or mutates `NHSMM.optimize()` itself.

## Semantics

Shared validation semantics are:

- two independent fits on disjoint training units;
- latent-state alignment from learned emission centers only;
- comparison on an explicit context evidence grid;
- selection requires both replicated direction and replicated effect size;
- no latent truth labels are needed;
- no domain-specific labels or trading assumptions are used.

Transition evidence retains its dedicated internal implementation because it has two latent-state axes and boundary semantics. The normalized facade hides that implementation distinction from normal callers.

## Compatibility and advanced helpers

The earlier component-specific helpers remain available and behavior-compatible. They are useful for advanced consumers that need the component-specific representation directly:

- `initial_context_tensor(...)`
- `duration_context_tensor(...)`
- `emission_context_tensor(...)`
- `transition_context_matrix(...)`
- `evaluate_initial_context_replication(...)`
- `evaluate_duration_context_replication(...)`
- `evaluate_emission_context_replication(...)`
- `evaluate_transition_context_replication(...)`
- corresponding `split_fit_*_context_evidence(...)` helpers.

Lower-level generic machinery remains under `nhsmm.validation`:

- `evaluate_state_context_replication(...)` for arbitrary `[K,...]` effects;
- `split_fit_state_context_evidence(...)`;
- `evaluate_context_replication(...)` for square transition-like effects;
- `split_fit_context_evidence(...)`;
- `align_state_centers(...)`;
- `reorder_square_matrix(...)`.

These are not required for the normal public workflow.

## Result objects

`ContextEvidence` contains `selected`, `replication_corr`, `direction_agreement`, `replica_amplitudes`, `min_replica_amplitude`, `state_alignment`, and `n_contexts`.

`SplitFitEvidence` adds `split_index` and `replica_sizes`.

These objects report evidence; they do not automatically enable, disable, prune, or retrain model components.
