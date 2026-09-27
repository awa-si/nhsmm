# Validation API

`nhsmm.validation` provides domain-neutral evidence checks for learned context effects. Validation is deliberately separate from `NHSMM.optimize()`: training policy stays with the caller, while validation tests whether a fitted context effect reproduces across independent fits.

## State-indexed context evidence

The universal path accepts any learned effect tensor whose first axis is latent state:

```text
[K, ...]
```

Only the first axis is interpreted by validation. Remaining axes belong to the component. State alignment uses learned emission centers; latent truth labels are not required.

Public helpers:

- `evaluate_state_context_replication(...)`
- `split_fit_state_context_evidence(...)`
- `initial_context_tensor(...)`
- `duration_context_tensor(...)`
- `emission_context_tensor(...)`
- `evaluate_initial_context_replication(...)`
- `evaluate_duration_context_replication(...)`
- `evaluate_emission_context_replication(...)`
- `split_fit_initial_context_evidence(...)`
- `split_fit_duration_context_evidence(...)`
- `split_fit_emission_context_evidence(...)`

Initial-state effects are represented as normalized `[K]` probabilities. Duration effects are normalized `[K,D]` distributions including `duration_logits_bias`. Emission effects are state-conditioned means `[K,F]`. All NHSMM adapters preserve the model's prior train/eval mode.

Example:

```python
from nhsmm import ContextEvidenceConfig, evaluate_duration_context_replication

policy = ContextEvidenceConfig(
    replication_corr_min=0.55,
    min_replica_amplitude=0.10,
)

evidence = evaluate_duration_context_replication(
    model_a,
    model_b,
    contexts=evidence_grid,
    config=policy,
)
```

Thresholds are always explicit policy; amplitude units are component-specific and therefore are not package defaults.

## Transition-context evidence

Transitions retain their dedicated validation path because they contain two latent-state axes and have conditional boundary semantics. For a fitted NHSMM, the effective boundary-transition matrix at one context value is available directly:

```python
from nhsmm import transition_context_matrix

matrix = transition_context_matrix(model, context=[0.2, -0.4])
```

For duration-dependent models this integrates transition probabilities over the model's duration distribution, including `duration_logits_bias`. The result is a normalized `K x K` boundary-transition matrix.

To compare two independently fitted replicas:

```python
from nhsmm import ContextEvidenceConfig, evaluate_transition_context_replication

policy = ContextEvidenceConfig(
    replication_corr_min=0.55,
    min_replica_amplitude=0.50,
)

evidence = evaluate_transition_context_replication(
    model_a,
    model_b,
    contexts=[[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]],
    config=policy,
)
```

## Split-fit validation

Split-fit helpers create two disjoint training partitions and delegate fitting back to the caller. The callback owns initialization, optimization, refinement, seeds, stopping rules, and application-specific training choices. Validation never calls or mutates `NHSMM.optimize()` itself.

## Semantics

Shared validation semantics are:

- two independent fits on disjoint training units;
- latent-state alignment from learned emission centers only;
- comparison on an explicit context evidence grid;
- selection requires both replicated direction and replicated effect size;
- no latent truth labels are needed;
- no domain-specific labels or trading assumptions are used.

Transition-specific semantics additionally support conditional-on-state-change normalization and optional self-transition exclusion.

Lower-level APIs remain available through `nhsmm.validation`:

- `evaluate_context_replication(...)` for square transition-like effects;
- `split_fit_context_evidence(...)`;
- `evaluate_state_context_replication(...)` for arbitrary `[K,...]` effects;
- `split_fit_state_context_evidence(...)`;
- `align_state_centers(...)`;
- `reorder_square_matrix(...)`.

## Result objects

`ContextEvidence` contains `selected`, `replication_corr`, `direction_agreement`, `replica_amplitudes`, `min_replica_amplitude`, `state_alignment`, and `n_contexts`.

`SplitFitEvidence` adds `split_index` and `replica_sizes`.

These objects report evidence; they do not automatically enable, disable, prune, or retrain model components.
