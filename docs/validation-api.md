# Validation API

`nhsmm.validation` provides domain-neutral evidence checks for learned context effects. Validation is deliberately separate from `NHSMM.optimize()`: training policy stays with the caller, while validation tests whether a fitted context effect reproduces across independent fits.

## Transition-context evidence

For a fitted NHSMM, the effective boundary-transition matrix at one context value is available directly:

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

if evidence.selected:
    ...
```

Thresholds are explicit policy. NHSMM does not treat any research threshold as a universal probabilistic default.

## Split-fit validation

`split_fit_transition_context_evidence` creates two disjoint training partitions and delegates fitting back to the caller:

```python
from nhsmm import ContextEvidenceConfig, split_fit_transition_context_evidence

policy = ContextEvidenceConfig(
    replication_corr_min=0.55,
    min_replica_amplitude=0.50,
)


def fit_replica(x_split, context_split, replica_index):
    model = make_model(seed=base_seed + replica_index)
    model.optimize(x_split, context=context_split)
    return model

result = split_fit_transition_context_evidence(
    observations,
    context,
    fit_model=fit_replica,
    evidence_contexts=evidence_grid,
    config=policy,
)

print(result.selected)
print(result.evidence.replication_corr)
print(result.evidence.min_replica_amplitude)
```

The callback owns initialization, optimization, refinement, seeds, stopping rules, and any application-specific training choices. Validation never calls or mutates `NHSMM.optimize()` itself.

## Semantics

The accepted transition-context validation semantics are:

- two independent fits on disjoint training units;
- latent-state alignment from learned emission centers only;
- comparison of learned context-induced transition functions on an explicit evidence grid;
- optional exclusion of self-transitions through conditional-on-state-change normalization;
- selection requires both replicated direction and replicated effect size;
- no latent truth labels are needed;
- no domain-specific labels or trading assumptions are used.

The lower-level generic API remains available through `nhsmm.validation` for custom model families or non-transition context effects:

- `evaluate_context_replication(...)`
- `split_fit_context_evidence(...)`
- `align_state_centers(...)`
- `reorder_square_matrix(...)`

## Result objects

`ContextEvidence` contains:

- `selected`
- `replication_corr`
- `direction_agreement`
- `replica_amplitudes`
- `min_replica_amplitude`
- `state_alignment`
- `n_contexts`

`SplitFitEvidence` adds:

- `split_index`
- `replica_sizes`

These objects report evidence; they do not automatically enable, disable, prune, or retrain model components.
