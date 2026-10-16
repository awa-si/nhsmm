# Validation

`nhsmm.validation` provides domain-neutral evidence checks for learned context effects. Validation is deliberately separate from `NHSMM.optimize()`: training policy stays with the caller, while validation tests whether a fitted context effect reproduces across independent fits.

## Validation contract and limitations

Context probes consume one scalar/vector in the fitted component-context space;
they do not encode an observation sequence. A probe temporarily uses evaluation
mode and restores every PyTorch submodule's original mode, including after an
exception. Transition probes require exactly B=T=1; additional batch/time rows
are rejected rather than discarded.

Transition-like callbacks may return non-negative row weights; validation
normalizes them stably. Negative/non-finite weights and empty support are invalid.
With exclude_self_transitions=True, selection measures the destination law
conditional on changing state, removing and renormalizing the diagonal. Thus a
two-state chain has no varying destination choice under this policy; self-renewal
or duration effects must be assessed separately. With False, all transition mass
is included. State-effect amplitudes retain the component's units.

Selection requires strictly positive amplitude in both replicas and the explicit
configured amplitude/correlation thresholds. Constant effects cannot be selected
even with zero thresholds. Correlation is scale-independent Pearson correlation;
constant/singleton vectors return zero, empty/non-vector inputs are rejected.
Callback outputs are captured immediately, so a reused scratch buffer cannot
overwrite earlier grid points.

Split-fit validates an integer split index and a grid of at least two contexts
before calling either fit, and requires two distinct returned model instances.
It splits the first axis as supplied: slicing does not establish statistical
independence, remove overlapping windows, purge adjacent labels, or prevent a
callback from sharing weights/data. The caller owns chronological partitioning,
training-only preprocessing, independent fitting state/seeds, and a predeclared
context grid. Reusing a model object is rejected, but hidden sharing is not
detectable here.

Emission-center alignment is a permutation diagnostic; indistinguishable centers
can make the assignment ambiguous. These checks are descriptive replication
evidence, not a p-value, calibrated false-positive rate, causal-effect estimate,
or held-out predictive/trading acceptance. Snapshot Viterbi/health summaries are
retrospective and are not causal filtered signals. Setting require_same_model=False
does not align independently fitted state labels; occupancy L1 then has no
state-semantic interpretation without caller-established alignment.

See [V3 validation audit](https://github.com/awa-si/nhsmm/blob/c0178857da54f7099d1cb39cb76a7c5a0a6680a8/docs/validation/v3-validation-path-20261015.md) for defects,
regression evidence and the unchanged research boundary.

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

`ContextEffectComponent` is exported as the corresponding typing alias.

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

- two distinct fitted replicas on disjoint first-axis slices; independence is caller-established;
- latent-state alignment from learned emission centers only;
- comparison on an explicit context evidence grid;
- selection requires positive effect size and the caller's explicit correlation/amplitude thresholds;
- no latent truth labels are needed;
- no domain-specific labels or trading assumptions are used.

Transition evidence retains its dedicated internal implementation because it has two latent-state axes and boundary semantics. The normalized facade hides that implementation distinction from normal callers.

## Internal implementation

Component-specific extraction and evidence helpers remain internal implementation details under `nhsmm.validation` modules. Application code should use the canonical component-selected facade above.

## Result objects

`ContextEvidence` contains `selected`, `replication_corr`, `direction_agreement`, `replica_amplitudes`, `min_replica_amplitude`, `state_alignment`, and `n_contexts`.

`SplitFitEvidence` adds `split_index` and `replica_sizes`.

These objects report evidence; they do not automatically enable, disable, prune, or retrain model components.


## Reproducible validation snapshots

For later train/OOS, walk-forward, replay, or deployment validation, use:

```python
from nhsmm import (
    evaluate_validation_snapshot,
    compare_validation_snapshots,
)

train = evaluate_validation_snapshot(model, train_x, label="train")
oos = evaluate_validation_snapshot(model, oos_x, label="oos")
comparison = compare_validation_snapshots(train, oos)
```

`ValidationSnapshot` records domain-neutral evidence without changing model state:

- deterministic model fingerprint from resolved config + state dict;
- deterministic data fingerprint and optional explicit-context fingerprint;
- sequence counts and original lengths;
- total, per-sequence, and per-timestep log likelihood;
- decoded state-switch rate and mean decoded run length;
- the full `ModelHealthReport`.

`compare_validation_snapshots(...)` reports reference-to-candidate deltas for likelihood, effective-state count, occupancy concentration, posterior entropy, switch rate, run length, and occupancy L1 distance. By default it requires the same fitted-model fingerprint, preventing accidental direct comparison of unaligned latent-state labels from independently fitted models.

Snapshots and comparisons expose `.as_dict()` and are JSON-serializable. They deliberately contain no timestamps or trading-specific labels so identical model/data inputs produce stable validation evidence.


## Research ownership

This package retains only domain-neutral component probes, alignment diagnostics,
split-fit evidence and validation snapshots used by the host Qlib recorder.
Standalone experiment orchestration and the former synthetic scenario configs
are retired. Research tasks, datasets, candidate budgets and tracking belong to
[awa-si/nautilus/adapter/qlib](https://github.com/awa-si/nautilus/tree/main/adapter/qlib).
See [testing.md](testing.md) for current correctness checks and retained Oracle controls.
