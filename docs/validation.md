# Validation

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

- two independent fits on disjoint training units;
- latent-state alignment from learned emission centers only;
- comparison on an explicit context evidence grid;
- selection requires both replicated direction and replicated effect size;
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

## Validation configuration contract

Controlled learning/prediction acceptance uses a separate public configuration surface from `ModelConfig`:

- `ValidationConfig` — schema version, seeds, boundary tolerance, model template, health thresholds, data policy, and named scenarios;
- `ValidationDataConfig` — sequence counts/length and synthetic duration-generation policy;
- `ValidationScenarioConfig` — explicit learning and structural-usefulness thresholds.

The contract is strict and JSON-serializable. Unknown keys, unsupported schema versions, invalid ranges, incompatible dimensions, and contradictory policies fail closed. Tuning returns validated copies via `with_overrides(...)`; no scenario name carries hidden behavior.

See [configuration.md](configuration.md) for the full contract and CLI examples.

## Controlled package evidence

### Scope

This document records controlled package-level empirical evidence. It is independent of Nautilus, market data, trading targets, and downstream supervised models. The claims are mechanism-specific and do not establish external-domain utility.

### Duration-context recovery

The duration benchmark uses two latent states and an observed context that changes episode-duration ground truth. Training explicitly opts into the internal `ContextEncoder` and uses causal NHSMM, `K=2`, `D=30`, `max_iter=40`, and independent evaluation sequences. Independent seed runs may execute in parallel; model, data, seeds, thresholds, and per-seed semantics are unchanged. The main diagnostic is the difference in mean probability of ending within five steps between short- and long-duration contexts. The null guard requires both absolute median and mean gaps to stay at or below `0.015`.

Seeds `201..215`, default `emission_init_mode="spread"`:

| Scenario | Positive gap | Gap >= 0.02 | Median gap | Mean gap | Non-collapsed |
| --- | ---: | ---: | ---: | ---: | ---: |
| strong, means 6 vs 18 | 15/15 | 15/15 | 0.09642 | 0.09432 | 15/15 |
| moderate, means 9 vs 15 | 13/15 | 8/15 | 0.02198 | 0.02939 | 15/15 |
| null, means 12 vs 12 | 9/15 | 7/15 | 0.01141 | 0.01238 | 15/15 |

Acceptance: **PASS**.

### Permutation-invariant latent-state recovery

The state benchmark uses `K=3`, three observed coordinates, explicit-duration episodes, and state-dependent Gaussian means. Labels are scored only up to permutation. The benchmark explicitly opts into `emission_init_mode="kmeans"`; the package default remains `spread`.

Predefined gates: strong median accuracy >= `0.90`, ARI >= `0.80`; moderate accuracy >= `0.70`, ARI >= `0.40`; null accuracy <= `0.45` and |median ARI| <= `0.10`; strong/moderate require at least 14/15 non-collapsed runs.

Seeds `301..315`:

| Scenario | Median accuracy | Median ARI | Median effective states | Non-collapsed |
| --- | ---: | ---: | ---: | ---: |
| strong | 0.99889 | 0.99661 | 2.99113 | 15/15 |
| moderate | 0.94222 | 0.83349 | 2.99204 | 15/15 |
| null | 0.36444 | 0.00287 | 2.93496 | 15/15 |

Acceptance: **PASS**.

The earlier data-independent `spread` initialization could enter persistent two-state local minima on strongly identifiable synthetic data. More iterations and marginal state-usage regularization did not reliably rescue those seeds. Training-time K-Means++/Lloyd initialization did, so K-Means remains an explicit opt-in rather than the global default.

### End-to-end learning and held-out prediction

A dedicated end-to-end harness tests whether training changes the model and improves prediction on strictly disjoint OOS sequences rather than merely exercising package APIs.

Five independent seeds were run for each of four signal strengths using emission_init_mode="kmeans" for identifiable-state acceptance:

| Scenario | Median OOS LL gain | Median OOS accuracy | Median accuracy gain | Median OOS ARI | OOS collapse |
| --- | ---: | ---: | ---: | ---: | ---: |
| strong | +7.7661 | 0.9989 | +0.3433 | 0.9965 | 0/5 |
| moderate | +2.1210 | 0.9633 | +0.3400 | 0.8936 | 0/5 |
| weak | +0.9926 | 0.7856 | +0.2522 | 0.4642 | 0/5 |
| null | +0.9303 | 0.3589 | -0.0144 | 0.0000 | 5/5 |

All 20 fits changed parameters, preserved train/OOS model fingerprints, used distinct train/OOS data fingerprints, and passed the variable-length prediction contract. The null control improves density fit without creating meaningful state-label recovery.

Acceptance: **PASS**.

The acceptance harness now distinguishes **learning_pass** from **usefulness_pass**. In addition to state recovery and OOS likelihood, usefulness is scored using boundary F1, ±1-timestep boundary F1, run-length recovery, transition-matrix MAE, occupancy L1, posterior calibration, collapse behavior, and train/OOS stability.

Strong and moderate pass both gates cleanly. Weak also passes the maintained weak-signal usefulness contract, but exact boundary timing is intentionally qualified: median exact boundary F1 is 0.3222 versus 0.6000 with ±1-timestep tolerance. Null remains a negative control and does not receive a usefulness claim.

Detailed historical evidence is indexed under validation/README.md; current accepted claims are summarized here and in state.md.

### Transition-context recovery

The transition benchmark isolates transition learning with `K=3`, strongly identifiable Gaussian emissions, `max_duration=1`, and a binary causal context `c_t` that selects the known transition law for boundary `t -> t+1`. State identity is aligned up to permutation before transition matrices are scored.

Two package deficiencies were isolated before final acceptance:

- one-dimensional external context inherited a one-unit distribution hidden layer, so `LayerNorm(1)` erased all context variation; distribution context networks now retain a minimum hidden width of 16;
- joint training learned transition-effect direction but underfit strong context amplitude. Increasing only clamp size, training budget, learning rate, initial `delta_scale`, or changing activation did not robustly solve it.

Component ablation showed that post-joint refinement needs only `transition.context_net + transition.delta_scale`; base transition logits need not move. The maintained benchmark therefore opts into:

- `transition_context_max_delta=1.5`;
- `transition_refine_steps=20`;
- `transition_refine_lr=0.03`;
- exact marginal sequence likelihood, temperature `1.0`, no duration-bias penalty;
- all non-refined parameters frozen.

Predefined gates: strong MAE <= `0.15`, delta correlation >= `0.70`, direction >= `0.80`; moderate MAE <= `0.18`, correlation >= `0.45`, direction >= `0.70`; null median absolute learned context delta <= `0.08`; strong/moderate require at least 14/15 non-collapsed runs.

Seeds `401..415`:

| Scenario | Median MAE | Median KL | Delta correlation | Direction | Abs context delta | Non-collapsed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| strong | 0.02172 | 0.00517 | 0.99999 | 1.000 | 0.47229 | 15/15 |
| moderate | 0.01746 | 0.00243 | 0.99377 | 1.000 | 0.17472 | 15/15 |
| null | 0.00533 | 0.00031 | 0.00000 | n/a | 6.82e-7 | 15/15 |

Acceptance: **PASS**. The null control remains far below the `0.08` spurious-effect guard.

### 2026-10-06 revalidation

The complete controlled suite was re-run after the ContextEncoder opt-in migration and duration-harness repair. Every maintained scenario gate passed. Runs were split into scenario-sized batches so each result had an independent timeout and persistent workspace artifact; this batching changes execution orchestration only, not model/data/seed/threshold semantics.

### Interpretation

Current controlled evidence supports three narrow package claims: duration-context ordering is recoverable without spurious null separation; identifiable latent states are recoverable up to permutation with explicit K-Means initialization; and known context-conditioned transition laws are recoverable with explicit transition capacity/refinement while preserving a near-zero null effect.

These results do **not** establish domain semantics or downstream predictive value.

### Reproduction

```bash
python scripts/validate_duration_context.py --workers 4 --output /tmp/nhsmm-duration-context.json
python scripts/validate_state_recovery.py --scenario all --workers 1 --output /tmp/nhsmm-state-recovery.json
python scripts/validate_learning_prediction.py --scenario all --seeds 401,402,403,404,405 --max-iter 40 --workers 4 --output /tmp/nhsmm-learning-prediction.json
python scripts/validate_transition_context.py --scenario all --workers 1 --output /tmp/nhsmm-transition-context.json
```
