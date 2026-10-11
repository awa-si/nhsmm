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

### Bounded D+ duration-tail recovery and fallback

The advanced-duration harness compares `duration_tail=True` against the historical finite-support model with all other model/training settings matched. It uses causal `K=2`, `D=8`, eight deterministic seeds (`321..328`), disjoint OOS sequences, and survival forecasts at horizons `1,3,5,8`.

The `true_tail` generator places about 59% of completed sampled durations above `D` with mean duration about `11.11`. The `finite_control` generator has no duration above `D` and mean duration about `5.00`.

Observed summaries:

| Scenario | Positive OOS LL gain | Median OOS LL gain | Positive survival Brier gain | Median Brier gain | Positive survival MAE gain | Median MAE gain | Median learned h_tail |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| true_tail | 6/8 | +0.00219 | 8/8 | +0.05844 | 8/8 | +0.01630 | 0.2460 |
| finite_control | 6/8 | +0.000075 | 0/8 | -0.000093 | 2/8 | -0.000185 | 1.0000 |

Acceptance is intentionally mechanism-specific. True-tail data must show broad survival-calibration improvement, non-negative median held-out likelihood gain with at least 5/8 positive seeds, and an activated tail hazard. The finite control must return to `h_tail>=0.99` and remain within `0.001` median drift in likelihood, MAE, and Brier score.

Acceptance: **PASS**.

This supports the narrow claim that the bounded `D+` extension can learn/use genuine beyond-`D` duration structure while collapsing back to the historical finite-support semantics when the data contain no tail. It is not evidence that real market regimes require a duration tail or that enabling the option improves trading utility.

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

### 2026-10-07 causal-context regression repair revalidation

After restoring raw causal per-timestep encoder features for HSMM component modulation, the maintained duration- and transition-context acceptance harnesses were rerun on commit `9c4429b`. No acceptance threshold was changed.

Duration context again passed all maintained gates: strong `15/15` positive, moderate `13/15` positive with median gap `0.02716`, and the null guard passed with median gap `0.01138` and mean gap `0.01266`.

Transition context also remained accepted across all `15` seeds per scenario: strong median delta correlation `0.999987`, moderate `0.993766`, and null median absolute learned context delta `6.90e-7`; all scenarios were `15/15` non-collapsed.

This revalidation confirms that the causal sequence-context repair restores the maintained duration benchmark without weakening transition-context recovery or its null guard.

### External implementation cross-check

On 2026-10-07 the maintained non-causal explicit-duration inference path was cross-checked against the independently developed `gaussian-hsmm 0.1.0` implementation.

The comparison deliberately bypassed both training procedures and used matched fixed parameters. The external package's own explicit-duration `score()` and `decode()` dynamic programs were compared with NHSMM forward likelihood and non-causal Viterbi. Twenty deterministic random cases used `K=3`, `Dmax=5`, `F=2`, and `T=11`, with identical initial distributions, zero-diagonal boundary transition matrices, arbitrary positive duration distributions over `1..Dmax`, and diagonal-Gaussian emission log probabilities.

Results:

| Measure | Result |
| --- | ---: |
| Cases | 20 |
| Exact Viterbi path matches | 20/20 |
| Viterbi mismatches | 0 |
| Max absolute log-likelihood difference | 2.84217094304e-14 |
| Median absolute log-likelihood difference | 3.5527136788e-15 |

The external package imports `hmmlearn` for its `fit()` initializer, but its fixed-parameter `score()` and `decode()` paths use its own explicit-duration HSMM inference. The cross-check therefore tests inference semantics rather than optimizer agreement.

The maintained reproduction harness is `scripts/validate_external_hsmm_reference.py`; it emits a JSON summary and fails non-zero if any Viterbi path differs or the maximum absolute log-likelihood difference exceeds `1e-12`. The external package is deliberately not a core/test dependency.

Claim boundary: this supports the shared stationary non-causal explicit-duration HSMM core only. It is not evidence for NHSMM causal dynamic hazard, context-dependent duration/transition behavior, Neural ContextEncoder semantics, training recovery, or real-market utility.

### Interpretation

Current controlled evidence supports four narrow package claims: duration-context ordering is recoverable without spurious null separation; identifiable latent states are recoverable up to permutation with explicit K-Means initialization; fitting changes model parameters and improves held-out density/state recovery on the maintained strong/moderate/weak synthetic contracts while the null control does not support a state-usefulness claim; and known context-conditioned transition laws are recoverable with explicit transition capacity/refinement while preserving a near-zero null effect.

These results do **not** establish domain semantics or downstream predictive value.

### Reproduction

```bash
python scripts/validate_duration_context.py --workers 4 --output /tmp/nhsmm-duration-context.json
python scripts/validate_duration_tail.py --workers 4 --output /tmp/nhsmm-duration-tail.json
python scripts/validate_state_recovery.py --scenario all --workers 1 --output /tmp/nhsmm-state-recovery.json
python scripts/validate_learning_prediction.py --scenario all --seeds 401,402,403,404,405 --max-iter 40 --workers 4 --output /tmp/nhsmm-learning-prediction.json
python scripts/validate_transition_context.py --scenario all --workers 1 --output /tmp/nhsmm-transition-context.json
```

## Trading feature-redundancy pilot — 2026-10-11

This is a short diagnostic on the corrected synthetic episode-end-context dataset,
not a full recovery acceptance run or real-market evidence. It uses core commit
`392af2b2032f1fd1d7c1f561216ce72bf9dcb405`, the corrected variance-aware
initializer, causal filtering, seeds `901,902,903`, eight training windows and
four OOS windows of 240 rows, four joint iterations and two transition-refinement
steps. Dataset SHA256:
`16cab157954b32f11af274bd617d6811a3776e81eb926a3f1dae80c954d7a1f8`.

Three arms were fixed before scoring: the original eight features; four features
(`log_return_1`, `trend_48`, `ewma_vol_12`, `volume_z_48`); and all eight
features transformed with training-only ZCA whitening. Standardization, context
scaling, whitening mean and covariance use chronological training windows only.
The whitening eigenvalue floor is `0.001`; the observed smallest eigenvalue was
`0.058777`, so this floor was inactive. The frozen transform acts pointwise on
OOS observations. Labels are used only for permutation-invariant scoring.
Dimensions and initial clustering differ across arms despite matched seeds and
iteration budgets.

All entries below are medians across the three fits, evaluated on the same four
OOS windows. Those windows contain eight true state switches; this small boundary
sample limits the conclusions.

| Arm | Accuracy | ARI | Boundary F1 ±5 | Run-duration relative error | Predicted switches | Comparable 8D LL / row |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full eight features | 79.69% | 0.5650 | 0.1449 | 90.97% | 61 | -8.0721 |
| Four fixed features | 66.35% | 0.3982 | 0.0828 | 93.10% | 137 | n/a |
| ZCA eight features | 63.85% | 0.3974 | 0.1096 | 89.09% | 138 | -7.3775 |

All arms predict all four states in every seed. The duration metric measures
state-run lengths, not latent episode durations, and remains far from the
recovery target. The full-eight control reproduces the preceding corrected
initializer pilot exactly.

Each model uses a normalized Gaussian likelihood. For ZCA, the reported
eight-dimensional likelihood adds `log|det(W)| = 2.671696584689829` per row
to the transformed-space likelihood. This compares full8 and whiten8 in the same
original standardized observation space. The four-dimensional density is not
comparable and is explicitly recorded as null in that comparison. Neither
eight-dimensional value includes the common initial standardization Jacobian
back to raw feature units.

Whitening reduces maximum absolute pooled training correlation from `0.867558`
to `3.08e-9`; the reduced subset reaches `0.252714`. Despite this reduction,
both alternatives decrease median recovery and increase predicted switching.
Whitening improves normalized OOS density fit by about `0.695` per row while
reducing state accuracy by about 15.8 percentage points. Pooled decorrelation
alone therefore does not solve this pilot's regime identification or
fragmentation problem. It does not establish conditional independence within
states or remove temporal dependence. These short fits do not prove the same
ordering after longer training.

No arm is selected for deployment, no package default changes, and no core code
changes. Repeated exploratory evaluation on these OOS windows is not a fresh
confirmatory holdout. Full corrected-dataset recovery acceptance remains open;
post-tail-fix statistical revalidation also remains open after the separately
documented 300-second timeout. Historical package acceptance is distinct from
these pending revalidations.

Execution: AWA job `66796aa0b7cf40b594250b24041fd0e2` completed successfully
in `65.506` seconds, including runner overhead, under a hard 300-second timeout.
The preceding two-arm training/filtering smoke passed in `24.219` seconds.
An initial smoke invocation supplied two context coordinates where the benchmark
requires one; it failed validation and was corrected before the pilot.
The new regression tests cover training covariance normalization, pointwise
prefix invariance, an independent Gaussian change-of-variables density identity,
and finite invertible regularization of singular covariance.

Verification: all 11 focused tests in the new feature diagnostic and the existing
variance-initialization suites passed; Ruff, Black and diff-check passed. The full
package suite was not rerun for this script-only change.

Reproduction, with the same bounded AWA environment and single-threaded workers:

```bash
PYTHONPATH=/workspace OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /work/env/nhsmm/bin/python scripts/diagnose_trading_feature_redundancy.py \
  --dataset /data/workspace/nhsmm-trading-regimes-v2/trading_regimes.npz \
  --output /data/workspace/stage2/feature-redundancy-v2-2026-10-11.json
```

Machine-readable per-seed scores, fitted whitening transform and execution
provenance: [results/feature-redundancy-v2-2026-10-11.json](validation/results/feature-redundancy-v2-2026-10-11.json).

## Trading training-budget pilot — 2026-10-11

The predeclared comparison changes joint training `max_iter` only: 4, 12 and 30.
Every arm is fitted independently with the same seed reset; transition-only
refinement stays at two steps. Features, training-only scaling, corrected
`kmeans_cluster_variance` initialization, context, learning rate and optimizer
policy are fixed. All arms use eight training windows and four OOS windows of
240 rows on the corrected episode-end-context synthetic dataset. Seeds are
`901,902,903`; the source base is
`da9682522dd9213917753df7b5581d0c1fccaea4`. Dataset SHA256 remains
`16cab157954b32f11af274bd617d6811a3776e81eb926a3f1dae80c954d7a1f8`.
Labels enter scoring only, never training, stopping or budget selection.

Causal filter results below are medians across the three seeds:

| Joint iterations | Accuracy | ARI | Boundary F1 ±5 | Run-duration error | Switches | Train LL / row | OOS LL / row |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 79.69% | 0.5650 | 0.1449 | 90.97% | 61 | -8.1461 | -8.0721 |
| 12 | 78.85% | 0.5567 | 0.1471 | 90.00% | 60 | -8.0423 | -8.0887 |
| 30 | 79.17% | 0.5644 | 0.1563 | 90.34% | 56 | -7.8637 | -8.1520 |

The same evaluation windows contain eight true state switches. All nine fits
cover all four states and match exactly five of the eight true boundaries within
±5 rows. The modest F1 increase comes from fewer false switches, without an
increase in matched boundaries. The four-iteration control reproduces the
preceding corrected-initializer and full-feature pilots exactly.

From 4 to 30 iterations, training likelihood improves in all three paired seeds,
while OOS likelihood and state accuracy worsen in all three. Median switching
decreases only from 61 to 56 and run-length error remains about 90%. On this
limited data coverage, simply adding training iterations does not recover the
regime structure; the training/OOS divergence is consistent with overfitting.
It does not prove what would happen with more training windows or independent
data. No stopping rule or deployment budget is chosen from these held-out scores.

At zero standardized context, learned episode medians move from 31–32 rows at
four iterations to 20–27 at thirty. Duration parameters therefore do move;
extra optimization does not freeze them at the initializer. These episode-PMF
medians at one reference context must not be equated with dynamic observed
same-state run lengths, which can combine self-transition episodes.

Retrospective Viterbi remains a separate diagnostic. Across budgets 4/12/30 its
median accuracy is 81.25%/80.73%/80.62%, boundary F1 is
0.3182/0.2667/0.2400, and switches are 36/37/42. Longer training does not resolve
its boundary problem either. Full-window MAP uses future rows and cannot
substitute for the live causal filter.

This changes a script and evidence/documentation only. Package core, generic
API and defaults are unchanged. Even the 30-iteration arm is not a full recovery
protocol: data coverage and transition refinement remain reduced, no
no-context/HMM/no-regime comparison is run, and transition recovery is not
scored here. Full corrected-dataset recovery acceptance and post-tail-fix
statistical revalidation remain open. Repeated exploratory use of these windows
is not independent confirmatory evidence.

Execution: AWA job `2d6b2d6879b74d318b0cbf3c6dadeefd` completed in
`143.213` seconds under a hard 300-second timeout. Syntax, CLI and a one-seed,
one-iteration training/inference smoke passed first in `34.017` seconds.
The existing benchmark-metric and decoder/duration regression suites passed
`16/16`; their read-only pytest-cache warning did not affect execution.
Ruff, Black and diff-check passed. The full package suite was not rerun because
package training/runtime code did not change.

Reproduction with the same bounded AWA environment:

```bash
PYTHONPATH=/workspace OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /work/env/nhsmm/bin/python scripts/diagnose_trading_training_budget.py \
  --dataset /data/workspace/nhsmm-trading-regimes-v2/trading_regimes.npz \
  --output /data/workspace/stage2/training-budget-v2-2026-10-11.json
```

Per-seed results, episode diagnostics and execution provenance:
[training-budget-v2-2026-10-11.json](validation/results/training-budget-v2-2026-10-11.json).

## Trading causal boundary-evidence pilot — 2026-10-11

This diagnostic separates marginal argmax switching from model probability of
a latent transition. It fits the same eight-feature corrected synthetic data,
variance-aware initializer and training-only scaling as the preceding budget
pilot. Seeds `901,902,903`, joint budgets `4,30`, two transition-refinement
steps, eight training and four OOS windows of 240 rows are fixed. Source base:
`0b8e942c6b57c59b1a462442327574b2258944ef`. Dataset SHA256:
`16cab157954b32f11af274bd617d6811a3776e81eb926a3f1dae80c954d7a1f8`.
Both arms reproduce the preceding accuracy, boundary F1 and switch counts.
Labels enter scoring/event grouping only; they do not enter model fitting or
decoder/threshold selection.

For edge `t-1 -> t`, the prior uses the state/age posterior and context at
`t-1`. Boundary mass is then Bayes-updated with the emission likelihood at
`t`, without any later row. Self-transition episode resets count as episode
ends but not state changes. The same decomposition also recovers
`P(z_(t-1) | observations through t)`: the new observation may revise belief
about the previous state instead of supporting a transition to the new state.

The Float64 edge decomposition using the public transition forecast matches
production Float32 state
posteriors with maximum absolute error `1.90735e-6` and age-one episode-reset
mass with maximum error `8.74025e-7`, across every evaluated edge in all six
fits. The declared comparison tolerance is `5e-5`. Age-one identifies a reset
here because `D=64>1`; the D+ bucket caps reported age means at 64 and is not
an exact age measurement beyond that.

Primary causal-filter results, with medians across seeds:

| Joint iterations | Argmax switches | Unmatched switches | Median prior state-change probability at unmatched switches | Median updated state-change probability | Median current-state confidence |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 61 | 56 | 1.41% | 6.21% | 91.25% |
| 30 | 56 | 51 | 2.68% | 15.67% | 94.32% |

The evaluation windows have eight true state boundaries and eight true episode
boundaries; there are no same-state episode resets in this particular OOS
sample. Each fit matches five state boundaries under maximum-cardinality
one-to-one ±5 matching. Unmatched means unmatched under that scoring contract,
not necessarily a fully independent or instantaneous ground-truth event.

Pooling predictions across seeds, while retaining the same evaluation windows:

| Joint iterations | Unmatched events | Updated change probability <50% | Previous-state posterior revised toward current selected state |
| --- | ---: | ---: | ---: |
| 4 | 170 | 118 (69.4%) | 102 (60.0%) |
| 30 | 153 | 99 (64.7%) | 87 (56.9%) |

These two subsets overlap and must not be added. The 50% split is descriptive,
not a decoder rule or an acceptance threshold. The repeated seed predictions
are not independent new data or evidence for statistical significance.

The high confidence in the current state does not imply high confidence that
a latent switch just occurred. Most unmatched hard switches have less than
50% updated transition evidence, and many involve revision of the previous
state posterior. Marginal argmax labels taken at successive information sets
thus reveal many decision flips with weak transition evidence; successive
marginal labels are not guaranteed to be a jointly supported or MAP latent
path. Independent tests construct an explicit argmax flip with zero
possible episode-end/state-change probability.

This does not explain every error. The remaining 52/170 and 54/153 unmatched
events have updated change probability at least 50%; model/feature
identification and timing errors remain. At the exact eight ground-truth
boundary rows, all fits have updated change probability below 50%, even though
five boundaries match within ±5. Exact timing and tolerant matching are
different claims. A low pre-observation forecast alone is not evidence against
a genuine transition: the tests also show an unlikely forecast becoming
strong transition evidence after the new observation.

The online probability sums are stored for inspection, but each edge uses a
different information set. They are not a common full-window posterior expected
path count and cannot be treated as a replacement boundary decoder. Window
starts force initial age one; row zero is excluded from edge analysis. Of pooled
unmatched events, 71/170 (41.8%) at four iterations and 63/153 (41.2%) at thirty
occur at row 64 or later. The errors are not confined to the first 64 rows,
but this post-hoc position check does not eliminate window-initialization effects.

No production decoder, core recursion, training policy or package default is
changed. This pilot supports investigating a coherent causal decoding contract
before interpreting raw marginal switches as recovered durations. It does not
establish that a probability gate or confidence threshold would improve
recovery. Full corrected-dataset recovery acceptance and post-tail-fix
statistical revalidation remain open.

Execution: AWA job `bfd20c7220bd4115af449b92c77a8f81` completed successfully
in `107.009` seconds under a hard 300-second timeout. Syntax, CLI and the
one-seed/one-iteration smoke passed before the pilot; the successful smoke
and unit-test job took `33.994` seconds. All `28` focused tests passed in
the new diagnostic plus maintained filtering/transition suites. Ruff, Black
and diff-check passed. The full package suite was not rerun because the change
is confined to a script, tests and evidence/documentation.

Reproduction with the same bounded single-threaded AWA environment:

```bash
PYTHONPATH=/workspace OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /work/env/nhsmm/bin/python scripts/diagnose_trading_boundary_evidence.py \
  --dataset /data/workspace/nhsmm-trading-regimes-v2/trading_regimes.npz \
  --output /data/workspace/stage2/boundary-evidence-v2-2026-10-11.json
```

Per-event state/age evidence, per-seed groups and execution provenance:
[boundary-evidence-v2-2026-10-11.json](validation/results/boundary-evidence-v2-2026-10-11.json).


## Trading causal decision-rule pilot — 2026-10-11

This matched exploratory batch follows the causal boundary-evidence diagnostic.
Source base: `47919056d4ef4707ef3207ab4f8a337de2e0acb2`. Corrected
episode-end dataset SHA256:
`16cab157954b32f11af274bd617d6811a3776e81eb926a3f1dae80c954d7a1f8`.
Seeds 901/902/903, 4/30 joint iterations, two transition-refinement steps,
eight training and four OOS windows of 240 rows, eight features, training-only
scaling and variance-aware K-Means starts are unchanged. Every fitted model is
evaluated with all four rules. No label enters fitting or rule selection.

Rules are fixed before the run:
- Marginal argmax: mode of the causal state posterior.
- Prefix MAP endpoint: current endpoint of the highest-probability joint
  state/age/episode path through the current row. Uses max-product with the same
  explicit-duration hazards, previous-row context, self-transitions and D+ tail.
  Successive emitted endpoints need not constitute one joint MAP path: a later
  observation can revise the best history.
- Committed state: sum-product filtering conditional on every previously
  emitted state, then greedy current-state marginal selection and conditioning
  on that selection, while retaining the selected state's age distribution.
  This gives a supported emitted state history but can lock in erroneous
  earlier decisions; it is neither full-path MAP nor the unconditional filter.
  If the committed history loses all support, the diagnostic raises explicitly.
- Full-window Viterbi: retrospective control; not available at live decision time.

All three causal rules use zero delay and no future row. No probability gate,
confidence threshold, score weight or acceptance threshold is tuned. Ties use
the first flattened index. Window starts retain the preceding pilot's age-one
initialization. Run-duration error describes hard state runs, not directly the
learned episode-duration law.

Medians across three seeds; there are eight true state changes in these OOS
windows:

| Iterations | Rule | Accuracy | ARI | Boundary F1 ±5 | Run-duration error | Switches |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 4 | Marginal argmax | 79.69% | 0.5650 | 0.1449 | 90.97% | 61 |
| 4 | Prefix MAP endpoint | 79.27% | 0.5573 | 0.1449 | 90.34% | 61 |
| 4 | Committed state | 78.12% | 0.5336 | 0.1404 | 86.73% | 49 |
| 4 | Full-window Viterbi | 81.25% | 0.5936 | 0.3182 | 82.38% | 36 |
| 30 | Marginal argmax | 79.17% | 0.5644 | 0.1562 | 90.34% | 56 |
| 30 | Prefix MAP endpoint | 78.65% | 0.5555 | 0.1515 | 90.34% | 58 |
| 30 | Committed state | 77.08% | 0.5121 | 0.1509 | 87.66% | 45 |
| 30 | Full-window Viterbi | 80.62% | 0.5953 | 0.2400 | 84.32% | 42 |

Prefix MAP endpoints do not reduce fragmentation. Committing past decisions
reduces switch counts by about 20% but lowers accuracy and does not improve
boundary F1. Even the retrospective control remains below boundary F1 >=0.45
and run-duration error <=0.50. This batch provides no basis to promote either
zero-lag causal rule. A supported history alone is insufficient for recovery.
The earlier revision diagnosis remains valid, but removing that freedom with
greedy commitment does not solve the underlying model/observation problem.

Likelihood belongs to the unchanged model and is not separately redefined for
these rules. Marginal-filter results reproduce the preceding matched pilot.
The same small OOS windows have been inspected repeatedly; this is exploratory
evidence, not independent confirmatory validation or statistical significance.
Core, public decoder, training policy and defaults are unchanged. Full corrected
dataset recovery acceptance and post-tail-fix statistical revalidation remain open.

Verification: nine new tests compare against independent exhaustive
probability-space path enumeration (including D=1, K=1, self-transitions,
finite/tail support), plus prefix invariance, batch equivalence, supported-history
behavior and explicit impossible-support failure. All 14 focused decision-rule
and boundary-evidence tests passed. Syntax and CLI checks passed; the one-seed,
one-iteration end-to-end smoke completed in 34.891 seconds. The complete batch
was AWA job `8f4c4e037a104e44ab46f99c0ed69b79`, exit 0, 117.825 seconds under
a hard 300-second timeout. Full package tests are not rerun for this script/test/
documentation-only change.

Reproduction:

```bash
PYTHONPATH=/workspace OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /work/env/nhsmm/bin/python scripts/diagnose_trading_causal_decisions.py \
  --dataset /data/workspace/nhsmm-trading-regimes-v2/trading_regimes.npz \
  --output /data/workspace/stage2/causal-decisions-v2-2026-10-11.json
```

Per-seed scores, emitted paths, prefix endpoint ages and execution provenance:
[causal-decisions-v2-2026-10-11.json](validation/results/causal-decisions-v2-2026-10-11.json).
