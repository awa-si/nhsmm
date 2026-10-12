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


## Trading observation-control pilot — 2026-10-12

Source base: `ce41788846b72e7740ffbde134462d7d6007edef`.
Original corrected dataset SHA256:
`16cab157954b32f11af274bd617d6811a3776e81eb926a3f1dae80c954d7a1f8`.
Same frozen latent states, episode ages and macro context, chronological windows,
eight features, training-only scaling and variance-aware K-Means setup as the
preceding pilot: eight training/four OOS windows of 240 rows, optimization seeds
901/902/903, four joint iterations and two transition-refinement steps.

Four arms were fixed before execution:
- Trading: unchanged original observations; exactly reproduces the prior
  marginal-filter summary.
- Matched IID: diagonal-Gaussian observations generated using per-regime means
  and variances estimated from labeled training windows only; variances have a
  1e-8 floor.
- Matched AR(1): same population Gaussian marginals, with standardized residual
  correlation rho=0.9. Both matched arms use the same base Gaussian innovations.
  Residuals begin stationary. Finite-sample moments and separately fitted
  training scalers may differ.
- Separated IID: fixed four-state one-hot codebook with amplitude 4 in the first
  four of eight features, independent Gaussian noise standard deviation 0.35.

Labels explicitly define the synthetic generating laws and are used for scoring;
they are never passed into optimize or used to choose a decoder. Matched moments
are a supervised diagnostic construction, not an unsupervised learned
transformation or a deployable trading feature pipeline. The synthetic controls
therefore are not new trading-recovery acceptance evidence.

Medians across optimization seeds; every arm has eight true state changes:

| Observation arm | Decoder | Accuracy | ARI | Boundary F1 ±5 | Run-duration error | Switches |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Trading | Causal filter | 79.69% | 0.5650 | 0.1449 | 90.97% | 61 |
| Matched IID | Causal filter | 96.46% | 0.9189 | 0.3721 | 81.25% | 35 |
| Matched AR(1) | Causal filter | 78.44% | 0.6147 | 0.2400 | 82.31% | 42 |
| Separated IID | Causal filter | 100.00% | 1.0000 | 1.0000 | 0.00% | 8 |
| Trading | Retrospective Viterbi | 81.25% | 0.5936 | 0.3182 | 82.38% | 36 |
| Matched IID | Retrospective Viterbi | 99.27% | 0.9817 | 0.9412 | 2.14% | 9 |
| Matched AR(1) | Retrospective Viterbi | 80.42% | 0.6492 | 0.4000 | 55.31% | 27 |
| Separated IID | Retrospective Viterbi | 100.00% | 1.0000 | 1.0000 | 0.00% | 8 |

All three separated-IID fits recover the state sequence and its hard run lengths
exactly. This demonstrates that fragmentation is not unavoidable in the current
filter/training path under very strong independent state evidence.

The paired matched-IID/AR(1) contrast supports investigating observation temporal
dependence: introducing rho=0.9 lowers median causal accuracy from 96.46% to
78.44%, and retrospective boundary F1 from 0.9412 to 0.4000. It is not a universal
per-seed effect. Matched-IID causal accuracy is 98.96%, 75.52%, 96.46% for seeds
901/902/903; the corresponding AR(1) values are 78.44%, 78.12%, 82.29%.
Thus initialization/local-fit sensitivity remains important even for IID Gaussian
observations. Causal matched-IID recovery still has many extra switches and fails
median boundary/duration thresholds.

Replacing Trading with matched IID simultaneously changes non-Gaussian tails,
within-row feature dependence and temporal dependence. That comparison alone
does not isolate rolling-feature correlation, and this batch does not prove that
rho=0.9 describes actual Trading residuals.

Perfect hard state runs do not establish recovered episode-duration parameters.
Separated-IID learned episode medians at zero standardized context remain 32–34
bars in every fit, near the initial duration law. They are reported in learned
state order in JSON. The original duration families are non-geometric and their
tail is not exactly the model's geometric D+ law; unchanged window initialization
also starts age one inside sampled episodes. This is not an exactly model-conform
duration-learning test. Duration learning and transition-law recovery require
their own controlled parameter tests rather than inference from high accuracy.

All optimization seeds share one observation-generation seed (56001) and the same
latent/context trajectory. They measure fitting sensitivity, not independent
dataset replication or statistical significance. Likelihoods are retained only
as within-arm train/OOS density measurements; cross-arm observations differ.
Core, public decoder, training policy and defaults are unchanged. Full corrected
Trading recovery acceptance and post-tail-fix statistical revalidation remain open.

Execution: AWA job `bb8845343a6849df8d08625c745199a1`, exit 0, 103.560 seconds,
hard timeout 300 seconds. Syntax/static and CLI checks passed before the
one-seed/one-iteration four-arm smoke. Smoke plus the six generator/report tests
completed in 76.505 seconds. The combined observation-control, causal-decision
and boundary-evidence suites passed all 20 tests. Tests cover known moment
statistics, training-only moment fitting, population Gaussian marginals and AR(1)
autocorrelation, reproducibility, prefix invariance, invalid support and explicit
missing-state penalties with strict JSON encoding. Ruff, Black and diff-check
passed. No full package rerun is required for this script/test/evidence-only change.

Reproduction:

```bash
PYTHONPATH=/workspace OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /work/env/nhsmm/bin/python scripts/diagnose_trading_observation_controls.py \
  --dataset /data/workspace/nhsmm-trading-regimes-v2/trading_regimes.npz \
  --output /data/workspace/stage2/observation-controls-v2-2026-10-12.json
```

Per-seed scores, learned durations, observation hashes and generator parameters:
[observation-controls-v2-2026-10-12.json](validation/results/observation-controls-v2-2026-10-12.json).


## Model-conform duration-law pilot — 2026-10-12

Audit qualification: the iteration counts below describe requested maximum budgets. The subsequent code audit found that convergence_stop=False was ignored. These historical observations are retained; the native continuation below records actual completed updates.

Source base: `3ec822186eda91a174b5e15bdce72d7e60c02bf1`. This package
mechanism diagnostic removes the Trading features and window-slicing mismatch.
It is not a Trading recovery or full duration-tail acceptance run.

Each of seeds 901/902/903 has independently generated training/test data using
seed offsets +20000/+30000. Six training and four OOS sequences each have 128
rows, start at an episode boundary, and have initial probabilities [0.5,0.5].
Emissions are independent two-dimensional Gaussian observations with means
[-3,0]/[3,0] and noise standard deviation 0.35. The known episode transition
matrix has self probability 0.02 and off-diagonal probability 0.98.

The stationary duration PMFs for buckets 1..8 are:
- State 0: [0.01,0.02,0.04,0.08,0.15,0.20,0.25,0.25].
- State 1: [0.15,0.25,0.25,0.15,0.08,0.05,0.04,0.03].

The finite scenario ends at age 8. In the tail scenario, bucket 8 denotes
P(L>=8), with geometric end hazards 0.12/0.35 and
L=7+Geometric(h). Thus these stationary laws are exactly representable in
the tested causal finite/D+ model. Same-state episode resets are retained.
Every final episode is right-censored in the diagnostic score, even when the
generator's sampled endpoint coincides with the sequence endpoint:
P(L>=observed_length) is used instead of completed-duration mass.

Four fixed arms:
- Production scheduled joint optimize, four iterations.
- Production scheduled joint optimize, 40 iterations.
- Same joint optimizer/initialization/data/lr, 40 iterations, with script-local
  _compute_loss override fixing categorical temperature at 1.
- Duration-only control with 80 iterations at temperature 1, known initial,
  transition and Gaussian-emission parameters frozen, and the soft duration
  gate frozen. Only duration logits and tail hazards learn. No episode labels
  enter marginal-likelihood optimization. Gaussian variance uses inverse
  Softplus so the injected noise law is correct.

All arms use lr=0.008, one K-Means-variance restart and no early stopping or
scheduler. The duration-only arm has twice the iteration budget of the joint
arms; its advantage cannot be assigned solely to frozen nuisance parameters.

Direct distribution distance is total variation (0 is exact, 1 maximal), averaged
over aligned states. The finite masses and geometric tail through length 1024
plus its residual bucket are included. Default inference temperature is 1.
Medians across seeds:

| Training arm | Finite duration TV | Tail duration TV | Finite OOS LL/row | Tail OOS LL/row |
| --- | ---: | ---: | ---: | ---: |
| Scheduled joint, 4 | 0.3101 | 0.3270 | -1.3188 | -1.2418 |
| Scheduled joint, 40 | 0.1924 | 0.2260 | -1.2548 | -1.1986 |
| Fixed-temperature joint, 40 | 0.1794 | 0.2067 | -1.2525 | -1.1884 |
| Duration-only oracle, 80 | 0.1182 | 0.1112 | -1.1749 | -1.1627 |

Every fitted model has 100% causal state accuracy and exactly matches the true
hard state-change count in its OOS sequences. Perfect state/run recovery again
does not establish recovered latent episode-duration parameters.

Duration parameters do learn on these model-conform data. Longer joint training
reduces TV. Fixed temperature gives a modest further TV/likelihood improvement,
but does not uniformly improve all episode diagnostics. In the tail scenario,
median labeled episode-duration NLL changes from 254.6074 under scheduled joint
to 272.6315 under fixed-temperature joint (lower is better); the duration-only
control gives 199.5003. No schedule/default change is justified by this pilot.

Joint models retain high self-transition probabilities: the unweighted mean over
state/age rows after 40 iterations is approximately 0.363–0.375 for finite and
0.413–0.441 for tail, versus known 0.02. This is not a posterior-weighted average.
Duration laws and same-state resets remain coupled in joint learning, while
known transition/emission parameters permit a closer duration-only fit.
This is evidence for further isolation, not proof of a recursion or gradient bug.

For the tail state's better-covered state 0, duration-only learned hazards are
0.1057/0.1107/0.1170, versus true 0.12. State 1 learned hazards are
0.5203/0.1768/0.5000, versus true 0.35. Its complete training D+ episode counts
are only 2/3/0, compared with 18/19/14 for state 0. Counts include lengths >=8;
censored episodes are reported separately. Scarce tail information limits any
state-1 recovery claim. The JSON includes observed episode coverage, means,
hazards, full PMFs, self probabilities by age, true and learned labeled episode
scores, and the counterfactual PMF at the scheduled terminal temperature.

No new mathematical defect is established. The next useful isolation is a
matched-budget duration/transition experiment with adequate tail coverage and
controlled self-transition capacity. Core, public decoder and defaults remain
unchanged; full post-tail-fix statistical acceptance and Trading recovery remain
open.

Execution: AWA job `c74d13fe0508473f89f91c7edde5b272`, exit 0, 111.765 seconds
under a hard 300-second timeout. Syntax, CLI and eight-arm finite/tail smoke
passed first (smoke 34.327 seconds). Twelve focused tests passed after adding
episode-coverage reporting; this includes six independent diagnostic tests and
the maintained duration-prior/tail-projection tests. Ruff, Black and diff-check
passed. Reporting initially used the base interpreter without NumPy after tests
passed; the episode coverage was then regenerated using the existing NHSMM
interpreter. This reporting-only retry did not rerun or change fits/scores.
No full package rerun is required for this script/test/evidence-only change.

Reproduction:

```bash
PYTHONPATH=/workspace OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /work/env/nhsmm/bin/python scripts/diagnose_model_conform_duration.py \
  --output /data/workspace/stage2/model-conform-duration-2026-10-12.json
```

[model-conform-duration-2026-10-12.json](validation/results/model-conform-duration-2026-10-12.json)
retains the exact generative law, per-seed measurements and execution provenance.


## Model-conform duration pilot after audit, native workspace — 2026-10-12

This bounded continuation uses the audited production base
`e52a3f58d1ea0b6bbce49295fd5f030dee17ea0a` in the native OpenAI workspace.
The diagnostic now records observed optimizer iteration counts and fails if a
joint arm does not complete its requested fixed budget. Counts come from finite
entries in production Convergence.scores, excluding final restart scoring;
the oracle loop separately counts completed optimizer steps. Report schema v2
also records the base commit, diagnostic SHA-256 and runtime versions.

Seed 901 was rerun for both finite and D+ scenarios with exactly the generative
law, train/test seed offsets, independent episode starts, terminal censoring,
optimizer settings and four arms of the preceding model-conform pilot.
All joint fits demonstrably completed 4 or 40 updates; both oracle fits completed
80. Runtime: Python 3.12.14, PyTorch 2.14.1+cpu, NumPy 2.3.5; one worker and one
Torch/BLAS thread. The materialized package and diagnostic inputs were retrieved
through the GitHub connector at the immutable audited base. Package source is
unchanged; only diagnostic iteration/provenance reporting and its tests changed.

| Training arm | Observed steps | Finite duration TV | Tail duration TV | Finite OOS LL/row | Tail OOS LL/row |
| --- | ---: | ---: | ---: | ---: | ---: |
| Scheduled joint | 4 | 0.3124 | 0.3276 | -1.3129 | -1.2418 |
| Scheduled joint | 40 | 0.1997 | 0.2266 | -1.2451 | -1.1986 |
| Fixed-temperature joint | 40 | 0.1924 | 0.2132 | -1.2420 | -1.1884 |
| Duration-only oracle | 80 | 0.1182 | 0.1112 | -1.1678 | -1.1627 |

Every arm reaches 100% causal state accuracy and exactly matches OOS hard
state-change counts: 106 finite, 86 tail. Joint duration laws improve with the
larger budget but remain imperfect. For tail data, fixed-temperature training
lowers TV slightly while worsening the labeled episode NLL from 259.3013
(scheduled 40) to 315.4378 (fixed 40); duration-only oracle NLL is 199.5003.
A temperature/default change is therefore still unsupported. The oracle uses
twice the joint budget and known frozen nuisance parameters.

This is one data/fitting seed, not replicated acceptance or evidence of trading
utility. The native runtime differs from the earlier AWA runtime, and earlier
reported iteration counts were maximum budgets. Differences cannot be attributed
solely to the stopping correction. Full post-fix statistical acceptance, the
three-seed continuation and controlled transition/duration identifiability remain
open.

Before the run, syntax/static checks and a native focused suite passed:
**18 passed** (seven duration-diagnostic tests and eleven audit regressions).
The iteration-evidence test checks a real three-step fit, excludes restart
rescoring and rejects a missing recorded update. No full-package rerun was
performed for this diagnostic/test/evidence-only change; the audited base's
787-test CPU gate remains historical AWA evidence.

Reproduction from the repository root:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1   python scripts/diagnose_model_conform_duration.py   --output /tmp/model-conform-duration-post-audit-native.json   --seeds 901 --scenarios finite,tail --joint-iters 40 --oracle-iters 80   --workers 1 --base-commit e52a3f58d1ea0b6bbce49295fd5f030dee17ea0a   --runtime-label openai-native-workspace
```

[Raw schema-v2 results](validation/results/model-conform-duration-post-audit-native-20261012.json)
retain all per-arm measurements, iteration evidence and execution provenance.


## Structural diagnosis of poor recovery — 2026-10-12

Research base: `2fa38999a71f05e9792ff718ea0b8ea26fac0b9e`, native OpenAI
workspace. The diagnosis combines current source inspection, fixed-parameter
counterexamples and analysis of the existing seed-901 fit; it does not change
production parameters, defaults or estimator behavior.

### Historical neural emission normalization defect

`NHSMM._build_sequence_set` at the research base feeds the current observation into its encoder and
then uses the resulting current-row context to parameterize that same
observation's emission. Thus the score has the form
Normal(x; mu(encoder(x)), variance), rather than a density whose conditioning
information is held fixed while x varies. The causal initial-state context can
also contain the first observation. Prefix causality does not establish density
normalization.

A supported injected causal identity encoder gives a two-state, one-row
counterexample using existing Gaussian emission networks, finite parameters,
unit variance, means 0/4 and bounded state-centered context deltas. Numerical
integration over [-12,16] with 28,001 and 56,001 grid points gives **1.0590187**
for the public log_likelihood score exponentiated, while holding emission context
fixed gives **1.0000000**. Grid refinement leaves the discrepancy unchanged.
Positive density outside the interval cannot repair an integral already above 1.
This disproves a general normalized-generative-likelihood claim for the supported
internal-context model family. It is not future lookahead.

The proof uses an injected encoder, not a fit of the built-in CNN/LSTM. The
built-in path shares current-row emission conditioning, so its normalization
requires an independent argument; this evidence does not quantify distortion
for every parameter setting. The current model-conform pilot has its encoder
disabled and is unaffected by this particular counterexample.

Predictive emission parameters based only on past observations or appropriately
held-fixed exogenous covariates would avoid this self-conditioning issue.
An alternative discriminative/energy-model interpretation requires its own
objective and normalization contract. Simply adding a Jacobian to an arbitrary,
noninvertible encoder is not a valid generic repair.

### Episode duration and self resets are not generally identifiable

The allowed D=1 geometric-tail subfamily has observable state-switch hazard
q = h * (1-r), where h is episode-end hazard and r is self-reset probability.
Models (h=0.2,r=0.5) and (h=0.125,r=0.2) both have q=0.1. Their latent
episode means are 5 and 8, but both observable run means are 10. They define the
same observable hidden-state process when emissions/initial probabilities match.

Production evaluation on three length-20 Float64 sequences gives exactly equal
log likelihoods (-102.89941355147715) and maximum filtered-posterior difference
3.33e-16. Even perfect state labels cannot identify which same-state resets
occurred. This is an exact counterexample in an allowed subfamily, not a theorem
that every restricted finite-duration model is unidentifiable.

Johnson and Willsky explicitly distinguish self-reset episode durations from
observable state durations and motivate excluding self-transitions when durations
are intended to model state runs directly. The current transition structures all
allow self-transitions. Excluding resets must be an explicit model contract,
not a silent patch; in particular K=1 and datasets with true self resets require
separate handling.

### Correct hard states do not establish correct run distributions

An independent probability-space absorbing age chain combines each fitted
duration hazard with its age-specific self-reset probabilities. This measures
observable contiguous state runs, rather than individual latent episodes.

For seed 901 scheduled joint training at 40 updates:

| Scenario | Mean latent episode TV | Mean observable run TV | State-1 predicted run mean | State-1 true run mean |
| --- | ---: | ---: | ---: | ---: |
| Finite | 0.1997 | 0.3400 | 5.831 | 3.286 |
| Tail | 0.2266 | 0.2863 | 5.925 | 3.343 |

Despite 100% hard state accuracy on these strongly separated emissions, the
probabilistic duration forecasts are materially wrong. This is more than a
semantic relabeling of otherwise recovered run laws. High joint self-reset
probabilities and unrestricted age-specific transitions give learning multiple
ways to explain persistence.

### Training and inference temperatures differ

Production _compute_loss changes categorical temperature from approximately 1
toward 0.3, while public inference normally uses 1. For the 40-update scheduled
fit, final training temperature is 0.3060042. The same learned duration parameters
have finite/tail episode TV 0.1431/0.1891 at that counterfactual final temperature,
versus 0.1997/0.2266 at inference temperature 1. These numbers hold fitted tail
hazards fixed and compare duration laws only, not a complete alternate decoder.

The trainer therefore optimizes changing distributions and serves a different
temperature at the end. This is a demonstrated mismatch, but fixed-temperature
training alone is insufficient: the matched pilot improves TV slightly while
worsening tail episode NLL. Duration logits and the separately learned sigmoid
duration gate also provide redundant PMF parameterization at a fixed temperature.
The gate's small one-sided smoothness penalty does not identify episode resets.

### Observation assumptions and tail information limit recovery

The historical matched-observation controls give median causal accuracy 96.46%
for IID Gaussian data versus 78.44% after introducing rho=0.9 AR(1) dependence
with the same population Gaussian marginals. Original Trading accuracy is 79.69%.
This supports a temporal-emission misspecification investigation; it does not
prove actual Trading residuals have rho=0.9. Within-row covariance and marginal
non-Gaussianity also remain unresolved. Student-t changes marginal tails but does
not model residual autocorrelation or full feature covariance.

Seed-901 state 1 has only two completed training tail episodes, lengths 10 and 8.
Their tail exposure is four rows, so the complete-tail geometric MLE is 2/4=0.5,
despite generator hazard 0.35. The duration-only fit's 0.5203 is close to this
sample estimate. A separate 600-step labeled likelihood calculation including
terminal censoring also yielded 0.5 for state 1 and 0.1132075 for state 0.
That is diagnostic labeled evidence, not an unsupervised acceptance result.

### Consequences and next controlled work

1. Resolve the internal-context emission likelihood contract before claiming
   general neural generative likelihood or density-calibrated forecasts.
2. Define whether duration means latent episode length or observable regime run;
   compare a self-reset-free run model against the current episode model.
3. Use one inference/training temperature in that controlled comparison, tie
   unnecessary age-specific transitions and fix the redundant duration gate.
4. Compare an independent classical fixed-parameter/EM baseline and a labeled
   censored-duration estimator under matched budgets and adequate per-state tail
   exposure. More optimizer steps cannot resolve exact nonidentifiability.
5. Test past-conditioned AR/full-covariance emissions on residual diagnostics;
   measure held-out predictive density and survival/boundary calibration as well
   as state accuracy.

The verified recurrence/filtering results remain conditional on supplied
distribution scores. They do not prove that observation-derived neural scores
integrate to one, that all generative parameters are identifiable, or that
marginal argmax decisions optimize boundary timing.

Reproducer:
`scripts/diagnose_model_identifiability.py --pilot
docs/validation/results/model-conform-duration-post-audit-native-20261012.json
--output /tmp/structural-diagnostics.json --base-commit
2fa38999a71f05e9792ff718ea0b8ea26fac0b9e` (with PYTHONPATH=.).
It validates the geometric absorbing-chain formula against an analytic law,
asserts likelihood/posterior equivalence and checks density quadrature against
fixed-context controls at two resolutions. Ruff, Black and bounded native
execution passed. No production code or full-package test suite changed.

[Machine-readable structural evidence](validation/results/structural-diagnostics-20261012.json).

Primary references:
- Johnson and Willsky (2013), [Bayesian Nonparametric Hidden Semi-Markov Models](https://jmlr.org/papers/volume14/johnson13a/johnson13a.pdf), section 3.
- McFarland et al. (2011), [Explicit-Duration Hidden Markov Model Inference of UP-DOWN States from Continuous Signals](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0021606), conditional IID emissions and no self-transitions.
- Rojas-Salazar et al. (2021), [HSMMs for high-frequency environmental data](https://arxiv.org/abs/2109.09949), conditional-independence violations.
- Papamakarios et al. (2021), [Normalizing Flows for Probabilistic Modeling and Inference](https://www.jmlr.org/papers/v22/19-1028.html), normalized transformation-based densities.


## Predictive neural repair — 2026-10-12

Step 1.2 replaces causal internal emission self-conditioning with past-only
context and a zero initial-prior context, updates streaming, and versions
artifact timing. Noncausal internal generative likelihood/training is rejected;
retrospective scores retain their separate interpretation.

The identity-encoder counterexample that previously integrated to 1.0590187
now integrates to **1.0** on both 28,001 and 56,001 point grids.
The held-fixed control remains 1.0. Fixed-prefix predictive density and gradient
timing are covered by passing contract tests. Historical measurements above
remain evidence of the original defect.

[Raw repair diagnostic](validation/results/predictive-neural-step12-20261012.json)
records prerequisite base 56c485aaa69fc766596798bd42b737894fa79a55 and runs with
this stage's updated diagnostic and production code. The exact self-reset
counterexample remains: episode means 5 versus 8 produce the same observable
run mean 10 and equal likelihoods. Predictive timing does not resolve this.

Verification: 80 focused passes; broader native verification had 802 passes,
three CUDA skips and one excluded historical frozen-NPZ test whose exact bytes
were unavailable through the connector. See
[scope](testing.md#step-12--predictive-neural-production-migration-2026-10-12).
There is no new statistical fitting or trading acceptance claim.

## Controlled duration and self-reset baseline — 2026-10-12

Research base: 6eeac114228cda26ff969b0ec207a4bd54902e91, native OpenAI
workspace. This experiment isolates reset capacity from encoder, emission,
temperature and duration-gate learning. No production distribution or default
changes. Script owner: `scripts/diagnose_duration_resets.py`.

Each seed (901/902/903) uses six independent 128-row training sequences and four
independent 128-row test sequences. The existing model-conform PMF, Gaussian
means and variance are unchanged. Finite and D+ laws are generated separately,
each with true reset probability 0 or 0.02. Every sequence starts at an episode
boundary; its final episode is scored as right-censored. All marginal fits use
the same observations within each seed/scenario/reset cell; test labels and
episode ledgers never enter fitting. Train/test array hashes are recorded.

All four arms fix the true Gaussian emission law, the 0.5/0.5 initial law and
duration gate 1. They fit duration logits and, for D+, tail hazard at temperature
1, Float64, Adam lr 0.008, gradient clip 5 and 80 actual updates:

| Arm | Boundary transition contract |
| --- | --- |
| age_specific_resets | Free reset probabilities per state and episode age |
| tied_resets | One free reset probability per state, shared across ages |
| no_resets | Exact zero diagonal support; in this K=2 fixture the off-diagonal law is fixed |
| known_transition | True generator boundary matrix frozen |

The zero-support mask and tied-gradient hook are script-only. The tied hook sums
age gradients and applies identical updates to duplicated logits; exact equality
is asserted after fitting. Gradient clipping and parameter coordinates can also
affect optimization, so this is not a proof attributing every difference solely
to parameter count. With true resets, no_resets intentionally imposes a different
run-duration interpretation and is a misspecified episode model.

An independent NumPy scaled probability-space age-chain evaluates every fitted
held-out likelihood. A separate labeled EM estimator allocates short censored
episodes among surviving duration buckets and estimates the tail hazard from
completed ends plus censored survival exposure. It has privileged episode
labels and known transitions; it is not an independent unsupervised EM baseline.
Its unregularized PMF can assign zero mass to unseen held-out durations; complete
episode scores use the existing diagnostic's 1e-300 mass floor. Metrics separate
episode-duration TV from observable-run TV, including residual mass beyond the
run-law horizon. State accuracy alone cannot establish duration recovery.


### Measured reset-capacity results

All **48 marginal fits completed 80 updates**. The maximum independent/model
held-out likelihood difference was **2.665e-15 per row**. Every arm attained
100% filtered hard-state accuracy on these strongly separated Gaussian data.
These are exploratory measurements, not acceptance thresholds.

Medians across seeds 901/902/903:

| Generator | Arm | Episode TV | Observable-run TV | Test log likelihood/row |
| --- | --- | ---: | ---: | ---: |
| Finite, reset 0 | Age-specific | 0.1235 | 0.2103 | -1.1936 |
| Finite, reset 0 | Tied | 0.1170 | 0.2104 | -1.1940 |
| Finite, reset 0 | No resets / known | 0.1143 | 0.1143 | -1.1646 |
| Finite, reset 0.02 | Age-specific | 0.1341 | 0.2017 | -1.2026 |
| Finite, reset 0.02 | Tied | 0.1264 | 0.2044 | -1.2030 |
| Finite, reset 0.02 | No resets | 0.1189 | 0.1211 | -1.1727 |
| Finite, reset 0.02 | Known | 0.1182 | 0.1170 | -1.1749 |
| D+, reset 0 | Age-specific | 0.1651 | 0.1456 | -1.1582 |
| D+, reset 0 | Tied | 0.1545 | 0.1595 | -1.1589 |
| D+, reset 0 | No resets / known | 0.0968 | 0.0968 | -1.1587 |
| D+, reset 0.02 | Age-specific | 0.1744 | 0.1428 | -1.1625 |
| D+, reset 0.02 | Tied | 0.1613 | 0.1530 | -1.1636 |
| D+, reset 0.02 | No resets | 0.1054 | 0.1011 | -1.1623 |
| D+, reset 0.02 | Known | 0.1112 | 0.1086 | -1.1627 |

The no-reset arm improves observable-run TV versus the age-specific arm in all
12 seed/scenario/reset cells. Even on true zero-reset data, free reset fits retain
median reset probabilities 0.2525 (finite) and 0.3410 (D+). Tying ages reduces them
to 0.2178 and 0.2729 but does not close the run-law gap at this budget. This
supports reset/duration optimization coupling as a mechanism in this fixture.
It does not prove that longer or differently initialized fits cannot improve.

A material exception is hidden by medians: seed 903, finite law with true resets
0.02, has an observed same-state test run of length 9. A finite reset-free model
with Dmax=8 cannot represent its true state path. Its test likelihood is
**-1.4224/row**, versus **-1.2047** age-specific and **-1.1820** known transition,
despite all having 100% marginal hard-state accuracy. Likelihood sums coherent
paths; marginal argmax labels need not form a supported path. Removing resets
must therefore be paired with an explicit run-duration contract and adequate
support, not silently imposed on episode-model data.

Tail information remains inadequate: state 1 has 2/4/0 completed training D+
episodes for zero-reset seeds and 2/3/0 with reset 0.02. For seed 903 the labeled
tail hazard stays at its initial 0.5 because it has no observed tail exposure;
that is an unidentified value, not a recovered estimate. The labeled estimator's
run-TV results also vary by seed and are not uniformly better than the finite
optimizer. More samples and explicit per-state exposure controls are the next
duration experiment. An independent unsupervised EM fit and observation-model
experiments remain unexecuted.

[Raw results](validation/results/duration-reset-step2-native-20261012.json)
retain every fitted law, per-age reset probability, held-out score, actual update
count, independent likelihood error, labeled EM history and coverage count.
The completed checks are owned by
[testing.md](testing.md#step-2--controlled-durationself-reset-diagnostic-2026-10-12).
No trading-utility, general recovery, or new full canonical-suite claim is made.

## Fixed tail exposure and run-duration support — 2026-10-12

Research base: fbbff77e177453bed120a4fa6947c5787db23f45. The Step 3 diagnostic
(`scripts/diagnose_tail_coverage.py`) fixes 128 training and 32 independent test
sequences of 128 rows for each of seeds 901/902/903. It preserves each prior
six-sequence training prefix exactly and records its array hash. There is no
adaptive dataset expansion or fit/test-driven sample selection. The true PMF,
emissions, initial law and 0.02 boundary reset probability remain unchanged.
Terminal episodes are always right-censored.

Before any fitting, each tail dataset must contain at least 30 completed D+
episodes and 60 tail trials per state. A trial is a completed tail end or an
observed survival step beyond age D; a censored episode contributes survival
without an end. A failed gate stops the seed. These are exploratory exposure
requirements, not general precision or statistical-acceptance guarantees.

Two comparisons separate the remaining mechanisms:

| Generator | Arms | Comparison |
| --- | --- | --- |
| D+ episodes, real resets | Known transitions with PMF/hazard learned; known transitions and true PMF with only hazard learned | Sampling uncertainty versus duration-PMF optimization |
| Finite episodes, real resets | Reset-free finite D=8; reset-free D=8 plus geometric tail | Observable-run support on identical train/test data |

Each arm fixes the true Gaussian emission and initial laws, uses temperature 1,
duration gate 1, Float64, Adam lr 0.008, gradient clip 5 and 80 actual updates.
Episode labels never enter marginal fitting. True PMF and transition parameters
are privileged nuisance controls; the labeled censored EM remains separate.
This is not independent unsupervised EM or a production default change.

Reset-free models interpret duration as observable state-run length. Their
episode-TV and complete-episode scores against generator episode labels are
therefore omitted from the report. They are evaluated by the independent
absorbing-chain run law and held-out marginal density. D+ gives unbounded
support, but a single geometric tail need not exactly represent the run law
obtained by composing arbitrary reset episodes. Adequate support and accurate
law recovery are separate questions.


### Exposure and measured Step 3 results

All **12 marginal fits completed 80 updates**. All source hashes, prior
six-sequence prefix hashes, and observed update counts were verified. Maximum
NumPy/model held-out likelihood difference: **4.219e-15 per row**.

| Seed | State 0 completed tails / trials | State 1 completed tails / trials | State 1 labeled hazard | State 1 hazard-only marginal fit |
| --- | ---: | ---: | ---: | ---: |
| 901 | 327 / 2712 | 48 / 113 | 0.4248 | 0.4362 |
| 902 | 345 / 3075 | 45 / 130 | 0.3462 | 0.3226 |
| 903 | 328 / 2856 | 43 / 105 | 0.4095 | 0.4106 |

The true state-1 hazard is 0.35. Exposure now meets the specified minimums for
every seed/state, replacing the earlier 2/3/0 completed state-1 tails. It still
does not guarantee precise parameter recovery: the rare tail yields substantial
sampling variation. State-0 labeled hazards are 0.1206/0.1122/0.1148 versus truth
0.12. Known-PMF marginal hazards are 0.1233/0.1088/0.1191. Episode resets remain
unobserved even when Gaussian state labels are effectively certain.

Medians across the three seeds:

| Comparison | Arm | Observable-run TV | Test log likelihood/row |
| --- | --- | ---: | ---: |
| True D+ episodes | PMF + hazard learned, transition known | 0.0711 | -1.1365 |
| True D+ episodes | Only hazard learned, PMF + transition known | 0.0029 | -1.1334 |
| Finite episodes with resets | Reset-free finite D=8 | 0.0883 | -1.4961 |
| Finite episodes with resets | Reset-free D=8 plus geometric tail | 0.1010 | -1.1490 |

The labeled censored estimator's observable-run TVs are 0.0264/0.0184/0.0310
(median 0.0264). That privileged reference is not an unsupervised acceptance
result. More data reduce the earlier coverage deficit, but the 80-update
PMF-learning arm still differs from the true law. The known-PMF and labeled
controls motivate a matched optimizer/independent-EM comparison; they do not
prove a production optimizer bug or establish that every remaining discrepancy
is optimization rather than latent-reset uncertainty or estimation error.
Low aggregate TV with a known PMF also does not imply accurate rare-tail hazard:
the state-1 tail bucket has only 0.03 probability.

On finite-episode data with real resets, all test sets contain observed state
runs above D=8 (14/11/6 runs, maximum length 16). D+ improves held-out likelihood
for every seed: -1.5705 to -1.1490, -1.4961 to -1.1555, and -1.3224 to -1.1449.
Run-TV is slightly worse at this fixed budget in every seed, illustrating why
small mass discrepancies can hide severe density penalties from unsupported
latent paths. Gaussian emissions give alternative paths nonzero density, so
the finite model's observation likelihood remains finite. D+ is a support fix,
not evidence that its fitted geometric run tail is the true composed law.
The finite model's median hard-state accuracy is 99.9756%; D+ is 100%.

Per-seed raw results:
[901](validation/results/tail-coverage-step3-seed901-native-20261012.json),
[902](validation/results/tail-coverage-step3-seed902-native-20261012.json),
[903](validation/results/tail-coverage-step3-seed903-native-20261012.json).
They preserve each fitted law, independent likelihood check, exposure, run
length summary and labeled history. Source provenance refers to this stage's
diagnostic/helper changes atop the recorded base commit.

Final focused verification: **29 passed**, with AST/Ruff/Black green.
Production defaults are unchanged. The historical frozen-NPZ check remains
unavailable and the broad suite was not rerun for these script-only controls.
Next: independent unsupervised EM versus matched gradient fits on this frozen
dataset; observation-model experiments remain pending. No trading or general
statistical acceptance claim.
