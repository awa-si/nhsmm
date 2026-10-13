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

## Independent duration EM versus gradient fitting — 2026-10-12

Step 4 uses the immutable Step 3 D+ train/test datasets, including their
128/32 independent sequences, 128 rows, known Gaussian emissions and initial
law, fixed true 0.02 reset matrix, temperature 1 and duration gate 1. Array hashes
are required to match the preceding committed reports before fitting. No new
data or selection by held-out results. Script owner:
`scripts/diagnose_duration_em.py`.

The independent marginal EM implementation uses NumPy only during fitting.
Its hidden state stores regime, drawn duration bucket and elapsed bucket age.
Finite buckets progress deterministically; the D+ bucket survives geometrically
after age D. Boundary destinations use the known transition matrix, and new
buckets are drawn from the learned PMF. Scaled forward/backward recursion
computes posterior expected bucket draws, tail ends and tail survival steps.
The PMF and hazard have closed-form M-steps. Terminal backward weights are 1:
the final episode is right-censored, with no forced terminal end. Initial bucket
draws and censored final bucket draws contribute to PMF counts. Tail trials count
only transitions between observed rows. EM sees no labels or episode ledger.

The three predeclared arms start from uniform PMFs and tail hazards 0.5:

| Arm | Update budget | Update rule |
| --- | ---: | --- |
| NumPy EM | 80 | Posterior expected counts and closed-form M-step |
| Adam | 80 | Existing diagnostic helper, Float64, lr 0.008, gradient clip 5 |
| Adam longer-budget control | 400 | Fresh fit with the same initialization, larger fixed budget |

Identical initial marginal likelihoods are asserted. The 80-update Adam arm must
reproduce the immutable Step 3 held-out likelihood exactly within 1e-12 per row.
EM records every pre-update likelihood and the final likelihood, asserting
monotonicity. The fitted EM law is loaded into a model only for evaluation;
its training likelihood must agree within 1e-7 in total. Held-out density and
filtered posteriors use the existing public inference path and independent
probability-space likelihood check. Equal update counts do not mean equal
compute; fit timings are recorded and are environment-specific observations.

Five new tests verify expected counts against complete generative path
enumeration for duration widths 1/2/3, including overlapping survival and
self-reset edges, single-row censoring with no tail trial, EM monotonicity,
normalization, and public-model likelihood agreement. This is independent
unlabeled duration fitting with known nuisance parameters, not full unsupervised
recovery of emissions/transitions or evidence that EM is universally preferable.
Production optimizer/defaults remain unchanged.


### Measured Step 4 optimizer results

All nine fits completed their predeclared budgets (80 EM, 80 Adam, 400 Adam
updates per seed). Source/provenance hashes, frozen train/test hashes and the
preceding Adam-80 scores were verified. Every EM history contains 81 likelihoods
and is monotone within 1e-7 total. Independent held-out likelihood agreement:
maximum **4.219e-15 per row**. All arms attain 100% hard-state filtering accuracy.

Medians across seeds 901/902/903:

| Arm | Episode-duration TV | Observable-run TV | Test log likelihood/row | Fit seconds |
| --- | ---: | ---: | ---: | ---: |
| Adam 80 | 0.0723 | 0.0711 | -1.136506 | 16.70 |
| Independent EM 80 | 0.0290 | 0.0284 | -1.133808 | 9.67 |
| Adam 400 | 0.0293 | 0.0287 | -1.133809 | 78.30 |

| Seed | Adam-80 train LL below EM | Adam-400 train LL below EM | EM run TV | Adam-400 run TV |
| --- | ---: | ---: | ---: | ---: |
| 901 | 58.2290 | 0.07163 | 0.02842 | 0.02873 |
| 902 | 71.2599 | 0.03384 | 0.02073 | 0.02072 |
| 903 | 74.7452 | 0.00070 | 0.03244 | 0.03250 |

The larger gradient budget closes nearly all of the training-objective gap and
approaches the independent EM law in every seed. The earlier 80-update
duration-PMF gap therefore has a demonstrated optimization-budget component
in this fixed-nuisance setup. For a concrete example, seed 901's state-0
one-row duration probability is 0.04875 with Adam-80, versus 0.013942 with EM
and 0.013943 with Adam-400 (generator truth 0.01). Perfect hard-state labels
alone never diagnosed that incomplete duration fit.

The EM training likelihood reaches within 1e-7 of its final value by six
updates for seeds 901 and 902; all 80 updates are still executed and recorded.
This favorable speed is specific to strongly separated emissions and known
nuisance parameters. It is not a general comparison of EM and neural fitting
complexity, convergence basins, compute budgets or package defaults.

Residual duration/run error around 0.03 is not eliminated by matching the
optimizer objective. Rare-tail estimates remain sample-sensitive: seed 901's
EM state-1 hazard is 0.4418 versus generator 0.35, while Adam-400 is 0.4418 too.
This separates the closed optimization gap from remaining estimation and
latent-reset uncertainty; it does not prove a general identifiability result.
Known true PMF controls from Step 3 are privileged and should not be treated as
unlabeled achievable recovery guarantees.

[Seed 901 raw results](validation/results/duration-em-step4-seed901-native-20261012.json),
[902](validation/results/duration-em-step4-seed902-native-20261012.json),
[903](validation/results/duration-em-step4-seed903-native-20261012.json)
retain every fitted PMF/hazard, observed update count, EM likelihood history,
fit timing, held-out score and preceding-report hash.

Final native focused gate: **34 passed**; AST/Ruff/Black green. These diagnostic
and test changes leave production optimizer/defaults unchanged. No new full
canonical-suite claim; the unavailable historical NPZ verification remains open.
Next: controlled observation-model/autocorrelation and covariance diagnostics.
Any production budget or initializer change requires a separate matched
full-model study rather than applying this fixed-nuisance result universally.


## Controlled emission autocorrelation and covariance — 2026-10-12

Step 5 base: 22bb15e3abfe994d6acf354bd117f833e98007f1.
Owner: scripts/diagnose_emission_dependence.py. Seeds 901/902/903 each share the
same latent sequences across four observation cases: IID or stationary AR(1)
residuals (rho 0/0.8), with feature correlation 0/0.75. Training uses 128
independent 128-row sequences; testing uses 32 independent sequences.
The known PMF, D+ hazard, initial and 0.02 reset law stay fixed. Observation
means [-0.75,-0.75]/[0.75,0.75] create overlap. Residuals retain unit per-feature
population variance and Gaussian marginals. AR residuals continue across
latent boundaries. Base Gaussian draws and labels are paired across cases;
train/test noise seeds are independent.

Train-only labeled residuals estimate regime means, shared marginal covariance
and pooled scalar rho. Eigenvalues have a fixed 1e-6 floor and rho is clipped
to [-0.95,0.95]. This is privileged supervised nuisance fitting, not unsupervised
training of new emission families. Test labels enter metrics only. Four
likelihood arms combine diagonal/full covariance and IID/AR dependence.
No production or supported-family/default changes.

For the generated AR law, the next mean is
mu_j + rho * (x_(t-1) - mu_i), conditional on previous state i and current state
j. Covariance is (1-rho^2) * Sigma; the first row uses stationary Sigma.
The causal reference propagates (state, age) probabilities with state-pair
emission densities. It sums over unknown previous states, rather than using
their true labels or a hard prediction. Each Gaussian branch is normalized
conditional on the past; normalized transition mixtures preserve normalized
predictive density. A current-state emission row cannot exactly express this
particular switched-mean residual law. Past-conditioned neural context may
approximate dependence but is not automatically equivalent to this edge model.

Seven new tests check stationary variance/covariance/autocorrelation, all-sequence
prefix reproducibility, density against an independent Torch multivariate normal,
exact IID collapse, exhaustive state-path likelihood/posterior agreement,
causal future-mutation/prefix invariance, train-only nuisance fitting and
nonuniform finite/D+ IID agreement with public NHSMM inference. Public NHSMM
canonicalizes observations to Float32: that comparison uses identical rounded
values while retaining the strict 1e-12 Float64 recurrence tolerance.

### Measured Step 5 results

All **48 fixed-parameter filtering comparisons** completed; source hashes were
verified. No iterative dynamics fit or fit-budget change enters the comparison.
Train estimates identify AR cases around 0.794–0.806 and IID cases around
-0.011–0.013. Medians across the three seeds:

| Data | Likelihood arm | Accuracy | Brier, lower better | Log likelihood/row |
| --- | --- | ---: | ---: | ---: |
| IID, diagonal | Diagonal IID | 90.16% | 0.1411 | -3.0875 |
| IID, diagonal | Full AR | 90.14% | 0.1411 | -3.0875 |
| IID, correlated | Diagonal IID | 85.08% | 0.2211 | -2.9970 |
| IID, correlated | Full IID | 85.08% | 0.2054 | -2.6079 |
| AR, diagonal | Diagonal IID | 86.08% | 0.2005 | -3.0174 |
| AR, diagonal | Diagonal AR | 93.24% | 0.1004 | -2.1862 |
| AR, correlated | Diagonal IID | 80.27% | 0.3047 | -2.8636 |
| AR, correlated | Full IID | 80.15% | 0.2797 | -2.5248 |
| AR, correlated | Diagonal AR | 86.57% | 0.2070 | -2.0978 |
| AR, correlated | Full AR | 87.70% | 0.1780 | -1.7078 |

Covariance improves density and probability scores on correlated IID data while
median hard-state accuracy stays unchanged. AR likelihood improves state
accuracy and probability quality when serial dependence is present. On the
combined case, full AR improves held-out likelihood over diagonal IID by
1.1196–1.1798 per row across all seeds, with lower Brier in every seed.
The IID-diagonal null shows essentially no gain. Brier is mean summed squared
error against the true-state one-hot vector; state log score is mean log posterior
probability of the true state. These proper scores assess probability quality,
not a complete reliability/calibration proof.

These controls demonstrate how conditionally IID diagonal emissions can fail
despite correct dynamics. They do not establish actual Trading residual
rho/covariance, prove AR(1) sufficient for markets, or justify a production
family change. The prototype has known dynamics and privileged train-label
nuisance estimates. Marginal Student-t tails alone do not represent the Gaussian
dependence isolated here.

[Raw results](validation/results/emission-dependence-step5-native-20261012.json)
retain source provenance, observation/label hashes, fitted train statistics,
per-seed metrics and medians. Final focused gate: **24 passed**; AST/Ruff/Black
green, 118 materialized Python files unchanged by Black. No new full-suite
claim; the historical frozen-NPZ check remains open.

The research sequence separates repaired causal self-conditioning, episode/run
reset ambiguity, sparse tail exposure, short PMF optimization and controlled
emission-dependence mismatch. Next: diagnose residual dependence on the
maintained Trading feature path before selecting production changes.


## Maintained Trading feature residuals — 2026-10-12

Step 6 base: 800a6e6098f443bc4af946584ecb52bd51764fe3.
Owner: scripts/diagnose_trading_residuals.py. Current v2 generation is repeated
natively at fixed seeds 424242/424243/424244, each with 60,000 rows, start price
50,000 and episode-end transition context. This is the maintained synthetic
Trading generator/feature path, not real-market data or a reconstruction of
the unavailable legacy v1 NPZ. All runs report schema
nhsmm-synthetic-trading-regimes-v2 and feature/label/episode hashes.

The eight canonical benchmark features are used directly. Contiguous ranges
are train [0,36000), validation [36000,48000), test [48000,60000); this residual
study does not use the benchmark's sampled-window training protocol. Means and
scales use training rows only. Training state labels define oracle state means.
Within-episode training pairs fit each feature's own-lag AR(1) and a full VAR(1),
both with intercepts and fixed slope ridge 1e-4. Intercepts are unpenalized.
All fitted parameters are frozen before validation/test residuals are computed.

The variants are state-mean-centered features, diagonal AR(1) innovations and
VAR(1) innovations. Lag pairs at 1/3/12/48 stay inside both split and ground-truth
episode; first episode/split rows have no AR/VAR innovation. Monotone episode IDs
prevent joining recurring regimes across a boundary. Test labels condition
diagnostic summaries only; these oracle controls are not live predictors,
actual fitted NHSMM innovations, or normalized observation likelihoods.
No test/validation-driven parameter or feature selection is performed.

### Measured Step 6 residual evidence

The table reports median absolute test ACF per feature over the twelve
seed/regime cells, giving each cell equal weight:

| Feature | State-centered lag 1 | Own-lag AR innovation lag 1 | VAR innovation lag 1 |
| --- | ---: | ---: | ---: |
| log_return_1 | 0.1505 | 0.0192 | 0.0155 |
| log_return_3 | 0.6737 | 0.1315 | 0.1839 |
| log_return_12 | 0.9168 | 0.1485 | 0.0522 |
| trend_12 | 0.8721 | 0.1570 | 0.0164 |
| trend_48 | 0.9717 | 0.1646 | 0.0226 |
| ewma_vol_12 | 0.8913 | 0.0200 | 0.0165 |
| ewma_vol_48 | 0.9875 | 0.0161 | 0.0132 |
| volume_z_48 | 0.1156 | 0.0195 | 0.0186 |

VAR does not uniformly dominate own-lag AR. More importantly, low lag-1 ACF
does not establish whiteness: after VAR, log_return_3 retains median signed
lag-3 ACF **-0.3193** (all cells -0.3992 to -0.2606); log_return_12 retains
lag-12 ACF **-0.3760** (all cells -0.4815 to -0.3248).
Finite overlapping return windows therefore leave moving-window memory
that a one-lag AR/VAR control does not remove.

| Test statistic, median over twelve cells | State centered | Own-lag AR | VAR |
| --- | ---: | ---: | ---: |
| Maximum absolute cross-feature correlation | 0.8528 | 0.9836 | 0.9972 |

The smallest innovation correlation eigenvalue has median **1.403e-5** after
VAR, versus 0.06972 before AR removal. VAR innovations most strongly correlate
log_return_1 with trend_12 or trend_48. Removing predictable memory exposes
nearly duplicated contemporaneous shocks, so diagonal innovation covariance
is still a poor assumption for this full feature vector. This is conditional
oracle evidence; fitted context/age dependence in a full NHSMM could change
the residual law.

Gaussian residual shape is also incomplete: maximum held-out VAR innovation
excess kurtosis is **304.67**, in seed 424244 bull ewma_vol_48 (3,654 rows).
The state-centered maximum is 21.90. Maxima are descriptive tail-sensitive
sample statistics, not distributional hypothesis tests or universal population
kurtosis estimates. The generator has jumps and nonlinear volatility features.

Source inspection supports the mechanisms: log_return_3/12 use overlapping
price windows, trend_12/48 reuse overlapping price means, and ewma_vol_12/48
filter the same squared returns. A single AR(1) addition does not solve remaining
window memory, innovation redundancy and heavy tails together. Student-t
marginals can address some tails but do not supply cross-feature covariance or
moving-window dependence. Earlier synthetic AR/full-covariance controls
identified those mechanisms separately; this study confirms them on the
maintained synthetic feature path, not on live market residuals.

[Raw results](validation/results/trading-residual-step6-native-20261012.json)
preserve train parameters, all split/state/variant statistics, signed ACFs,
pair counts, covariance spectra and kurtosis. Source hashes were verified.
Final focused gate: **32 passed** across residual, observation, redundancy,
edge-emission and audit tests; AST/Ruff/Black green. Production implementation
and defaults are unchanged; no new canonical full-suite or trading acceptance
claim. The historical frozen-NPZ test remains unresolved.

Next: a matched feature/observation-contract comparison targeting window memory
and duplicated shocks, evaluated on common held-out state/predictive outcomes.
Raw densities across different feature dimensions must not be compared as if
they described the same observation target. Any production emission change
requires an explicit supported model contract and separate inference/runtime
verification.

## Matched Trading feature and emission contract pilot — 2026-10-12

Step 7 base: 836ada51dad9cca04d7420d228afbe93791d01db.
Owner: scripts/diagnose_trading_feature_contract.py. The fixed native v2
dataset at generator seed 424242 must match Step 6's committed feature hash.
Fit seeds 901/902/903 use eight deterministic training windows from [0,36000)
and four test windows from [48000,60000), each 240 rows. The validation segment
is not used to select features, families, hyperparameters or stopping points.
This is a bounded benchmark-window pilot, not the full acceptance protocol or
three independent datasets.

Six predeclared arms compare full8 Gaussian, maintained reduced4 Gaussian,
return_volume2 Gaussian, train-only whiten8 Gaussian, full8 Student-t and
reduced4 Student-t. Reduced4 is log_return_1, trend_48, ewma_vol_12 and
volume_z_48. Return_volume2 keeps log_return_1 and volume_z_48. Whitening uses
the maintained training-only ZCA helper with its fixed eigenvalue floor.
All arms retain K=4, D=64 plus tail, explicit macro context and one restart.
Production optimize executes 40 updates with early stopping disabled; a script
subclass fixes temperature 1 and no transition refinement is applied.

KMeans centers are unlabeled training estimates. Gaussian uses supported
kmeans_cluster_variance. Student-t uses supported kmeans plus a script-only
hook converting cluster variance to the corresponding Student-t marginal
variance at its current degrees of freedom. The hook runs at restart
initialization. Production initialization/defaults are not changed. Matching
base variance does not make the two hypothesis classes or parameterizations
identical.

Common targets avoid comparing densities over different feature dimensions:

| Target | Frozen train-only preparation | Held-out evaluation |
| --- | --- | --- |
| Ground-truth state probability | Best state permutation on training labels only | Accuracy, Brier and state log score |
| Next raw log_return_1 | Ridge mean head on current return and frozen current posterior; posterior-weighted residual variance | Gaussian log score in raw return units and RMSE |

The first 64 rows in every window are a fixed warmup exclusion. State scores use
704 test rows; next-return scores use 700 adjacent pairs, with targets strictly
one row ahead and no crossing of window boundaries. The return head is explicit
supervised downstream calibration; its gradients do not change the HSMM.
It uses only training next-return targets, ridge 1e-3, an unpenalized intercept,
and positive residual variance floors. A current-return-only head supplies the
same-target no-regime baseline. No state labels enter either model training or
return-head inputs; training state labels determine evaluation permutation only.
Input-space likelihoods are deliberately excluded from cross-arm ranking.

State log score uses a 1e-300 floor in Float64, with exact zero true-state
probability counts reported separately. This is a clipped numerical score,
not a claim that zero-probability predictions have finite mathematical log loss.
The full-model pilot also keeps the maintained window episode-start convention;
warmup mitigates but does not prove elimination of mid-episode initialization
effects. No live-market, PnL, production-default or full recovery claim.


### Measured Step 7 common-target results

All **18 accepted fits completed 40 updates**. Source hashes and the frozen
dataset hash were verified. Each arm uses the same 704 state-score rows and
700 next-return targets. The constant-posterior unit control reproduces the
current-return-only baseline, and raw-unit return scores match an independent
Gaussian density. Final focused gate: **35 passed**, with AST/Ruff/Black green.

Medians over fit seeds 901/902/903 on the same test windows:

| Arm | State accuracy | State Brier | Next-return log score | Next-return RMSE |
| --- | ---: | ---: | ---: | ---: |
| Full8 Gaussian | 88.64% | 0.2097 | 4.64173 | 0.00262687 |
| Reduced4 Gaussian | 74.86% | 0.4133 | 4.64013 | 0.00262639 |
| Return/volume2 Gaussian | 38.49% | 0.9495 | 4.59148 | 0.00261696 |
| Whiten8 Gaussian | 69.89% | 0.5541 | 4.60063 | 0.00264010 |
| Full8 Student-t | 87.36% | 0.2278 | 4.64062 | 0.00262645 |
| Reduced4 Student-t | 58.81% | 0.7840 | 4.61344 | 0.00264560 |
| Current-return-only baseline | — | — | 4.51151 | 0.00265486 |

Full8 state accuracy is stable: Gaussian 88.35–88.78%, Student-t
87.36–87.50%. Reduced4 is initialization-sensitive: Gaussian 66.48–86.08%,
Student-t 53.41–84.09%. The earlier residual redundancy result therefore does
not justify indiscriminate removal of predictive state information. Whitening
also does not automatically improve the state representation at this budget.

Full8 Gaussian and Student-t have essentially the same common-return score in
this pilot (median difference 0.00111 nats/target); no statistical superiority
claim is made. Reduced4 can predict returns similarly while reconstructing
the four named regimes less reliably. Return/volume2 has the lowest median
point RMSE but a lower predictive log score and poor named-state accuracy.
State reconstruction, point forecast error and predictive density are distinct
objectives; no single arm dominates them all.

All arms improve the weak current-return-only baseline log score on these
windows. That baseline does not receive the full raw features or macro context,
so this is not evidence that HSMM state structure adds value beyond the same
observable information. The return heads are supervised downstream calibrators,
not native HSMM emission forecasts. Three fit seeds on one dataset are not
three independent data replications, a walk-forward test or a PnL evaluation.
The 40-update budget also does not establish optimizer convergence.

Full8 Gaussian reports one zero true-state probability from Float32 posterior
exponentiation in each seed. Its auxiliary clipped state-log scores are sensitive
to the declared 1e-300 floor and are not used to rank arms in the table.
Accuracy/Brier and common-return metrics retain their explicit separate roles.
A raw log-posterior evaluation would be required for a more precise extreme-loss
comparison.

[Seed 901 raw report](validation/results/feature-contract-step7-seed901-native-20261012.json),
[902](validation/results/feature-contract-step7-seed902-native-20261012.json),
[903](validation/results/feature-contract-step7-seed903-native-20261012.json)
record mappings, fit-step counts, return-head coefficients/variances and all
scores. Production implementation, families and defaults remain unchanged.
The historical frozen-NPZ check remains open; no new full canonical-suite claim.

Next: direct-feature/context baselines with the same information, independent
dataset or chronological-window replication, and raw log-posterior scoring.
Those checks precede any production feature/family/default decision.

## Direct observable baselines and data replication — 2026-10-12

Step 8 base: c2528258a66c5f65399030b711ff5ffab1ba3c96.
Owner: scripts/diagnose_trading_information_baselines.py. Three predeclared
synthetic v2 generator seeds (424242/424243/424244) reproduce Step 6's frozen
feature hashes. Each uses the same Step 7 chronological ranges, eight training
and four test windows of 240 rows and fixed 64-row warmup. Fit seed 901 is held
constant across datasets to isolate data replication from restart variation.
Full8 Gaussian and Full8 Student-t retain the matched Step 7 initialization,
external macro context, K4/D64-plus-tail and fixed 40-update budget. No validation
or test observations select lags, penalties, families or stopping points.

Direct forecast controls receive all eight training-standardized features plus
the training-standardized macro context. Instantaneous designs use the current
row; lagged designs concatenate rows t, t-1, t-3, t-12 and t-48 within each
window. All score the identical 700 raw next-return targets. Both designs have
a homoscedastic and a heteroscedastic Gaussian head. Means use ridge 1e-3 with
an unpenalized intercept. Heteroscedastic heads fit log squared training
residuals with the same ridge and train-only multiplicative smearing. Log
variance predictions are clipped to [-12,12] before exponentiation; normalized
residual variance has a 1e-6 floor. These floors concern the forecast head,
not the state-log metric. Every coefficient and smearing factor is frozen
before test evaluation. Residual calibration is in-sample, not cross-fitted.

To measure incremental representation value, the lagged heteroscedastic design
is augmented with four frozen current HSMM posterior columns, retaining its
observable columns and the identical head rules. State labels do not enter
model training or forecast calibration. Training labels only freeze the state
permutation for evaluation. The old Step 7 posterior/current-return head and
current-return-only control are rerun for continuity.

Observable feature/context access is matched; history representation is not
identical. The fixed lag baseline reaches 48 rows back, whereas the HSMM uses
the entire current window prefix. This is a restricted direct baseline and a
bounded representation comparison, not a universal same-history predictor
benchmark. Extra posterior columns also increase fitted head capacity. Neither
fixed ridge nor 40 model updates establishes convergence or an optimal baseline.

State log scores gather the true class directly from logsumexp over state-age
log posteriors, before exponentiation. No state-log metric floor is applied.
The minimum true-class log probability and number of zeros that Float32
exponentiation would produce are reported separately. This removes the Step 7
scoring-floor ambiguity, while preserving the model's Float32 inference
arithmetic. Accuracy/Brier remain separate objectives from extreme log loss.

This is synthetic data replication over sparse benchmark windows with one fit
seed. It does not establish real-market performance, significance, walk-forward
robustness, duration recovery or trading utility. Production code and defaults
are unchanged; the historical frozen-NPZ verification remains open.

### Measured Step 8 results

All **six fits completed exactly 40 updates**. Each dataset has 704 test state
rows and 700 next-return pairs for every head. All three frozen dataset feature
hashes and recorded source hashes were verified. Focused code gate: **40 passed**;
AST/Ruff/Black passed before execution.

Held-out next-return log scores in raw return units (higher is better):

| Generator seed | Instant hetero | Lagged hetero | Step 7 Gaussian head | Step 7 Student-t head | Lagged + Gaussian posterior | Lagged + Student-t posterior |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 424242 | 4.56062 | 4.59632 | 4.64262 | 4.64064 | 4.59475 | 4.59398 |
| 424243 | 4.69261 | 4.63994 | 4.71519 | 4.71654 | 4.65845 | 4.65818 |
| 424244 | 4.54383 | 4.42515 | 4.49507 | 4.48757 | 4.37525 | 4.38667 |

Adding Gaussian posteriors to the otherwise identical lagged heteroscedastic
head changes log score by -0.00157/+0.01850/-0.04990 across datasets.
Student-t changes it by -0.00234/+0.01824/-0.03848. Both additions worsen
point RMSE on all three datasets. Thus this fixed augmentation does not supply
a stable forecast improvement. Longer fixed lags themselves also fail to
dominate the instantaneous baseline; extra predictors can overfit at this
sample size and fixed penalty.

The simpler Step 7 posterior-weighted variance heads outperform both direct
heteroscedastic designs on datasets 424242/424243, but lose to the instantaneous
direct head on 424244. They use a different mean/variance function class.
Their successes and the failed identical-head augmentation must both be
retained: the experiment neither proves general HSMM value nor proves that
posterior representations are useless. Forecast-head specification and
calibration materially affect the outcome. Full raw scores, point errors and
coefficients are retained in the reports.

Raw held-out state scores with the training-frozen permutation:

| Generator seed | Gaussian accuracy | Student-t accuracy | Gaussian Brier | Student-t Brier | Gaussian state log score | Student-t state log score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 424242 | 88.78% | 87.50% | 0.2062 | 0.2234 | -1.53671 | -1.22326 |
| 424243 | 71.59% | 73.15% | 0.5439 | 0.5196 | -3.18327 | -2.69047 |
| 424244 | 55.26% | 55.11% | 0.8399 | 0.8340 | -4.70676 | -4.12150 |

Named-state recovery is strongly data-dependent at this budget. The stable
fit-seed results on dataset 424242 do not transfer to the other two synthetic
realizations. Student-t improves raw state log score on all three, while
accuracy and forecast density show no uniform family dominance. This is
bounded calibration evidence, not a Student-t default recommendation or proof
of an additional production defect.

On dataset 424242 the Gaussian minimum true-state log probability is
-153.98352. Float32 exponentiation produces one zero, while the raw log score
remains finite. The old clipped Step 7 seed-901 state score was -2.29920;
the corresponding raw score is -1.53671. The difference is the old metric's
1e-300 floor after Float32 exponentiation, not a model improvement. The other
two Gaussian datasets and all three Student-t fits have no such Float32
true-state zeros. Step 7 reports remain historical evidence with their stated
clipped-score contract.

[Dataset 424242 report](validation/results/information-baselines-step8-data424242-native-20261012.json),
[424243](validation/results/information-baselines-step8-data424243-native-20261012.json),
[424244](validation/results/information-baselines-step8-data424244-native-20261012.json)
record frozen head parameters, mappings, actual update counts, provenance and
unfloored state scores. No production change or full canonical-suite claim.
Next: training-only calibration/regularization controls and longer optimization
budgets on predeclared independent windows before any default decision.

## Trading calibration and optimization budget controls — 2026-10-12

Step 9 base: d98678d2b9b3446181ba89e9e91a096a9c072435.
Owner: scripts/diagnose_trading_calibration_budget.py. Generator seeds,
feature hashes, split ranges, window anchors, fit seed 901, feature/context
normalization and all model settings inherit Step 8. Gaussian and Student-t
each execute one uninterrupted 120-update fit per dataset. A script-only hook
captures weights before update 41, after exactly 40 completed updates.
Checkpoint state scores and fixed forecast-head scores must match the committed
Step 8 reports within fixed numerical tolerances before a run is accepted.
Checkpoint and final weights use the same initialization and optimizer trajectory;
production optimizer and initialization code are unchanged.

Three fixed ridge candidates (0.001/0.01/0.1) are chosen only inside the eight
training windows. Four outer folds hold out contiguous pairs of windows.
Within each outer training set, the mean predictor is cross-fitted by window
blocks; each window's residual excludes its own block's return targets from
the mean fit. A ridge regression of log squared out-of-fold mean residuals
fits the variance predictor. Multiplicative smearing is still fitted in-sample
for that variance regression; log-variance clip [-12,12] and normalized
variance floor 1e-6 remain fixed. The candidate maximizing mean outer-fold
raw-unit next-return log score wins, with deterministic grid order for ties.
It is then refitted on all training windows, including new out-of-fold mean
residuals, and frozen before external test evaluation. Ridge selection and
out-of-fold residual calibration change jointly; the experiment does not
isolate their individual contributions.

The nested selection compares direct instantaneous and lagged observable heads,
and the lagged head augmented with each budget's frozen HSMM posteriors.
Each receives all eight features and macro context; lags remain 0/1/3/12/48.
Fixed Step 8 heads and the simpler Step 7 posterior-weighted-variance head are
also evaluated at both checkpoints. All external evaluations use the same
700 next-return targets and 704 state rows.

Cross-fitting applies to the supervised forecast head, not the representation:
HSMM parameters and observable standardization use all training windows.
Held-out head-fold observations can therefore influence the unsupervised
representation. The outer folds estimate conditional head selection only,
not end-to-end out-of-sample representation quality. Training folds can include
later windows than the held-out block and are not walk-forward folds. Neither
the external validation nor test segment participates in this selection.
State labels only determine a training-frozen evaluation permutation.

The recorded training likelihood trajectory is the production monitor's
pre-update likelihood, with a separate likelihood after update 120. More
updates do not establish convergence; likelihood gain need not improve state
semantics or next-return forecasts. This bounded synthetic pilot has one
fit seed, sparse windows and different baseline/HSMM memory representations.
It cannot establish statistical significance, live-market or trading utility.
Production defaults and the historical frozen-NPZ verification are unchanged.

### Measured Step 9 results

All six uninterrupted fits completed **120 updates**; all twelve evaluated
40/120 snapshots have the same 704 state rows and 700 forecast targets.
Every 40-update checkpoint reproduced Step 8's state and fixed-head scores
within the predeclared tolerances. Source, prior report and frozen dataset
hashes were verified. The initial focused gate passed **45 tests**, with
AST/Ruff/Black green before execution.

| Dataset | Family | Accuracy40 | Accuracy120 | State log40 | State log120 | Training LL gain40→120 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 424242 | gaussian | 88.78% | 86.08% | -1.53671 | -2.03524 | 482.29 |
| 424242 | studentt | 87.50% | 80.40% | -1.22326 | -1.86174 | 479.52 |
| 424243 | gaussian | 71.59% | 71.73% | -3.18327 | -2.93544 | 467.02 |
| 424243 | studentt | 73.15% | 72.02% | -2.69047 | -2.61231 | 475.27 |
| 424244 | gaussian | 55.26% | 53.55% | -4.70676 | -5.42184 | 570.83 |
| 424244 | studentt | 55.11% | 52.98% | -4.12150 | -4.61444 | 559.54 |


Training likelihood rises by 467–571 nats from updates 40 to 120 across
the six fits, but state accuracy falls in five of six. The sole gain is
0.14 percentage points for Gaussian on dataset 424243. State log score
improves on that dataset for both families and worsens on the other two.
The last 20 updates still gain 43–84 training nats; convergence is not
established. Increased observation likelihood does not reliably recover the
four named regimes under this feature contract. This supports objective and
representation mismatch as a working hypothesis, not proof of a new
mathematical implementation defect.

All **18 head selections choose ridge 0.1**, the largest predeclared candidate
and 100 times the old fixed penalty. The combined tuned/cross-fitted head
improves predictive log score over the corresponding fixed head in all
12 family/budget snapshots. This identifies sensitivity to forecast-head
regularization/calibration, while not isolating their separate contributions
or establishing a globally optimal penalty. Selection at the grid boundary
is training evidence for a wider future grid, not permission to select a new
penalty from these external test outcomes.

Tuned common next-return log scores in raw units (higher is better):

| Dataset | Instant tuned | Lagged tuned | Gaussian40 | Gaussian120 | Student-t40 | Student-t120 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 424242 | 4.60477 | 4.58847 | 4.60949 | 4.60933 | 4.60900 | 4.61125 |
| 424243 | 4.69552 | 4.67553 | 4.69159 | 4.69073 | 4.69531 | 4.68808 |
| 424244 | 4.55166 | 4.50654 | 4.50222 | 4.49645 | 4.50404 | 4.51983 |


The instantaneous observable baseline has a higher log score than the lagged
baseline on all three datasets. Posterior augmentation beats the tuned lagged
baseline on datasets 424242/424243 at both budgets; it loses on 424244 except
for Student-t at 120 updates. Against the instantaneous baseline, augmented
heads win only on 424242. Point RMSE is worse than the instantaneous baseline
for every augmented family/budget snapshot. Thus the added representation
does not show stable dominance over direct observables, even after head
calibration and a longer training budget.

Increasing model updates improves the tuned forecast log score in only two
of six family/dataset comparisons, both Student-t. The remaining four decline.
The simpler Step 7 posterior-weighted variance head still gives higher log
scores than the tuned augmented head on datasets 424242/424243 at both budgets.
On 424244 the instantaneous direct head wins over every tested HSMM head.
Forecast-head function class, model training objective and named-state recovery
remain distinct parts of the diagnosis; none supports a production-default
change by itself.

[Dataset 424242 report](validation/results/calibration-budget-step9-data424242-native-20261012.json),
[424243](validation/results/calibration-budget-step9-data424243-native-20261012.json),
[424244](validation/results/calibration-budget-step9-data424244-native-20261012.json)
record all training likelihoods, selected penalties, nested fold scores,
head coefficients, permutations, raw state scores and common return metrics.
Production code/defaults remain unchanged. No full canonical-suite, historical
frozen-NPZ or trading-utility acceptance claim.

Next: an age-free HMM control on the same targets to separate explicit-duration
structure from emission/head effects, and a wider training-only penalty grid
motivated by the internal CV boundary result. External test scores must remain
evaluation evidence rather than hyperparameter-selection input.

## Age-free HMM decision control — 2026-10-12

Step 10 base: 64e48f648bfd3a87b500330d3b99806258885d89.
Owner: scripts/diagnose_trading_hmm_control.py. This deliberately narrows the
next research step to a decision about explicit state-age machinery; the
proposed wider ridge sweep is deferred. All three Step 9 datasets, feature
hashes, ranges, window anchors, fit seed 901 and 40/120-update targets are frozen.
The existing Step 9 HSMM reports are immutable comparison evidence, not refitted
or replaced. Their source and Git identities were verified.

The maintained production D1 configuration, with duration tail disabled,
implements an age-free state Markov model. Self-transitions remain allowed;
observable same-state runs can exceed one row even though the model's episode
duration is one. Macro-conditioned transitions remain supported. The control
removes both explicit duration memory and multiple age-specific transition rows.
Its transition probabilities can vary with macro context; this is not a claim
of an unconditional stationary geometric run law.

A script-only initializer reconstructs the Step 9 HSMM restart without training
a prototype, then copies its complete emission and initial-prior parameters to
the HMM restart. This matches centers, family marginal-variance initialization
and emission/initial context networks exactly. Both families use the same
features, macro inputs, optimizer, learning rate, fixed temperature and budget.
Intermediate weights are captured before update 41. HMM transition capacity and
context-network draws differ; its initial effective state persistence is not
matched to the HSMM's duration-induced persistence. The experiment compares
these bounded configurations, not fully identical parameters or a proof that
age structure is necessary/unnecessary in general.

Forecast heads hold ridge 0.1 fixed because all Step 9 internal training
selections chose it. The same training-only out-of-fold mean-residual variance
calibration is used for HMM and HSMM. No new hyperparameter grid, external-test
selection or production-default change is performed. The shared training
representation and non-walk-forward head-fold limitations remain.

An independent NumPy state-only forward/filter recurrence uses the learned
initial, transition and emission scores, including transition row t-1 for the
boundary into t. Production D1 duration log mass must be exactly zero. Every
train/test posterior and public sequence likelihood is compared at both budgets.
Unit tests additionally compare NumPy recursion against complete prefix path
enumeration and verify that the reconstructed initializer equals the real
production optimizer's pre-update checkpoint for Gaussian and Student-t.

### Measured Step 10 results and decision

All six HMM fits completed 120 updates. Twelve 40/120 snapshots retain identical
704 state-score rows and 700 return targets. All 24 independent train/test
recursion checks passed; maximum log-posterior discrepancy is 5.84e-5 and
maximum public likelihood error is 1.96e-6 per row. Source, prior-report and
dataset provenance was verified. Focused gate: **48 tests passed**.

| Dataset | Family | HMM accuracy40 | HMM accuracy120 | HSMM accuracy120 | HMM return log120 | HSMM return log120 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 424242 | gaussian | 87.07% | 84.66% | 86.08% | 4.61177 | 4.60933 |
| 424242 | studentt | 85.65% | 70.03% | 80.40% | 4.61108 | 4.61125 |
| 424243 | gaussian | 70.88% | 71.88% | 71.73% | 4.69482 | 4.69073 |
| 424243 | studentt | 71.16% | 70.74% | 72.02% | 4.69212 | 4.68808 |
| 424244 | gaussian | 55.54% | 54.12% | 53.55% | 4.51648 | 4.49645 |
| 424244 | studentt | 55.82% | 50.57% | 52.98% | 4.50187 | 4.51983 |


HMM minus HSMM return log-score differences over all twelve comparisons range
from -0.01797 to +0.02004 nats per target, with six positive and six negative.
There is no consistent duration-model forecast advantage at these budgets.
Named-state accuracy favors the HSMM in eight of twelve comparisons. The
largest gap is Student-t on dataset 424242 after 120 updates: HMM 70.03%
versus HSMM 80.40%, despite nearly equal forecast density. State naming and
next-return prediction are separate objectives.

Both models remain weak on dataset 424244: HMM state accuracy after 120
updates is 54.12%/50.57% for Gaussian/Student-t versus HSMM 53.55%/52.98%.
The instantaneous observable head beats all compared augmented HMM/HSMM
forecast heads on datasets 424243/424244. Removing explicit duration memory
therefore does not remove the central observed failure. HMM state accuracy
also falls from 40 to 120 updates in five of six fits. Longer likelihood
optimization is not a reliable named-regime repair for either model here.

At 40 updates the HMM raw true-state log score is better in all six comparisons,
even though HSMM hard-state accuracy is usually higher. At 120 updates that
log-score advantage is mixed. More confident state labels, calibrated
probabilities and forecast value must not be collapsed into one acceptance
claim. The reports retain accuracy, Brier, raw log score and return metrics.

**Decision:** retain production families/defaults and stop treating extra
duration optimization or an HMM default switch as the immediate remedy.
The control is useful because the recovery/forecast limitations persist in
both architectures; it does not prove that duration structure has no value.
The working diagnosis remains the observation/feature contract and its
alignment with named regimes, informed by Step 6's overlapping-window memory
and strongly correlated residuals, plus the demonstrated sensitivity to the
forecast head. Transition capacity, initial persistence, one fit seed and
incomplete convergence limit causal attribution.

The next work should consolidate this evidence into one explicit observation
contract candidate with normalized causal scores and a falsifiable control,
rather than continue general budget/penalty sweeps. This is a research direction,
not approval of a new production emission family or evidence of live utility.

[Dataset 424242 report](validation/results/hmm-control-step10-data424242-native-20261012.json),
[424243](validation/results/hmm-control-step10-data424243-native-20261012.json),
[424244](validation/results/hmm-control-step10-data424244-native-20261012.json)
record head coefficients, mappings, independent errors and frozen reference
metrics. Production defaults and the historical NPZ verification limit remain
unchanged. No full canonical-suite, statistical-significance or trading-utility
claim.

## Trading ground-truth and HSMM contract audit — 2026-10-12

Step 11 base: d1686859e2fc4efb0eb58f2441d23bba56238760.
Owner: scripts/diagnose_trading_ground_truth.py. This rechecks the generator
and the HSMM-specific interpretation directly, rather than treating the
HMM/HSMM forecast comparison as duration recovery evidence.

### Ground-truth correctness and label repair

All three frozen v2 generations pass ledger-to-row checks for episode IDs,
regime IDs, ages, planned and observed lengths, nonself episode transitions,
macro-conditioned boundary probabilities, OHLCV consistency and rolling return
identities. Generation at 1100 rows reproduces the first 1100 rows of all 16
non-GT columns of the 60000-row generation exactly. Code inspection confirms
causal feature construction; this prefix check is bounded evidence, not an
exhaustive test of every possible prefix. Regime truth means the generating
specification, not the sign of every realized return.

The final episodes are right-censored: realized/planned lengths are
27/37, 112/168 and 20/103 for seeds 424242/424243/424244. The old
gt_remaining_duration is zero at the final row in all three despite latent
residual lives 10/56/83. The field counts remaining rows observed in the file,
not always the complete latent episode. Treating it as latent remaining life
would therefore mislabel the terminal episode.

Corrected v2 now adds gt_latent_remaining_duration = planned_duration-age,
gt_episode_right_censored, explicit ledger right_censored flags and metadata
definitions. The old observed-file residual field is preserved. Neither
observations nor pre-existing label arrays change. The legacy episode_start/v1
path receives no new fields or metadata: every array and metadata hash in a
3000-row pre-change control matches afterward. This is separate from the
still-unavailable historical frozen v1 NPZ verification.

The gt_p_next_* fields are boundary-conditioned probabilities copied onto
every episode row. In v2 their context can be future information for an
interior episode row. They are evaluation labels, not model inputs or a
contemporaneous forecast target away from a true boundary. The maintained
transition MAE uses actual boundaries, which is consistent with this contract.

### The duration class does not match the current Trading generator

Exact discrete PMFs are derived from the generator's continuous CDFs, rounding
and clipping, including probability atoms at clipped endpoints. Independent
duration samples (20000/state, fixed fresh seeds) pass a predeclared CDF error
bound 0.02. Core finite-duration hazards agree with these exact PMFs to 1e-12.

| Regime | Maximum true duration | Probability duration >=64 | KL-optimal geometric-tail hazard | Projection TV | KL nats/episode |
| --- | ---: | ---: | ---: | ---: | ---: |
| bull | 260 | 97.409% | 0.016391 | 0.22212 | 0.15942 |
| bear | 240 | 74.439% | 0.019316 | 0.07984 | 0.05665 |
| range | 250 | 56.425% | 0.011577 | 0.30024 | 0.42639 |
| chaotic | 105 | 7.373% | 0.066606 | 0.00626 | 0.01381 |

The previous D64-plus-tail configuration distinguishes ages 1–63 and collapses
age >=64 into one bucket with a constant per-state end hazard. In particular,
most bull duration mass and the longer mode of range fall into that bucket.
Their clipped, nonconstant tail hazards cannot equal one static geometric
tail. Even its KL-optimal moment-matched projection assigns 3.76%/2.36%/6.39%/
0.41% probability beyond each regime's true maximum.

These are exact static no-self-reset duration projections, not minimum-TV
bounds and not lower bounds on every context-modulated, compound-reset or
state-splitting model. They identify a specific latent-duration contract
mismatch, not a mathematical error in the geometric-tail implementation.
The core finite-D hazard conversion passes the independent exact-law check.

### Reset semantics, observation windows and the old metric

The generator forbids same-state episode renewals. NHSMM permits an episode
to end and restart in the same state, resetting latent age without changing
the visible regime. Thus a learned episode PMF is not necessarily the
ground-truth same-state run PMF. A long visible run may be composed of multiple
short latent episodes. This is the identifiability issue already established
by the earlier exact self-reset counterexample, not a newly proven recursion bug.

All twelve pilot test windows start and end inside true episodes. In six of
them, the initial left-censored episode still continues after the 64-row
warmup. initialize_filter starts mass at age1, which is correct for a fresh
episode but not the joint state-age prior of an arbitrary mid-episode window.
Warmup does not prove that the missing true initial age has been recovered.

Complete episodes fully contained in the pilot windows:

| Dataset | Training complete episodes | Test complete episodes | Test counts bull/bear/range/chaotic |
| --- | ---: | ---: | --- |
| 424242 | 13 | 4 | 0/1/2/1 |
| 424243 | 13 | 2 | 1/0/1/0 |
| 424244 | 16 | 9 | 2/1/3/3 |

Partial episodes also contribute survival information, so these are not counts
of all usable information. They demonstrate sparse complete-boundary exposure.
The larger maintained 480-row/24-train/8-test protocol is separately audited
in the report and must not be conflated with the smaller recent pilots.

The maintained duration_recovery acceptance key compares medians of filtered
argmax state runs, with first/last truncated runs included. It is a state-run
proxy, not censored episode-PMF or hazard recovery. For dataset 424242 its
ground-truth bull window median is 50 rows despite a full-ledger completed
bull median 123.5 and no fully contained bull test episode. Exact hard-state
recovery can pass that proxy while the learned latent duration law remains
wrong. Steps 7–10 did not estimate duration-PMF recovery from their state and
forecast scores.

### Research interpretation and revised next gate

Yu's review of explicit-duration HSMMs describes zero same-state episode
transition probabilities and distinguishes boundary/inference assumptions.
Bulla et al.'s HSMM package paper explicitly treats terminal occupancy as
right-censored. These are relevant contract choices, not instructions to
silently change this repository's generalized model.

Primary references:
- [Yu, Hidden semi-Markov models, 2010](https://www.cs.ubc.ca/~murphyk/Teaching/CS540-Spring10/projects/Yu-hsmm09.pdf), explicit-duration formulation and boundary conditions.
- [hsmm — An R package for analyzing hidden semi-Markov models, 2010](https://www.sciencedirect.com/science/article/pii/S016794730800426X), terminal right-censoring (publisher abstract retrieved; full-page access returned403).
- [Johnson and Willsky, Bayesian Nonparametric Hidden Semi-Markov Models](https://arxiv.org/abs/1203.1365), explicit-duration model/inference context.

The diagnosis must therefore include duration-class mismatch, latent renewal
semantics, mid-episode initialization, sparse exposure and an unsuitable
duration acceptance interpretation, alongside the previously measured emission
dependence and optimization effects. The earlier focus on observations alone
was too narrow. No new evidence establishes that the HSMM recursion itself is
incorrect.

Next: isolate the HSMM mechanism using a finite support covering260, forbidden
same-state renewals for that controlled experiment, episode-aligned starts
or a valid initial state-age distribution, and censored episode likelihood/
hazard scoring. First use known/matched emissions and transitions; only then
reintroduce the Trading observation features and joint nuisance learning.
This separates representability, inference correctness and optimizer recovery.
It is a controlled research specification, not a production-default migration.

Final verification: 65 focused tests plus six benchmark-metric checks pass
(**71 total**); the unavailable historical NPZ check is explicitly deselected.
AST/Ruff/Black pass. The final audit was rerun after the additive v2 label
repair; source/prior-report/frozen feature hashes were verified.
[Raw Step 11 report](validation/results/ground-truth-step11-native-20261012.json)
contains exact PMFs, hazards, projections, window counts and ground-truth checks.

## Isolated finite-support HSMM verification — 2026-10-12

Step 12 base: 1028ec240c53ec797f7216f5d7e32e3cab0ef841.
Owner: scripts/verify_isolated_trading_hsmm.py. This executes the isolated
mechanism gate defined by Step 11. The production model defaults are unchanged.

### Controlled contract and independent references

The duration PMFs are the exact clipped/rounded Trading generator laws,
embedded in finite D260 with known per-state support masks and no geometric
tail. Transition probabilities are the known stationary BASE_TRANSITION,
with exact zero same-state renewal probability at every age. This deliberately
removes macro-conditioned transition learning from the mechanism control.
Initial states are uniform and every generated observation sequence starts
at a new episode with age1.

Emission parameters are known IID Gaussian means -6/-2/2/6, sigma 0.35,
one feature. The production NHSMM model is Float64; generated/public input
observations use the same Float32 values. Script-only masks apply through the
Duration constraint hook and Transition structural buffer. They are research
constraints, not a new supported config/runtime/artifact contract.

Stage 1 trains only production Duration logits, using 8192 labelled
administrative-censor records per state. Independent cutoff C is uniform over
integers 80..300. A completed event is observed if L<C; otherwise the contribution
is survival P(L>=C). Adam runs exactly 1200 steps at lr0.05, with all emission,
initial-prior, transition and other parameters frozen and checked unchanged.
An independent NumPy labelled EM runs 200 iterations, allocating censored
duration counts conditionally over surviving lengths. It supplies a separate
likelihood/MLE reference. This stage explicitly uses episode labels; it is
not the production optimize method or latent marginal-likelihood training.

Stage 2 evaluates the public NHSMM marginal likelihood and causal filter on
eight independently generated 600-row sequences per seed, using both true and
learned duration laws. The independent NumPy reference sums completed segments
using exact duration mass and terminal unfinished segments using survival.
It computes state probabilities at every prefix, not future-smoothed labels.
Unit tests also compare it against enumerated state-age paths in a small
ambiguous example.

A third control uses sigma 2.0, two 300-row sequences per seed and the true
duration law. It ensures the public recursion check has multiple plausible
state paths rather than only nearly deterministic state labels.
State accuracy is descriptive in this overlapping control; its gate concerns
likelihood/probability agreement and nonzero uncertain-row exposure.

Predeclared gates: each learned duration PMF TV<=0.12, labelled objective gap
to independent EM<=0.01 nats/record, state accuracy>=0.99 for the well-separated
control, independent sequence-LL discrepancy<=1e-7 and state-probability
discrepancy<=1e-8. Seeds 901/902/903, sampling sizes and budgets are fixed before
test evaluation. The final report supersedes the first separated-only run.

### Measured gate results

All three seeds pass every applicable gate:

| Seed | TV bull | TV bear | TV range | TV chaotic | Objective gap nats/record | Separated-state accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 901 | 0.07238 | 0.06661 | 0.06503 | 0.03346 | 0.00004886 | 100% |
| 902 | 0.06420 | 0.05487 | 0.06787 | 0.03685 | 0.00004664 | 100% |
| 903 | 0.06594 | 0.07319 | 0.06301 | 0.03952 | 0.00007232 | 100% |

All 3600 duration updates were completed; each seed uses 32768 labelled records.
The learned means remain close to the exact true means, with maximum absolute
error approximately 1.08 rows across the twelve state/seed cells.

Across the nine public-model/reference comparisons, maximum total sequence
likelihood discrepancy is 4.18e-10 and maximum prefix state-probability error
is 8.03e-14. Known and learned laws both yield 100% state accuracy on the
14400 unique separated-emission held-out rows. The overlapping controls have
91/66/139 rows with maximum state probability below0.99, and accuracies
98.50%/99.17%/99.00% over 600 rows per seed. Their independent likelihood errors
are at most 7.96e-12. Nontrivial uncertainty therefore does not reveal a recursion
disagreement.

Final focused gate: **70 tests passed**, AST/Ruff/Black pass, with 132 unchanged
formatted materialized Python files. Source hashes and all report gates were
verified. Every frozen nuisance parameter remains unchanged during training.
[Raw Step 12 evidence](validation/results/isolated-hsmm-step12-native-20261012.json)
includes learned PMFs, means, censor counts, EM objectives, actual optimizer
checkpoints and independent probability/likelihood errors.

### Accepted claim and remaining boundary

This accepts the finite-support duration module's labelled learning and the
public causal HSMM marginal likelihood/filter under this matched oracle
contract, including terminal survival and ambiguous observations. It gives
no new evidence for an inherent HSMM recursion failure.

It does not establish unsupervised joint training on the Trading features.
Known support, emission/transition parameters, zero self-renewals, correct
starts and many labelled duration records are substantial controls. Script
constraints have not been verified as a production runtime/artifact feature.
The historical frozen-NPZ check and full post-change canonical suite remain
outside this focused gate.

Next: remove episode labels from duration optimization while keeping the same
known nuisance parameters and matched finite-support/start contract. Only
after that separate marginal-learning gate should Trading observations,
macro/context learning and joint nuisance estimation be reintroduced.

## Observation-only marginal duration learning — 2026-10-12

Step 13 base: 43e6789c378e2d559839112d25a67340d9800f0d.
Owner: scripts/verify_unlabelled_trading_hsmm.py. This removes episode labels
from the training loss, while preserving Step 12's matched finite D260 laws,
known support, zero same-state resets, known stationary transitions and fresh
age1 starts. Known IID Gaussian emissions have means -6/-2/2/6 and sigma2.0,
so state uncertainty is present during training, not just in an inference control.
No production defaults, model implementation or public API are changed.

The fixed design uses seeds901/902/903, 128 independently generated training
sequences and24 held-out sequences per seed, each300 rows. Duration logits
start uniform within the known support. Adam lr0.08 runs exactly80 updates,
with shuffled batch2 observations and no replacement within each pass.
The loss is negative public NHSMM.log_likelihood(reduce=True) divided by
minibatch rows, at fixed model temperature. All nuisance parameters are frozen
and checked bitwise unchanged. The fit API accepts observations, update count
and shuffle seed only: no states, episode boundaries, ages, planned durations,
or labelled counts enter the loss or gradient. Labels are used after fitting
for descriptive exposure and held-out state accuracy only. This is duration-only
marginal optimization, not a run of stock joint NHSMM.optimize.

Independent validation uses central differences of the NumPy completed-segment
reference, including terminal survival, at12 duration-logit coordinates (first,
middle and last supported length/state). Production autograd must agree within
1e-5 absolute error; masked-out gradients must be exactly zero and every training
gradient finite. Held-out public likelihood and every-prefix state probabilities
must agree with the segment reference within1e-7 and1e-8 respectively.

The learning-progress criterion requires improved full-training and independent
held-out likelihood, plus reduced duration-mean RMSE against the exact DGP.
The stricter Step 12 criterion, every PMF TV<=0.12, is reported separately.
Likelihood or mean improvement is not substituted for fine-bin PMF recovery;
this bounded sample/update budget is much smaller than Step 12's labelled
32768 records and1200 updates per seed. No held-out hyperparameter tuning,
best-iterate selection or post-result threshold adjustment is performed.

A preliminary600-row execution was explicitly interrupted after its last
reported40/80 updates for seed901 because measured backward-pass cost exceeded
the intended bounded three-seed budget. Its final parameters, held-out fitted
scores and partial updates are not used as evidence. The revised300-row design
retains D260,80 updates and both acceptance criteria; it reduces episode and
complete-duration exposure and increases terminal censoring. This resource-based
revision precedes final fitted/held-out result inspection, not a result-driven
threshold or model-selection change.

### Measured results

The revised execution completes all240 updates. Each run uses38400 unique
training rows and7200 independent held-out rows; all128 training sequences are
visited. Full training likelihood increases in all runs. Mean recovery improves
substantially, but held-out generalization and fine PMF recovery are not accepted.

| Seed | Train LL gain nats/row | Held-out LL gain nats/row | Duration-mean RMSE rows | TV bull | TV bear | TV range | TV chaotic | Progress gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 901 | 0.008349 | 0.000949 | 4.696 | 0.48042 | 0.40849 | 0.37287 | 0.28220 | PASS |
| 902 | 0.009145 | -0.001315 | 5.741 | 0.49779 | 0.36913 | 0.36012 | 0.28793 | FAIL |
| 903 | 0.008601 | 0.001164 | 8.112 | 0.45897 | 0.31957 | 0.37109 | 0.28711 | PASS |

Uniform-support initial mean RMSE is30.872 rows in every run; initial PMF TV
is0.37524/0.31179/0.41660/0.41355. Bull and bear PMF TV become worse in all
three runs despite improved means. None of the12 fitted state/seed PMFs reaches
TV<=0.12. Thus even the2/3 learning-progress passes do not establish fine-bin
duration recovery; training likelihood is not an adequate acceptance substitute.

Maximum independent central-difference gradient error is1.91e-9 over36 checked
coordinates; masked-out gradients are exactly zero and all training gradients
remain finite. Maximum held-out total likelihood error is1.05e-11 and maximum
state-probability error is6.33e-14. The compared600 rows per seed have98/110/110
uncertain prefixes (maximum probability<0.99), with descriptive state accuracy
97.50%/97.50%/98.83%. This agreement therefore includes ambiguous state paths,
not only effectively observed labels. Frozen nuisance parameters remain bitwise
unchanged. Report/source hashes and completed update counts are verified.

Evaluation-only complete-episode exposure is sparse relative to the unrestricted
per-bin duration parameterization:

| Seed | Complete bull/bear/range/chaotic | Terminal-censored bull/bear/range/chaotic | Supported bins without a complete observation |
| --- | --- | --- | --- |
| 901 | 75 / 86 / 116 / 106 | 36 / 28 / 54 / 10 | 162 / 144 / 168 / 50 |
| 902 | 83 / 86 / 125 / 108 | 46 / 40 / 36 / 6 | 159 / 146 / 168 / 50 |
| 903 | 75 / 104 / 122 / 98 | 45 / 30 / 44 / 9 | 164 / 138 / 165 / 54 |

These counts are descriptive ground-truth diagnostics computed after fitting,
not training inputs. Zero complete-bin exposure does not imply zero information:
terminal survival and neighbouring state ambiguity also enter the marginal loss.
Nevertheless, each state's75–125 completed episodes is far less coverage than
Step 12's8192 labelled records/state. Sparse coverage and an unrestricted noisy
PMF are plausible overfitting mechanisms, consistent with seed902's divergence
between training and held-out likelihood. This experiment alone does not prove
whether sampling error, stochastic optimizer budget or the sample MLE dominates.
It does not show that more updates would fix generalization.

Conclusion: genuine label-free gradients and duration-mean learning work under
the matched oracle contract; numerical likelihood/gradient tests pass. Robust
held-out generalization and exact duration-PMF recovery remain open. No inherent
HSMM recursion failure is demonstrated by this control. The next focused step
is an independent observation-only EM/MLE comparison and a predeclared sample-size
control to separate optimizer error from statistical exposure, before joint
Trading features, macro/context and nuisance learning are reintroduced.

Final focused tests:8 passed; AST/Ruff/Black pass over35 verified materialized/new
Python files. No new full canonical-suite, historical-NPZ, runtime/artifact-mask
or real-market utility acceptance claim.
[Raw Step 13 evidence](validation/results/unlabelled-hsmm-step13-native-20261012.json).

## Independent unlabelled EM and episode exposure — 2026-10-12

Step 14a base: e34af5efa028b7dbff70215eaf3c16685d20e1a3.
Owner: scripts/compare_unlabelled_duration_em.py. Production behavior and
defaults remain unchanged. The independent NumPy reference uses a normalized
probability-space state-age forward recursion and log-domain backward messages.
Completed boundary posterior counts include only boundaries before the final
observed row. Terminal state-age posterior counts allocate latent total length
using p(L=d | L>=age); their sum is exactly one censored episode/sequence.
The M-step normalizes total expected duration counts separately per state.
No labels, hard decoded states or ground-truth episode records enter learning.

Mathematical verification includes complete enumeration of latent full-duration
paths (including the unobserved terminal length), a one-row zero-duration-score
control, central finite differences and the independent segment likelihood.
The posterior-count softmax score counts-p*sum(counts) is compared against all1040
production duration-gradient coordinates on ambiguous300-row observations.
Four new tests plus maintained Step12/13 tests pass:12 tests. AST/Ruff/Black
pass over37 verified materialized/new Python files.

Predeclared comparison: replay Step13 observations and fitted Adam PMFs exactly
by hash; fit the independent EM from the same uniform-support PMF on128 sequences
and on1024 paired sequences, each300 rows. The larger arm's first128 sequences
must equal the small arm byte-for-byte. Both arms use30 full-data EM updates,
chunks64, the same known nuisance parameters and unchanged24 held-out sequences.
Seeds901/902/903 are fixed. Ground-truth labels serve exposure/state evaluation
only after fitting. EM marginal likelihood must be monotonic within1e-6 absolute
tolerance; public/segment likelihood and all-logit gradient discrepancies must
remain<=1e-7; public state-probability error<=1e-8. Training objective gains,
held-out gains, mean RMSE and the unchanged each-PMF-TV<=0.12 criterion are
reported separately.30 updates are a bounded EM reference, not proof of a global
MLE; last-update objective increments and residual scores must be retained.

A128-sequence resource smoke and chunk16/32/128 comparison took approximately
one second per full expectation pass and gave identical likelihoods. The final
experiment uses one worker, native CPU dependencies and an1800-second bound.
Numerical mechanics were accepted at Step14a commit
ba81eb1997c3ccdf409ba1fc95f16beeaaae3def. Step14b executes the predeclared
comparison from that immutable base. This is not joint Trading-feature,
macro-context, runtime/artifact-mask or real-market acceptance.

### Step 14b measured comparison

All six EM arms complete30 updates,180 M-steps total. The larger arms contain
921600 unique training rows across three seeds, including the115200 rows in the
small paired arms. The unchanged held-out set contains21600 rows total.
Replay hashes, first128 equality, all source/report hashes, prior Adam training
scores and uniform held-out scores are verified. Labels enter neither EM nor
the prior Adam training loss; exposure and accuracy remain evaluation only.

| Seed | Small EM minus Adam train LL nats/row | Small EM held-out gain versus uniform | Large EM held-out gain versus uniform | Small EM mean RMSE rows | Large EM mean RMSE rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| 901 | 0.002729 | -0.016264 | 0.000855 | 2.242 | 1.269 |
| 902 | 0.002771 | -0.020558 | 0.003265 | 7.345 | 1.416 |
| 903 | 0.002807 | -0.023328 | 0.002115 | 7.147 | 0.831 |

Small-data EM also performs worse held-out than Step13 Adam in every seed;
Adam's held-out gains were0.000949/-0.001315/0.001164 nats/row. Adam therefore
has a measured optimization gap to this bounded reference, but closing that
gap on the same small sample does not fix generalization. These are observed
comparisons on24 held-out sequences/seed, not confidence bounds or a guarantee
for another sample or the Trading feature model.

| Seed | Arm | TV bull | TV bear | TV range | TV chaotic | Each-PMF-TV<=0.12 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 901 | EM128 | 0.60746 | 0.59810 | 0.57730 | 0.41011 | FAIL |
| 901 | EM1024 | 0.22715 | 0.26387 | 0.25675 | 0.13183 | FAIL |
| 902 | EM128 | 0.59856 | 0.65973 | 0.58668 | 0.35194 | FAIL |
| 902 | EM1024 | 0.23335 | 0.28524 | 0.25726 | 0.12296 | FAIL |
| 903 | EM128 | 0.60883 | 0.61120 | 0.58827 | 0.39333 | FAIL |
| 903 | EM1024 | 0.23044 | 0.28744 | 0.23938 | 0.16762 | FAIL |

Every state/seed TV improves with the paired larger sample. None of the12
large-arm PMFs passes the unchanged0.12 criterion, including the near-threshold
chaotic state in seed902. Accurate means are not substituted for full-law recovery.

| Seed | Large complete episodes bull/bear/range/chaotic | Large terminal-censored episodes | Supported bins without complete observations |
| --- | --- | --- | --- |
| 901 | 640 / 721 / 894 / 855 | 296 / 241 / 383 / 104 | 77 / 53 / 80 / 14 |
| 902 | 620 / 737 / 936 / 834 | 332 / 258 / 342 / 92 | 74 / 52 / 79 / 17 |
| 903 | 632 / 682 / 919 / 842 | 363 / 244 / 339 / 78 | 64 / 59 / 74 / 21 |

The larger sample substantially increases episode/bin exposure, although many
rare complete-duration bins remain unobserved. Terminal survival and uncertain
boundary posteriors still provide information beyond these descriptive counts.
Even the larger control supplies far fewer records/state than Step12's8192
labelled records/state and also must infer boundaries. The controlled size
effect, with fixed nuisance parameters, algorithm, initialization and budget,
supports finite-sample sensitivity of the unrestricted PMF. It does not prove
that sample size alone explains all remaining duration error or joint Trading
failures, nor that a larger sample automatically closes the fine-law gate.

### Numerical and bounded-reference evidence

All3120 production duration-score coordinates across the three seeds agree with
independent posterior-count scores within2.27e-14. Maximum public-vs-forward
likelihood discrepancy at the gradient check is9.10e-13 and segment-vs-forward
discrepancy is5.23e-12. Each large learned law also passes the public causal
likelihood/filter comparison against independent completed-segment recursion:
maximum LL discrepancy1.12e-11 and state-probability error1.10e-13. The compared
600 rows/seed have113/103/105 uncertain prefixes and state accuracies
97.67%/97.67%/98.83%. Every E-step checks backward/evidence consistency and
terminal/count identities; all six full-data marginal objectives are monotonic.

| Seed | Last small EM LL increment nats/row | Last large EM LL increment nats/row | Small max score/sequence | Large max score/sequence |
| --- | ---: | ---: | ---: | ---: |
| 901 | 3.171e-7 | 1.585e-7 | 2.878e-4 | 7.900e-5 |
| 902 | 5.405e-7 | 8.826e-8 | 5.285e-4 | 5.016e-5 |
| 903 | 9.527e-7 | 1.083e-7 | 8.566e-4 | 3.702e-5 |

Residuals are small but nonzero;30 EM updates are not a certificate of global
MLE or exact convergence. The gap to Adam is a gap to this attained reference
objective. The normalized probability forward is Float64 and checked against
log-domain backward evidence and the independent/public recursions; these
bounded controls do not claim universal underflow immunity for other inputs.

Conclusion: inference/gradient mechanics pass. Both an optimizer-objective gap
and a substantial sample/flexibility problem are present in this oracle control.
Stronger small-sample fitting produces worse duration-law and held-out metrics;
larger sample exposure reverses much of that degradation. The next step is a
compact or smoothed/regularized duration law, with a predeclared design and any
selection confined to training data. Joint Trading observations/context remain
deferred. No production default or public artifact/runtime contract is changed.
[Raw Step14 evidence](validation/results/unlabelled-em-step14-native-20261012.json).

## Compact duration family with training-only selection — 2026-10-13

Step15a base:8b67f37784667fb614f236412fda9648dd6279ff.
Owner:scripts/compare_compact_duration_em.py. This batch tests the sample/flexibility
finding from Step14 without changing production defaults or model implementation.
Dates for this continuation use Europe/Berlin. The stationary nuisance parameters,
known finite D260 support, zero self-renewals, fresh episode starts, Gaussian
sigma2.0 observations and correct terminal censoring remain the same oracle controls.

The family gives each state's contiguous interior duration bins a common density
within blocks of4,8 or16 lengths. The first and last supported lengths are separate
freely learned point masses, preserving clipped endpoint atoms without forcing
them to equal neighbouring densities. Only the known support is used; generator
probability values and regime-specific Gamma/Weibull/mixture shapes are not used
in fitting or selection. Endpoints/support remain substantial oracle assumptions.

| Family | Free parameters bull/bear/range/chaotic | Total |
| --- | --- | ---: |
| Unrestricted supported PMF | 218 / 210 / 232 / 97 | 757 |
| Interior width4 | 56 / 54 / 59 / 25 | 194 |
| Interior width8 | 29 / 28 / 30 / 13 | 100 |
| Interior width16 | 15 / 15 / 16 / 7 | 53 |

The independent E-step is unchanged from Step14. The constrained exact Q M-step
assigns each block a mass proportional to its total expected count and divides
that mass uniformly over its member durations. This is restricted maximum
likelihood in a lower-dimensional family, not an added prior or a smoothing penalty.
Observed marginal likelihood must increase monotonically within1e-6. The uniform
per-supported-length initial law is identical across widths and previous controls;
group masses are initially proportional to their block sizes, not uniform across
unequal blocks. Width1 recovers the unrestricted count-normalization update.

Predeclared batch:seeds901/902/903, candidate widths4/8/16. For each seed, a seeded
permutation partitions the128 small training sequences into96 inner training
and32 inner validation sequences. Every candidate gets20 constrained EM updates.
Selection uses only highest inner-validation marginal likelihood; exact ties
choose the smaller width. No episode labels, true PMF values, outer held-out rows
or outer recovery scores enter the selection API. The selected width is refitted
from the original uniform law on all128 sequences and on the paired1024-sequence
sample,30 updates each. It is not reselected using the larger sample. The larger
arm's first128 observations must equal the small arm, and the unchanged24 outer
held-out sequences per seed are constructed/scored after inner selection/refits.

Step13/14 input hashes and baseline scores are retained for exact same-data
comparisons. Small/large held-out gains, training fit costs, PMF TV, mean RMSE,
residual group scores and parameter counts are reported separately. The unchanged
full duration gate requires positive outer held-out gain versus the uniform law
and every state's PMF TV<=0.12, separately for each data-size arm. Public all-logit
gradient/likelihood discrepancy must remain<=1e-7, and state-probability error<=1e-8.
There are9 inner candidate fits and6 outer refits:360 total EM updates.

Six new tests cover exact/disjoint endpoint-preserving partitions, invalid support,
hand-solvable constrained Q maximization, zero grouped score at the M-step,
width1 equivalence, normalization/support/constant density, constrained monotonicity,
marginal-score finite differences and disjoint likelihood-only inner selection.
Together with Step12–14:18 tests pass. AST/Ruff/Black pass over39 verified
materialized/new Python files.45 immutable inputs were connector-verified;
the existing native Python/Torch environment remains available.

One worker and a1800-second bound. Step15a accepts the constrained mechanics and
freezes selection/acceptance before outer result inspection. Its verified commit
is d3e6e9ad983ac4f1558f21333c68b49c304819fd; Step15b executes from that base.
Approximation bias, small inner-validation samples and bounded EM remain
limitations; no joint Trading-feature/context, runtime/artifact-mask,
production-default or real-market acceptance follows from this control.

### Step15b measured selection and recovery

All9 inner fits complete20 updates and all6 outer refits complete30 updates:
360 M-steps total. All15 marginal-likelihood histories are monotonic. Replayed
source/input-report and observation hashes, disjoint96/32 inner indices, both
inner data hashes, selected argmax, support and constant block densities are
verified. The45 immutable inputs were connector-verified before work; baseline
Step13/14 reports are reused exactly, not regenerated or replaced.

| Seed | Validation NLL/row width4 | Width8 | Width16 | Chosen width | Free parameters |
| --- | ---: | ---: | ---: | ---: | ---: |
| 901 | 2.177901 | 2.169805 | 2.166724 | 16 | 53 |
| 902 | 2.184859 | 2.179685 | 2.177868 | 16 | 53 |
| 903 | 2.165044 | 2.159637 | 2.158585 | 16 | 53 |

Every inner-validation argmax chooses width16 without labels or outer test
access. These compare attained20-step fits, not certified family-global optima.
Width is selected once per seed on small training data and reused unchanged
for the paired larger sample; it is not optimized for true PMF TV.

| Seed | Arm | TV bull | TV bear | TV range | TV chaotic | Held-out gain versus uniform nats/row | Mean RMSE rows | Full gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 901 | Compact128 | 0.16294 | 0.18744 | 0.16340 | 0.11202 | 0.000384 | 2.470 | FAIL |
| 901 | Compact1024 | 0.06082 | 0.08049 | 0.11169 | 0.10946 | 0.003287 | 1.356 | PASS |
| 902 | Compact128 | 0.15943 | 0.23274 | 0.19397 | 0.12237 | 0.001114 | 7.232 | FAIL |
| 902 | Compact1024 | 0.07918 | 0.05823 | 0.10529 | 0.10550 | 0.003794 | 1.470 | PASS |
| 903 | Compact128 | 0.15319 | 0.10986 | 0.24327 | 0.10726 | -0.003483 | 7.130 | FAIL |
| 903 | Compact1024 | 0.05292 | 0.05593 | 0.10059 | 0.10853 | 0.003229 | 0.836 | PASS |

The1024-sequence arm passes all12 state/seed TVs and positive held-out gain
in all three seeds at the unchanged threshold. Small arms remain0/3 full
passes; seed903 is held-out-negative even after reducing flexibility. Thus
compactness does not substitute for adequate sample exposure, and accurate
means alone do not close the small-sample full-law gate.

### Same-data cost and benefit versus unrestricted EM

| Seed | Compact128 held-out gain over free EM nats/row | Compact1024 gain over free EM | Compact128 minus free train LL nats | Compact1024 minus free train LL nats |
| --- | ---: | ---: | ---: | ---: |
| 901 | 0.016648 | 0.002432 | -247.446 | -389.872 |
| 902 | 0.021672 | 0.000528 | -248.966 | -411.136 |
| 903 | 0.019845 | 0.001113 | -236.646 | -377.265 |

Every compact state/seed TV and every held-out likelihood is better than its
unrestricted Step14 counterpart; training likelihood is lower in all six arms.
This records the bias/flexibility tradeoff instead of treating better training
fit as recovery. The controls support unrestricted-PMF overfitting and limited
episode exposure as important mechanisms in this matched oracle problem; they
do not identify every joint Trading failure or justify a width16 production
default for other duration laws. Exposure is identical to Step14's paired data.
Group sharing reduces independent-bin evidence requirements; endpoint atoms
retain clipped masses. Known support/nuisance parameters remain oracle assumptions.

### Numerical verification and residuals

All3120 checked production raw duration-logit gradients at the three learned
small laws agree with independent posterior scores within3.01e-14. Maximum
public-vs-forward LL discrepancy is9.10e-13 and segment-vs-forward discrepancy
is4.89e-12. Six learned-law public likelihood/filter comparisons pass with
maximum LL error1.14e-11 and prefix state-probability error8.66e-14. Compared
600 rows/arm/seed include108/112/109 uncertain prefixes for small fits and
110/106/103 for large fits. Descriptive state accuracies are97.67%/97.50%/98.83%
in both arms, so these checks include ambiguous state paths.

| Seed | Last small LL increment nats/row | Last large LL increment nats/row | Small max group score/sequence | Large max group score/sequence |
| --- | ---: | ---: | ---: | ---: |
| 901 | 1.110e-11 | 2.500e-9 | 1.057e-6 | 1.582e-6 |
| 902 | 3.001e-13 | 1.239e-9 | 3.771e-10 | 1.338e-5 |
| 903 | 6.852e-9 | 1.760e-8 | 2.188e-5 | 5.517e-5 |

Residuals are nonzero; no global-MLE/convergence proof is claimed. Code remains
unchanged from the18-test/AST/Ruff/Black gate at Step15a. All360 updates,15
monotonic histories, source hashes and numerical gates are verified separately
from statistical outcomes. The report retains inner histories/scores/indices,
both outer PMFs, baseline differences, small failures and large passes.

Conclusion: observation-only compact duration learning closes the predeclared
large-sample gate across all three seeds under the known-support/nuisance/start
contract. Small-sample recovery, unknown support, stock joint optimize, Trading
observations/context and market usefulness remain open. Next: remove oracle
per-state support while keeping nuisance parameters known, before public-optimizer
integration or joint Trading learning.
[Raw Step15 evidence](validation/results/compact-duration-step15-native-20261013.json).

## Oracle support disabled by default, retained as a control — 2026-10-13

The compact-duration CLI now has --support-mode full|oracle, default full.
Full mode constructs support from configured K4/D260 only: every state permits
durations1..260 and its free endpoint atoms are at1/260. Oracle mode explicitly
restores the earlier bounds42..260,30..240,18..250,8..105 and their endpoint atoms.
The oracle implementation and historical reports are retained. This switches
duration support only; emission/transition/initial parameters remain known.

The selected mask reaches inner selection, fitting, expanded production models,
likelihood/filter and all-logit gradient reference checks. In full mode those
paths do not reinstate the ground-truth mask. Its held-out gain uses a newly
computed uniform full-support baseline, rather than Step14's uniform oracle
baseline. Comparisons to stored unrestricted Step14 fits are labelled as oracle
references and therefore include a support-assumption difference in full mode.
Ground-truth probability values enter recovery metrics after fitting only.

At width16, generic full support has18 free parameters/state (72 total), versus
the earlier15/15/16/7 (53) under oracle bounds. Thus turning off oracle support
also changes block alignment and approximation bias. New full-support statistics
must not be inferred from the successful oracle-support Step15 results.

Schema v2 records mode/flag, active per-state bounds and the matched uniform
baseline. Existing compact v1 reports are recognized as oracle-mode evidence;
the CLI rejects overwriting an existing report with the other mode. New full
results should use a distinct path, for example
docs/validation/results/compact-duration-full-support-native-20261013.json.
To rerun the oracle control with current code, specify --support-mode oracle.

Verification:22 focused tests; AST/Ruff/Black pass over39 materialized Python
files. The new regressions forbid ground-truth mask access in the full path,
check public probability/gradient agreement with positive mass outside true
support, verify default/explicit modes and output protection, and execute short
monotonic generic-support fits. No new full three-seed recovery batch is claimed.
Production defaults remain unchanged. Next: execute the generic full-support
control while retaining the explicit oracle mode for comparison.

## Full-support batch execution checkpoint — 2026-10-13

Step17a preserves the fixed Full-Support experiment: configured ages1..260 in
every state, no per-state oracle mask, widths4/8/16, inner96/32 likelihood-only
selection at20 updates and128/1024 paired refits at30 updates, three seeds.
The oracle mode and prior acceptance evidence remain available independently.

Resource profiling showed slower native execution than the oracle run. The
preliminary attempt was interrupted after its last reported small901 update30,
before any seed completed its full comparison or wrote a report. Its partial
updates are discarded, and no fitted held-out result from it is accepted.
A denormal-mode timing probe produced zero observed likelihood/count difference
on a128-sequence expectation pass but only a modest timing change; it was not
used to alter numerical mode. Source algorithms, sample sizes and thresholds
remain unchanged for the completed experiment.

Bounded segments now use --stop-after-seed. Subsequent --resume calls require
identical schema, base/source/input/runtime/contract metadata and complete
20-update inner/30-update outer histories for every saved seed. Only complete
seeds are reused; duplicates, unknown IDs or incomplete results are rejected.
This is execution/recovery tooling, not new likelihood optimization or a model
change. The three-seed experiment remains360 actual M-steps at completion.
24 focused test cases and static checks cover this checkpoint; remaining
Full-Support seeds are pending. Completed-seed and final checkpoints will
state coverage explicitly and retain the original numerical and recovery gates.

## Full-support seed901 result — 2026-10-13

Step17b reports one complete seed of three, at source commit
ca73c96cddf373677be0711882981e618e89e9a6, in
docs/validation/results/compact-duration-full-support-native-20261013.json.
Oracle support is disabled: every state learns over configured ages1..260.
Known emission/transition/initial nuisance parameters remain frozen. Inner
validation selects width16 (72 duration degrees of freedom); candidate NLLs
for widths4/8/16 are2.16874794/2.16520026/2.16511685 nats/row.

| Arm | PMF TV: bull / bear / range / chaotic | Mean RMSE | Held-out gain over full-support uniform, nats/row | Full gate |
| --- | --- | --- | --- | --- |
|128 sequences|0.120317 /0.174091 /0.181125 /0.149312|1.800663|+0.00712244|FAIL|
|1024 sequences|0.066578 /0.104138 /0.111409 /0.140846|1.249540|+0.00698475|FAIL|

The large fit improves the first three TVs below0.12, while the chaotic law
still misses the unchanged gate. Both held-out gains are positive. Relative
to the paired Step15 oracle-support compact fits, absolute held-out LL changes
by+0.00254739 nats/row in the small arm and-0.00049290 in the large arm.
These paired differences compare fitted likelihoods directly: the uniform
baselines differ between modes. Removing oracle support also changes block
alignment and approximation bias; this single seed does not identify their
separate contributions or establish generic duration recovery.

All120 updates, five complete monotonic histories, source/input/data hashes,
inner-selection and compact-PMF invariants are verified. Independent posterior
gradients and public/reference recursion agree within the original numerical
gates (gradient2.05e-14, LL1.10e-11, state probability7.29e-14). Numerical
correctness passes, statistical recovery fails. Seeds902/903 remain pending;
production defaults, stock joint optimization and Trading/market acceptance
are unchanged and not inferred from this checkpoint.

## Full-support seed902 result — 2026-10-13

Step17c appends the second complete seed to the same v2 report and source
identity ca73c96. Inner selection again chooses width16. All240 cumulative
updates over ten fits and the same replay/hash/selection/PMF/numerical gates
are verified. Known nuisance parameters remain frozen; oracle support is off.

| Arm | PMF TV: bull / bear / range / chaotic | Mean RMSE | Held-out gain over full-support uniform, nats/row | Full gate |
| --- | --- | --- | --- | --- |
|128 sequences|0.136356 /0.192948 /0.206291 /0.163519|7.402074|+0.00489078|FAIL|
|1024 sequences|0.074259 /0.074719 /0.099911 /0.140136|1.483045|+0.00645551|FAIL|

As in901, the large arm passes the first three state-TV gates but fails the
chaotic law. Positive held-out likelihood gain alone is not full recovery.
Seed902 maximum numerical errors are gradient1.94e-14, LL1.23e-11 and state
probability8.45e-14. Two seeds are complete; seed903 and the final three-seed
interpretation remain pending. No production or Trading acceptance is added.

## Compact-family representation bound — 2026-10-13

Evaluation-only analytic evidence is saved in
docs/validation/results/compact-duration-family-bias-native-20261013.json.
For true PMF p and each disjoint constant-density block B, every scalar density
c satisfies sum(i in B)|p_i-c| >= sum(i in B)|p_i-median(p_B)|. Summing over
blocks and dividing by2 yields a valid TV lower bound; additionally requiring
normalization cannot lower it. Endpoint singleton atoms contribute zero.
The retained oracle support contains all true mass, so its outside-support
contribution is also zero. This calculation uses truth only after training;
it changes neither the active run nor likelihood-only width selection.

| Support / width | Chaos TV lower bound | Chaos TV of forward-KL projection |
| --- | --- | --- |
|full /4|0.029476|0.031434|
|full /8|0.051848|0.055117|
|full /16|0.136925|0.138386|
|oracle /16|0.096626|0.101201|

Full-support width16 cannot satisfy the0.12 chaotic-law TV gate at any sample
size or optimizer budget. The observed large901/902 TVs near0.140 are close
to this representational obstruction. The forward-KL projection assigns each
block its true total mass, uniformly within that block; it is a feasible
family member, not the minimum-TV solution. Full-support width8 has a feasible
projection below0.12 for every state (TV0.022054/0.030194/0.050430/0.055117),
but this does not establish finite-sample recovery or guarantee its selection.

The ages2..17 block alone contributes at least0.077118 TV for the chaotic
state. It spans zero true mass at ages2..7 and positive mass starting at8;
one constant density cannot represent that boundary. Ages34..49 contribute
another0.022489. The raw diagnostic retains every block contribution, whose
sum reproduces the0.136925 lower bound.

Removing the oracle changes the alignment of blocks and eliminates separately
free atoms at true clipped endpoints. The above bound proves a family-bias
failure for this configured width16; it does not prove an HSMM recursion defect
or unrestricted identifiability. Likelihood selection optimizes predictive
evidence, which need not recover the fine PMF. Next research should examine
a support-independent duration family that can resolve abrupt shape changes,
with training-only selection and fresh evaluation; retain the unchanged current
batch and failed gate as evidence rather than changing its acceptance threshold.

## Complete full-support batch — 2026-10-13

Step17d completes the unchanged ca73c96 experiment with all three seeds.
The final v2 report above contains360 actual EM updates over15 monotonic fits,
with widths selected solely from inner training observations. All three choose
width16. Support is1..260 for every state; oracle support is disabled and its
explicit mode/evidence retained. Emission/transition/initial nuisance parameters
remain known and frozen. Ground-truth probabilities enter evaluation only.

| Seed / arm | PMF TV: bull / bear / range / chaotic | Mean RMSE | Held-out gain over full-support uniform, nats/row | Full gate |
| --- | --- | --- | --- | --- |
|901 /128|0.120317 /0.174091 /0.181125 /0.149312|1.800663|+0.00712244|FAIL|
|901 /1024|0.066578 /0.104138 /0.111409 /0.140846|1.249540|+0.00698475|FAIL|
|902 /128|0.136356 /0.192948 /0.206291 /0.163519|7.402074|+0.00489078|FAIL|
|902 /1024|0.074259 /0.074719 /0.099911 /0.140136|1.483045|+0.00645551|FAIL|
|903 /128|0.141035 /0.096494 /0.245746 /0.157305|7.492731|+0.00704970|FAIL|
|903 /1024|0.053135 /0.062190 /0.101877 /0.155491|1.044839|+0.00668295|FAIL|

All six arms improve predictive likelihood against the same-mode uniform
baseline. Every large arm passes the first three state-TV gates but fails
the chaotic-law gate. Large-arm mean RMSE remains1.04–1.48, illustrating why
accurate means and predictive gain do not establish fine PMF recovery.

| Seed | Full minus oracle fitted held-out LL/row,128 | Full minus oracle fitted held-out LL/row,1024 |
| --- | --- | --- |
|901|+0.00254739|-0.00049290|
|902|-0.00016030|-0.00127514|
|903|+0.00747040|+0.00039201|

These are paired absolute fitted-likelihood differences on identical held-out
observations, not differences between mode-specific baseline gains. Support
removal does not uniformly worsen predictive likelihood. Nevertheless, the
selected full-support width16 family has a proven chaotic TV floor0.136925;
it cannot satisfy0.12 with more samples or further optimization. The bound
and observed fits distinguish representational bias from a numerical defect.
They do not exclude finite-sample or optimization error above that floor.

Source/input/replay hashes, selection and PMF invariants, complete budgets and
independently recomputed metrics/gates all pass. Maximum public/reference
gradient/LL/state-probability errors are2.05e-14/1.23e-11/8.45e-14; completed
CLI resume skips all seeds. Numerical correctness is verified; generic duration
recovery remains unaccepted. Production defaults, joint optimization, historical
NPZ verification and Trading/live-market acceptance are unchanged. Next research
requires a support-independent family capable of resolving abrupt duration
shape changes, training-only selection and fresh evaluation. No acceptance
threshold or completed result is retroactively changed.

## Adaptive full-support research contract — 2026-10-13

Step18a implements an alternative family without per-state oracle bounds or
fixed-grid alignment. A partition covers every configured age1..260. For
posterior count n_i and fixed pseudocount eta=1e-6, define q_i=n_i+eta.
Given a block B, its exact optimal density is sum(q_i in B)/(size(B)*sum(q_i)).
Its unnormalized profile Q contribution is C_B*log(C_B/size(B)); the omitted
row-normalization term is independent of partition. Prefix dynamic programming
maximizes the sum of block contributions minus lambda*(number_of_blocks-1)
over all contiguous partitions, separately by state. Positive pseudocounts
prevent irreversible support exclusion and are included in the objective.

The observed objective is LL + eta*sum(log PMF) - lambda*sum(extra_blocks).
For fixed lambda, exact penalized Q maximization guarantees this objective
cannot decrease under exact EM; unpenalized LL need not be monotonic when the
number of blocks changes. The script checks the penalized objective explicitly.
The objective uses posterior counts from observations, not decoded labels,
true supports or generator duration probabilities. Nuisance parameters remain
known/frozen and valid starts/right censoring follow the verified E-step.

The predeclared bounded pilot uses fresh seed1901,64 training and16 held-out
sequences of300 rows. Inner48/16 likelihood selection compares fixed factors
0.5/1/2 with lambda=factor*log(inner training sequence count),six updates each;
ties prefer stronger penalty. Refit scales the selected factor by log(64)
and runs ten updates, paired with a fixed full-support width16 ten-update
control. Outer data and truth evaluation occur only after selection and both
fits. All38 M-steps are fixed before evaluation. This heuristic penalty scale
is an experimental contract, not a claim of BIC consistency or optimality.
Nine new tests and24 unchanged focused cases verify mechanics; statistical
results are pending. Production defaults and retained oracle controls remain
unchanged. Fresh data avoid reusing the previous diagnostic held-out set,
but one short pilot cannot establish general recovery or Trading usefulness.

## Adaptive pilot result — 2026-10-13

Step18b completes the predeclared38-update seed1901 pilot at source commit
324de0e100b44ecd3a85fad21b91170c589b56d1. Raw evidence is in
docs/validation/results/adaptive-duration-pilot-native-20261013.json.
Inner validation LL for factors0.5/1/2 is-10413.07914/-10414.63996/-10412.45309;
factor2 wins, with refit penalty8.317766 per additional block.

| Fit | PMF TV: bull / bear / range / chaotic | Mean RMSE | Held-out gain over full-support uniform, nats/row | Full gate |
| --- | --- | --- | --- | --- |
|adaptive|0.175780 /0.408261 /0.461643 /0.291849|22.240347|+0.00543874|FAIL|
|fixed width16|0.176501 /0.156730 /0.207276 /0.229820|5.693451|+0.00961770|FAIL|

Learned adaptive blocks have zero-based exclusive-end boundaries:
bull[0,76),[76,161),[161,260); bear/range[0,260); chaotic[0,59),[59,260).
Both bear and range remain uniform, with mean130.5. These are training-derived
blocks covering every configured age, not generator support masks. There is
no improved recovery: adaptive held-out gain is lower and three state TVs are
worse than the paired fixed control, with only a slight bull-TV improvement.

The ability to represent narrow boundaries does not mean penalized EM will
learn them from a uniform start. A strong complexity penalty can retain a
one-block solution, and short inner validation may select that solution.
Here the selected factor and resulting uniform rows are observed facts;
separating initialization, penalty and small-sample selection effects requires
a new controlled experiment. The new DP solves its count-based M-step globally,
but does not guarantee a global observed-likelihood optimum. Its correct
penalized objective is distinct from fine-PMF recovery.

All source/data/selection hashes,38 updates, PMF coverage/normalization, reported
metrics/gates, penalized-objective monotonicity and public numerical checks
pass. Maximum raw duration-gradient error is5.08e-14; likelihood/filter errors
remain below the original gates. This is a verified negative single-seed pilot,
not generic recovery or a production change. Next: predeclare a fresh-data
comparison of weaker penalties and an observation-only EM warm start before
spending time on another1024-sequence batch. Preserve these failed results and
the previous family-bound evidence; do not tune against this outer held-out set.
