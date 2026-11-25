---
title: Model verification
description: Canonical test ownership, verification commands, and model-core evidence.
type: guide
status: active
language: en
owner: NHSMM
created: 2026-09-26
updated: 2026-11-25
---

# Model verification

Canonical model, mathematical reference, distribution, training, runtime, artifact
and validation-snapshot tests have migrated to the [Nautilus ML stack](https://github.com/awa-si/nautilus/tree/main/tests/ml_stack/nhsmm).
There is no duplicate tests/ tree in this model package. Model implementation
remains here; this core never imports the host.

Install this source checkout editable in the research environment, then run from
Nautilus:

```bash
python -m pytest -q tests/ml_stack/nhsmm
python -m pytest -q tests/ml_stack/nhsmm/reference
```

The independent probability-space HSMM reference moved unchanged. Assertions,
numerical tolerances, gradient/likelihood/filtering/Viterbi/finite-D+ checks, and
all surviving core behavior tests are retained. Run the full model suite for
changes to shared inference, distributions, context, configuration or artifacts.
Do not hide failures with deselections or changed tolerances; report CUDA skips.

Oracle/benchmark helpers now live in Nautilus adapter.qlib.nhsmm_research.
Frozen fixtures/input reports live in tests/ml_stack/fixtures/nhsmm. Tests no
longer require research files in this core checkout. Run CLIs from Nautilus
with python -m and the new module name. Oracle remains OFF and opt-in.

## Local verification and canonical test ownership

NHSMM does not maintain GitHub Actions workflows. Local checks remain:

```bash
ruff check nhsmm scripts
black --check nhsmm scripts
python scripts/smoke_cpu.py
```

The public smoke verifies finite/D+ Float32/64 causal batch/stream parity,
artifacts and fresh reset; it is not the full mathematical suite.

The full canonical test suite is in `awa-si/nautilus/tests/ml_stack/nhsmm`.
For any NHSMM source change, test the exact source against a pinned Nautilus
test-owner SHA. Record JUnit results, source identity, CUDA skips and limits.
Historical results do not qualify subsequent source revisions.

Nautilus owns its own CI configuration independently. Removing NHSMM Actions
does not remove or disable Nautilus CI.

## Static HSMM external likelihood check (2026-11-24)

The causal, finite-support static HSMM likelihood at core
`d47128d645cba83bc6a538181937cc875b0bdbdf` was checked against
[LaMa 3.1.0](https://cran.r-project.org/package=LaMa)
(`forward_hsmm`) with fixed, untrained Float64 parameters: K=3, D=8,
state-specific duration PMFs, duration-independent embedded transitions
with zero diagonal, explicit initial probabilities and Gaussian emission
densities. Both implementations received the same parameters and
emission densities; no optimizer or parameter fitting was compared.

The observations were BTCUSDT 1h completed bars from 2025-12-01 through
2026-01-11, verified and transformed using the Nautilus research reader
at `efc8d8f6febde003c31743536e481f3e636bfa42`.
The close/bar-clock SHA-256 was
`419bb42fb74d80c526e8eb0f6fd4620f61dfcc311ccbf20a1723142bd4e16b34`.
Train-only standardized log-close returns produced 335 train, 168
validation and 504 historical-test observations. For each split, lengths
8, 9, 24 and the full length were compared (12 likelihoods). With
LaMa `eps=0`, the maximum absolute log-likelihood difference was
`3.98e-13`; full-sequence NHSMM/LaMa values were
`-492.020926500353/-492.020926500353` (train),
`-268.388063654549/-268.388063654549` (validation) and
`-652.320056313805/-652.320056313805` (historical test).
With LaMa's default `eps=1e-10`, the maximum difference was
`6.28e-10`. AWA comparison jobs:
NHSMM `2143dd03e6d749598fab637ff4141856`,
LaMa with `eps=0` `da8821c4df9547e7b93f7a4b6edc7112`,
difference calculation `b6638dea9685491d83b6f4b213138da9`.

This establishes numerical agreement of the **causal static HSMM
likelihood in the shared model subclass** on those fixed inputs. LaMa's
direct `forward_hsmm` call did not evaluate prefixes shorter than D=8
(`subscript out of bounds`). This check does not establish external
agreement for duration-dependent or time-varying transitions, dynamic
duration hazards, the non-causal completed-segment objective, gradients,
training, decoding or predictive quality. The D+ tail is checked separately below.
No model code was changed.

## Dynamic causal HSMM external likelihood check (2026-11-24)

The causal finite-support HSMM at model-code revision
`d47128d645cba83bc6a538181937cc875b0bdbdf` was checked against
[LaMa 3.1.0](https://cran.r-project.org/package=LaMa) using its
inhomogeneous `forward()` engine on 24 expanded `(state, episode age)`
states (K=3, D=8). The independent R mapping built a transition matrix
for each boundary `t -> t+1` from the exported `q_t(k,d)` and
`A_t(i,a,j)`: continuation uses remaining-duration survival;
episode end uses the age-specific hazard times the destination
probability, including self-resets to age one. Emission density for each
expanded age is the original state's density. The initial mass is at
age one. Every expanded transition row summed to one within `1e-12`.
The last episode remains right-censored. This is an independent forward
recursion, but it uses NHSMM's exported component probabilities.

The same verified BTCUSDT 1h splits and bar-clock/close digest recorded
in the static check above were used. Two fixed-parameter checks passed:

- Controlled time-varying duration PMFs and time- and age-dependent
  transitions, including self-transitions, were evaluated at prefix
  lengths 2, 3, 6, 8, 9, 24 and the full split length. Across 21
  likelihoods the maximum absolute difference was `4.55e-13`.
- The actual untrained NHSMM distribution heads were evaluated with
  `causal=True`, `context_dim=1`, `seed=17`, `dropout=0`, Float64,
  no duration tail, and explicit `CausalContext`. Emission context
  `E_t` used the previous standardized return (zero at the first row);
  outgoing-boundary context `U_t` used the current return. The heads'
  duration and transition probabilities varied by up to `0.0209`
  and `0.0507` respectively within a split. Across the same 21
  lengths the maximum absolute difference was `6.82e-13`.
  Full-sequence log likelihoods were
  `-495.671623710473` (train, 335 rows),
  `-266.071789772465` (validation, 168 rows) and
  `-604.040753525867` (historical test, 504 rows).
  AWA NHSMM job `475a3559bbbd4c63abe2c33f6a9f6400`;
  LaMa `forward()` job `4ee09f88aaaa43988d72b8d8d3d218d9`.

These results establish numerical agreement of the **dynamic causal
finite-support likelihood** for the tested parameter and context paths.
The R state expansion is a model-to-HMM mapping, not a second
implementation of NHSMM's duration/transition heads. It does not
establish their parameter gradients, optimization/training,
non-causal completed-segment likelihood, decoding, streaming behavior
or predictive quality. The D+ tail is checked separately below.
No model code was changed.

## Dynamic D+ tail external likelihood check (2026-11-24)

The canonical `ModelConfig.for_mode("nhsmm")` causal model with K=3,
D=8, `duration_tail=True` and `duration_tail_context=True` was
checked against the independent LaMa 3.1.0 `forward()` engine on
24 expanded state-age states. It used the same verified BTCUSDT 1h
splits and source digest as the static check above, Float64, fixed
untrained model parameters, explicit `CausalContext` with previous
return for emission and current return for the outgoing boundary,
seed 17 and zero dropout. The tail-context head weight was set to
0.18 to exercise a genuinely varying tail hazard; no fit was run.
The resulting per-boundary D+ end probability ranged from
approximately 0.263 to 0.722 across the three splits. The package's
no-self-transition constraint was active.

For ages below D, the independent R state expansion used the
remaining-duration survival ratio and end hazard. At age D it used
the exported context-dependent D+ end probability for an episode
reset and its complement for continuation in the same age bucket.
It applied the exported age-dependent destination probabilities on
end branches. Every transition row summed to one within `1e-12`.
LaMa independently propagated the expanded chain and scored the
same emitted densities; the active last episode was right-censored.

At lengths 2, 3, 6, 8, 9, 24 and the full split length, the maximum
absolute log-likelihood difference across **21 comparisons** was
`2.84e-13`. Full-sequence NHSMM log likelihoods were
`-505.640479223667` (train, 335 rows),
`-271.868521308586` (validation, 168 rows) and
`-639.747969664247` (historical test, 504 rows).
AWA jobs: NHSMM `7f9b71db135e4704a5fea7bc85cef4df`,
LaMa `dce4febd672b461f85b76432e1c5f3c3`,
difference calculation `d88f2ef175a241738b2575857fbebc0b`.

This establishes numerical agreement of the **causal dynamic D+
likelihood** for the tested head outputs and context schedule.
The external engine shares exported NHSMM component probabilities;
this check does not independently validate head parameterization,
gradients, training, decoding, streaming or predictive quality.
No model code was changed.

## Dynamic D+ gradient and one-step update check (2026-11-24)

At the same model-code revision, LaMa 3.1.0 independently evaluated
the causal D+ likelihood after perturbing nine actual NHSMM parameters
one at a time: initial logit, duration logit and context weight,
duration/age-dependent transition logit and context weight, Gaussian
emission mean and log variance, and tail base and context weight.
For each perturbation the NHSMM heads re-exported initial, duration,
transition, emission and tail probabilities; the independent R
state-age expansion and LaMa `forward()` recomputed the score.
Central differences used a parameter step of `1e-4` and were
compared with PyTorch autograd on 24 train and 168 validation BTCUSDT
1h rows (18 scalar derivatives). The maximum absolute error was
`3.42e-7`; the maximum relative error was `1.61e-6`
(the latter on a gradient of magnitude `0.00243`).
AWA export/autograd job `6bb369f0a38d4d808cf85fd0c5fe00ec`;
LaMa evaluation `f7b64bd57d8442bb9fe5394aca67a170`.

One explicit `torch.optim.Adam(lr=1e-3)` step minimizing the public
negative log likelihood on the 24-row train prefix changed 40 model
parameters. NHSMM log likelihood rose from `-39.442418262996554`
to `-39.341631417656735`. LaMa independently reproduced the
before/after likelihoods with absolute differences of `4.26e-14`
and `3.55e-14`; the external improvement was `0.100786845339897`.
AWA NHSMM step job `8f8ad72401d8404497f0391c3849875c`;
LaMa step job `95275ce047584baeb6a2c1c378e34d8e`.

This tests selected first derivatives of the complete causal D+
likelihood through actual heads and verifies the likelihood after one
explicit Adam update. Finite differences are approximate. The
repository's full `fit()` orchestration, convergence, other parameter
coordinates, second derivatives and out-of-sample quality were not
externally validated. No model code was changed.

## Opt-in Rust HMM likelihood verification (2026-11-24)

The core checkout's Python sources were copied byte-for-byte into the
source snapshot used by Nautilus `tests/ml_stack/nhsmm` at test-owner
`efc8d8f6febde003c31743536e481f3e636bfa42`. With the research
cdylib built from this checkout and explicitly enabled in the test environment, the canonical CPU suite passed
**1,701 tests, 3 CUDA-unavailable skips, 0 failures** (254.95 s; AWA
job `d1456a9cf9c3476da4e3b34bc7b2b280`). Focused checks cover
single-row and 19-row score/gradient/4-update parameter agreement,
state_dict/config invariance, unsupported input rejection, second-
derivative rejection and disabling the option. The source snapshot was
not a wheel installation; no CUDA or broader native shape coverage was
claimed. A local no-isolation wheel build included
`nhsmm/models/research_hmm_rust.py` and excluded the research
scripts and native library; wheel installation was not exercised.

## Finite HMM generative forward specialization (2026-11-24)

The finite causal D=1 forward path without a duration tail was compared with
the previous general state-age path at core base
f195a004599fcb7178c76a19b678b59718f433bb. The new
nhsmm/models/forward.py SHA-256 was
f64e00e5f668d9e3e17a4133c0f909afca8c4b1d6628c510a7790b3876c2bd20.
The 26 Python source files in the Nautilus test snapshot matched the working
core checkout; the test owner was Nautilus
d2fb4e83e86d22d30784f20dcc9ab7ca81fc0511. The canonical CPU suite
passed 1,698 tests, with three CUDA-unavailable skips and zero failures
(232.67 s). JUnit SHA-256:
8a4cd2703681b9aaff2008c59dbe1779273860c48de1e28cbdf257fdfd5f22ff.
Focused bit-exact forward and emission-offset tests passed 100 cases.
Separate Float32/Float64 ragged and full-path gradient comparisons passed;
the 335-row BTCUSDT 1h training slice gave identical -500.43743896484375
log likelihood and identical gradients for every model parameter at fixed
weights.

An isolated single-thread CPU comparison on that same 335-row BTC slice,
after one warmup per path and eight alternating timed repetitions, measured
forward plus backward median 0.4698 s (general) versus 0.3591 s
(specialized), a 23.6% reduction. Means were 0.4801 and 0.4073 s;
nearest-rank p95 values were 0.6843 and 0.7110 s. This is a local
fixed-parameter kernel result, not a full-fit, accelerator, or market-quality
claim. The optimizer, state dict, and artifact schema are unchanged.

## Current local canonical CPU qualification (2026-11-23)

The exact NHSMM model core at
`faca7a691ae914f863798dacbbfd81493b9fbef9` passed the Nautilus
canonical suite owned by `0c4b6359ee17fe243fca9b814aa15e5276317e74`:
**1,878 passed, three CUDA-unavailable skips, zero failures or errors**.
The wheel-installed model files were verified byte-for-byte against all 26
Python sources in the exact core checkout; the source-root test fixture passed.
This is local CPU evidence, not a hosted CI run, market validation or
live-trading qualification. The [source-bound report](https://github.com/awa-si/nautilus/blob/2fbdaf63858e52e5b3528270f9bcb8178c049387/docs/nhsmm-main-canonical-qualification-20261123.json)
records test-owner provenance, JUnit SHA-256, environment and limitations.
Later documentation-only commits do not alter the tested Python sources;
any further model-code change requires renewed qualification.

## P1 local runtime and host integration (2026-11-23)

The independently selected Nautilus integration suite passed **48 tests**,
with one GBM-free-isolation skip. A focused numerical/reference and streaming
suite passed **56 tests**. On CPU, artifact-loaded seeded synthetic runtime
baselines for batch size one, 96 steps and 24 warmup steps measured median
one-step latency of 1.90 ms (K=3, F=2, D=16) and 2.96 ms
(K=4, F=4, D=32, tail enabled). Both preserved bounded filter state.
The [source-bound P1 evidence](https://github.com/awa-si/nautilus/blob/fd9b3369bb937f6ca7a4380e080abf51c7096be1/docs/nhsmm-p1-runtime-integration-20261123.json)
includes JUnit and artifact hashes, p95/mean latency, separate Python heap
allocations, workload dimensions and limits. No optimization speedup or
production-market latency claim follows from these measurements.

## Runtime cache cleanup verification (2026-11-23)

Model source `2b4c658e2a744bed9e388812389a3893ca4e2b76` consolidates
version-aware score-cache keys and removes a no-op observation-shape branch.
The exact modified 26-file Python source snapshot passed **1,878 canonical
tests** with three CUDA-unavailable skips, zero failures and 17 third-party
warnings using Nautilus test owner `fd9b3369bb937f6ca7a4380e080abf51c7096be1`.
JUnit SHA-256: `95c9327baa1e6cabadb08d85f9925cf75375ba97013c4621665e58f603a5985a`.
The focused runtime/adapter suite passed 28 tests; mutation, inference-mode
and covariance-floor cache invalidation checks passed. Ruff lint and format
passed. No model probability or artifact contract was changed. This is local
CPU evidence; CUDA and trading qualification remain separate.

## Base model cleanup qualification (2026-11-23)

The behavior-preserving `nhsmm/models/base.py` refactor at source commit
`32ea469488b3f6d22c2db6478d0d46efad4a6d35` consolidates identical
encoderless/base-context initialization paths and moves the unchanged set of
30 permitted per-call optimization overrides into a module-level frozenset.
No public API, objective, model parameter or artifact schema changes.

A byte-identical 26-Python-file source snapshot passed the full Nautilus
canonical suite at test-owner SHA
`19a23c0356c43a2d05a809c4b9b68f19c065edd8`:
**1,878 passed, three CUDA-unavailable skips, zero failures** (401.32 s).
The focused configuration/context/optimizer suite passed 436 tests with
three CUDA skips. Ruff lint, formatting, and exact override-field parity passed.
JUnit SHA-256:
`51385fa49eaae72243f7bf7c9bca28b8746fcad56e9c9c88e920968f10eab847`.
This is local CPU verification and does not qualify accelerator or market use.

## Distribution validation cleanup qualification (2026-11-23)

The refactor in `nhsmm/distributions/default.py` at source commit
`38ea0af0cfc379ec63e8601eca6739468922d877` consolidates duplicated
categorical observation validation in the standalone categorical distribution
and neural categorical wrapper. Device handling, label range/error contracts,
probability computations, parameter shapes and artifact formats are unchanged.

A byte-identical 26-Python-file snapshot passed the full Nautilus grouped
canonical suite at owner SHA
`19a23c0356c43a2d05a809c4b9b68f19c065edd8`:
**1,878 passed, three CUDA skips, zero failures** (431.51 seconds).
The focused distribution/configuration tests passed 233 cases.
Ruff lint, formatting, and `git diff --check` passed.
JUnit SHA-256:
`2bf58be4e33997abf217f847cbe586579f1ed4744c8da20530ae65dcfe836e9d`.
This is local CPU verification; accelerator and market qualification remain separate.

## Shared forward tail-dispatch cleanup qualification (2026-11-23)

The code change at source commit
`15ec2d9553fb4b40ea12340869f3d6d010bc39f2` moves the identical
static/context-conditioned duration-tail dispatch from the causal and
noncausal forward paths into one internal helper in `nhsmm/models/forward.py`.
Forward recursions, causal timing, log probability calculations and artifact
contracts are unchanged.

The byte-identical 26-Python-file source snapshot passed the grouped canonical
Nautilus suite at test-owner commit
`19a23c0356c43a2d05a809c4b9b68f19c065edd8`:
**1,878 passed, three CUDA skips, zero failures** (397.02 seconds).
Focused filtering/duration-tail/model-mode tests: **389 passed**.
Ruff lint and formatting passed. JUnit SHA-256:
`9e0b533d74cc904a3e51370a0ff2e69196ee2cf1fe8136d659b751cc831130cd`.
This is local CPU qualification, not CUDA or market validation.

## Public filter validation boundary qualification (2026-11-23)

Code commit `8d9b73f734a4def7f38684af03ee1e4604e1c800` extracts
`_validated_filter_step_scores` from the public `filter_step` entry point.
It retains the exact shape, dtype, device, finite-support and normalization
checks in their existing order, returning normalized scores to the unchanged
`_filter_step_normalized` internal kernel. No public signatures, mathematical
operations or artifact schemas were modified.

The byte-identical 26-Python-file source snapshot passed the grouped
canonical Nautilus suite at test-owner commit
`19a23c0356c43a2d05a809c4b9b68f19c065edd8`:
**1,878 passed, three CUDA skips, zero failures** (398.29 seconds).
Focused filtering/runtime/validation checks: **356 passed**.
Ruff lint, formatting and source-byte parity passed. JUnit SHA-256:
`f670cab7e551fe98cfda17218feabcff10c6fe20b6a7e957f6f4adc712a6f480`.
This is local CPU evidence; CUDA and real-market qualification are separate.

## Context alignment validation cleanup (2026-11-23)

The source change at `a09247709e9bf5c6158d3fe4e835a89bd57fa1b5`
consolidates the four valid shape branches of `align_context_tensor` into
one shared availability-mask and finite-value validation exit.
The accepted [H], [T,H], [B,1,H] and [B,T,H] shapes, error messages,
causal masking semantics and tensor operations are unchanged.

Verified on a byte-identical 26-Python-file source snapshot against
Nautilus owner `19a23c0356c43a2d05a809c4b9b68f19c065edd8`:
**1,878 passed, three CUDA skips, zero failures** (426.72 seconds).
Focused context/config tests: **173 passed, three CUDA skips**.
Ruff lint and formatting passed. JUnit SHA-256:
`907451498d4b4cbe26d4e019997faee10b387995a3f4a9c1ddd9b213a91ecc54`.
These are local CPU checks, not GPU or market qualification.

## Source-bound repair and context evidence

Current behavior is owned by [the model/runtime contract](model.md) and the
[HSMM mathematical contract](hsmm-mathematical-contract.md). The following
reports preserve historical results for their named source revisions; their
pass counts do not certify the current branch head.

- **Mathematical repair (2026-10-19):** the source-bound audit corrected five
  issues: normalized non-causal likelihood, exact first-order tail-boundary
  gradients, segment support zeros, reachable-duration support, and artifact
  precision. The fixed snapshot added 59 regressions, matched 540 independently
  enumerated forward/MAP cases (maximum evidence residual
  `2.220446049250313e-16`), and passed 1,156 tests with three CUDA skips;
  Ruff and Black passed. Public method signatures and state-dict paths were
  unchanged, and a causal v2 reference artifact retained its exact fixed-prefix
  score. No real training ran.
  A three-repeat CPU A/B (`K=3,F=2,D=32,B=1,T=256`) measured median forward
  `0.1754 -> 0.1981 s` and backward `0.1919 -> 0.1635 s`; this mixed result
  supports no general speedup claim. Full hashes and run identity are in the
  [Nautilus repair record](https://github.com/awa-si/nautilus/blob/main/docs/nhsmm-correctness-repair.json).
- **External-context timing and streaming fallback:** exact-source reports retain
  timing checks (1,209 core tests, three CUDA skips, 30 host harness regressions),
  base-score checks (1,215 core tests, three CUDA skips; 162 host integration/TA
  tests plus 11 subtests), and masked-context checks. See the
  [timing report](https://github.com/awa-si/nautilus/blob/c45d94f4a8b6c87870ee2dc8f775c9609128c027/docs/nhsmm-causal-context.json),
  [base-score report](https://github.com/awa-si/nautilus/blob/cf3a8ef59fbbdf2484f5679307568b2c812f38bf/docs/nhsmm-context-fallback.json), and
  [availability report](https://github.com/awa-si/nautilus/blob/5adf39f034e02c72f2c5fd42b133669233865370/docs/nhsmm-masked-context.md).
  These are historical synthetic/integration results, not evidence of market
  advantage.


The canonical test_forecast_precision.py module checks rare finite/D+ end and state-change risks against 80-digit Decimal references in Float32/Float64, hard-support forecast gradients against independent probability formulas, geometric first/second derivatives, result identities, and separate duration/transition target learning.


## Central tunable verification

The canonical test_central_tunables.py checks invalid/valid overrides, actual probability laws, context/encoder propagation, Adam/scheduler/patience settings, supervised precedence and current/legacy artifact predictions. Host test_nhsmm_config_tuning.py checks the bounded search policy, grid, split separation, request guards and native Qlib/MLflow recording.

Alternate optimization-config regressions verify the actual annealing temperatures,
main/refinement Adam and clipping settings, and rejection of ignored construction
or supervised-only overrides before any parameter update.

2026-10-21 optimizer-config verification: 125 focused tests passed. The full
model plus tuning/auditor run passed 1474 tests with three CUDA skips; one
metadata setup error was resolved by exporting the original pyproject.toml.
That unchanged test passed separately (1475 distinct passes total). No model
code changed between those runs; Ruff and Black passed for the complete core.

Temperature-boundary verification (2026-10-21): test_temperature_numerics.py
adds 24 cases, including 100-digit Decimal schedule references, bit-exact
ordinary arithmetic, actual extreme annealed/constant training temperatures,
and rejection of an underflowing positive ratio/target combination before
model mutation. The complete model plus tuning/auditor suite passed 1499 tests
with three CUDA skips. Initial focused testing exposed an invalid zero-ratio
test setup; it was corrected to a valid underflowing configuration, without
changing configuration bounds. Corrected focused run: 39 passed.

Training-finiteness verification (2026-10-21): 18 new injected-failure cases
check objective/score finiteness, NaN/Inf gradients, total-norm overflow from
finite elements, mode/trainability restoration and absence of successful
completion telemetry. The focused training bundle passed 57 tests; the full
model plus tuning/auditor suite passed 1517 with three CUDA skips. The prior
source reproduced a NaN-gradient Adam update corrupting emission locations.
The initial reproduction test used a nonexistent parameter name; corrected
mu injection established the actual failure. Ruff and Black passed.

The canonical test_supervised_setup.py injects Adam-construction and mode-switch
failures in training/eval and base/context-only modes. It verifies unchanged
weights and restored trainability, including parameters frozen by the caller,
and retains coverage of the no-trainable-parameters cleanup path.

2026-10-21 setup-restoration gate: 28 focused checks passed; full model plus
affected native ML-stack bundle passed 1566 with three CUDA skips. The first
regression reproduces incorrect frozen base flags after Adam setup failure
on core 371e19c. No defaults, supervised scores or training budgets changed.

## Exact finite-support boundary derivatives

Canonical test_hsmm_boundary_derivatives.py adds 45 cases for the exact D+
h_tail=1 boundary: raw-parameter path-enumeration gradients in Float32/64,
D=1/2/3, self resets, mixed interior/boundary states, forecast first and mixed
second derivatives, next-state derivative conservation, ragged and empty
prefixes, one-sided numerical differences, projected supervised Adam direction,
and single evaluation of stochastic heads. The original 2026-10-21 gate
restricted the score-boundary VJP to first order. That restriction still
applies to marginal likelihood. At source 8c0998c, supervised filtering
separately supports exact second derivatives and rejects third-derivative
graphs, as covered by the usable-supervised-HSMM gate below.
Forecast values and artifact schemas remain unchanged. This is mathematical
verification, not empirical trading qualification.

2026-10-21 boundary gate: 45 new cases pass; the final same tests on core
84d38da have 44 failures and one pass. Full canonical model plus affected
native Qlib/MLflow, research publication, configuration/input audits and
retained-replay auditor bundle: 1647 passed, three CUDA-unavailable skips.
Ruff and Black pass for the complete core. No defaults, artifact schema,
market fit budget or reserved data changed. Exact source and JUnit evidence
are retained in Nautilus docs/nhsmm-hsmm-boundary-verification.json.

## Long-prefix HSMM and streaming state verification

Canonical test_hsmm_streaming_integrity.py adds 26 cases. Four 1440-row
Gaussian HSMM sequences (finite/D+ and Float32/64) compare every batch and
streaming state-age posterior with a separately constructed NumPy long-double
expanded-state probability chain. Its inputs are raw weights; it never calls
production distribution scoring, hazard or filtering helpers. Final likelihood
per row, one-step state/end probabilities and frozen-duration survival forecasts
are checked independently. Reset starts 128 fresh suffix observations from age
one. Two custom exact-support models retain deterministic 100-row age cycles.

The same module checks variance-floor cache invalidation, inference-tensor
parameter updates without version counters (including incremental LSTM gates),
unchanged persistent state during inference, invalid timestamp rejection before
state advance and valid datetime continuation. Model reset clears runtime caches.
Posterior comparison tolerances are 2e-5 for Float32 and 2e-12 for Float64;
log-evidence per-row tolerances are 1e-5 and 1e-10. These are bounded numerical
checks, not a proof over arbitrary sequence lengths or custom distributions.

2026-10-21 streaming gate: 26 new cases pass; the same final tests on core
c25da77 have 17 failures and nine passes. Focused runtime/encoder bundle:
47 passed. Full canonical model plus affected native Qlib/MLflow/RD-Agent,
research publication and input/configuration/retained-replay audit bundle:
1673 passed, three CUDA skips. Complete core Ruff and Black pass.
No model default or artifact-schema change, new market fit or reserved test
score. Evidence: Nautilus docs/nhsmm-streaming-verification.json.


## Viterbi common-offset gate

The host test_viterbi_numerics.py verifies MAP decisions under large shared
Gaussian density offsets, dynamic finite/D+ raw-weight path enumeration,
ragged empty/single-row inputs and 1440-row weak-preference causal HMMs.
Existing exhaustive decoder tests retain exact ties and hard support.
See [the numerical contract](model.md#viterbi-common-offset-stability)
for offset cancellation, noncausal endpoint limits and rounding boundaries.


## Causal filter emission-offset gate

Canonical host test_filter_offset_invariance.py checks identical Gaussian
emissions with large finite observations, ragged empty/single-row sequences,
batch/runtime posteriors, latent-law evidence gradients, and raw-parameter
path enumeration for likelihood/filter scores and all parameter/E/U gradients.
Finite/D+ durations, exact h_tail=1 and differing hard transition support sizes
are included. The host report records baseline failures, source/JUnit hashes
and full native ML-stack validation.


## Artifact publication and real Gaussian derivative gates

Canonical host test_artifact_publication.py injects serialization, file-sync
and replacement failures for absent and existing targets, verifies that the
target is unchanged until serialization completes, and checks Float32/64
Gaussian/Student-t round trips and existing POSIX permissions.

test_gaussian_offset_derivatives.py uses real Gaussian scoring and independently
enumerated prefix-conditioned occupancies to check mean, raw-variance and
observation gradients at large positive and negative observations. It covers
HMM, finite HSMM and D+ (D=1/3), Float32/64, and marginal/filter objectives.
Derivative errors are compared after division by their natural observation
scale (x or x squared); it does not claim arbitrary absolute precision.


Scoped CPU qualification and operational limits: [production.md](production.md).

The historical 2026-10-22 cross-repository test-owner pin was superseded
by owner-local canonical verification. NHSMM GitHub Actions definitions were
later retired; historical run records do not qualify new model revisions.

## Independent model-mode qualification

Canonical test_model_modes.py verifies static construction without neural
networks, raw-weight path-enumeration scores/posteriors, synthetic supervised
HMM/HSMM fitting, v3 artifact round trips, continued fitting, streaming and
external-context fallback. Public scripts/smoke_cpu.py also exercises HMM,
finite/D+ HSMM and internally conditioned NHSMM in Float32/64. Profile
selection adds no new inference algorithm or artifact field. See model-modes.md.

## HMM filtering performance regressions

Canonical test_hmm_optimized_filter.py checks bit-exact D=1 no-gradient steps
against the unchanged general kernel at K=2/3/6 and Float32/64, continued
selection of the autograd path, invalid reachable duration rejection,
three-state neural HMM prefixes against independent raw path enumeration,
ragged empty/singleton inputs and future-prefix causality. Existing full model
and serving gates still apply. Timing comparisons use identical artifact
weights/inputs and alternate old/new code; score tolerances remain unchanged.

## Extended HSMM and HMM preservation gate

Canonical test_hsmm_extended_validation.py checks 17/64-age finite and D+
batch/stream frontiers against independent raw-weight expanded Markov chains,
ragged dynamic age-dependent transition/duration adjoints against a separate
probability recurrence, interior Hessian-vector products and zero-sensitivity
duration suffix derivatives with exact hard support and separated scores.
Hazard batching excludes padded and unused final boundaries. Full model and
existing HMM optimization/usability gates remain required.

## Usable supervised HSMM gate

Canonical test_usable_hsmm.py verifies boundary provenance and censoring counts,
actual temperature/gate-aware smoothed laws, unchanged emissions/initial weights,
invalid-fit immutability, stationary state-age priors against independently
expanded Markov chains, batch/stream/reset parity, exact h_tail=1 Hessian-vector
products and explicit third-derivative rejection. Existing HMM gates remain.
Host synthetic calibration is owned by Nautilus usable_hsmm.py through native
Qlib TrainerR/MLflow, with a sealed split/source/candidate budget.

The completed 2026-10-24 native calibration is recorded in
[Nautilus](https://github.com/awa-si/nautilus/blob/1bbcd0b2ffd5d5d8766d702c8a0185ae9b351594/docs/usable-hsmm.md):
nine declared profiles pass; geometric controls retain HMM.
The unchanged core's canonical source pair is NHSMM 8c0998c / Nautilus 37028d7
(1852 passes, three CUDA skips). The later host-only scorer/replay correction
has 134 targeted rechecks; it does not replace or rerun the full canonical gate.

## Deterministic training-score reuse (2026-11-24)

For the built-in, encoder-free model with zero dropout and a constant
temperature schedule, `fit()` uses the accumulated pre-update likelihood
as its convergence/telemetry target score. The final post-update score is
still recomputed. Stochastic, annealed, custom-distribution and
method-overridden paths retain separate target evaluation. The public
configuration, artifact and state-dict schemas are unchanged.

On the verified BTCUSDT 1h train split above (335 rows, K=3, D=8,
dynamic D+ tail, Float64, one thread, 20 Adam updates), the unmodified
implementation and a temporary score-reuse prototype produced identical
parameter tensors, optimization history, final score
(`-460.4056754886496`) and summary. The same-host prototype comparison
measured 26.10 s versus 14.54 s; a later source-level run measured 18.96 s
on the optimized path, so these timings are workload-specific and variable.
Instrumentation confirmed 41 versus 21 causal forward calls in 20 updates,
including the final score. The final source snapshot passed the
Nautilus-owned training suite (262 tests) and full model suite (1,698 passed,
six skipped: three CUDA unavailable, three opt-in Rust library unavailable).
Ruff and Black passed. Test-owner revision:
`efc8d8f6febde003c31743536e481f3e636bfa42`.

## Opt-in Rust dynamic HSMM likelihood verification (2026-11-24)

A user-built Rust cdylib implements a log-domain expanded state-age
forward recursion and its first-order adjoint for one CPU Float64 causal
finite/D+ HSMM sequence. The native binary remains outside the wheel;
the opt-in Python wrapper and source probe are in the core repository.

With K=3, D=8 and the verified BTCUSDT 1h split above, eleven actual-head
prefixes/full sequences (T=2,3,8,24,168,335,504 across finite/D+ cases)
matched the PyTorch state-age likelihood within 2.27e-13 absolute and all
five input-tensor gradients (initial, emission, transition, end and
continuation log scores) within 5.22e-15 absolute. The integrated
`fit()` comparison on 335 train rows, 20 Adam updates, constant
temperature and zero dropout reached exactly the same final score
(`-460.4056754886496`); the maximum parameter difference was
2.14e-15 and the largest target-history difference 1.14e-13.
The API check verified opt-in/disable restoration, unchanged state_dict,
single-row support, rejection of batch and Float32, and explicit rejection
of higher-order derivatives. A wheel build included the wrapper but
excluded the native library and research source.

The same-host integrated 20-update fit A/B took 17.96 s (PyTorch) and
0.57 s (Rust), about 31.7x in this one workload. An isolated 335-row
likelihood-plus-backward A/B with eight alternating repetitions measured
Torch mean/p50/p95 599.91/587.05/774.58 ms and Rust
4.59/3.09/12.20 ms. The latter comparison excludes head evaluation,
hazard transformation and optimizer work; it is not an end-to-end estimate.
An observational 800-update Rust fit with validation checks every 20
updates took 27.64 s while the canonical suite was concurrently active.
The best validation log likelihood was -221.659 at update 180; the final
update-800 validation score was -224.379 despite further training-score
improvement. A single historical-test evaluation of the
validation-selected checkpoint scored -438.739. This is one split and one
seed, not predictive-quality evidence.

A repeatable 180-update fit from the same seed reproduced the
validation-selected checkpoint. Its exported duration, age-dependent
transition, emission and context-dependent D+ tail probabilities were
mapped independently to a 24-state expanded chain in LaMa 3.1.0.
Across train, validation and historical-test prefixes T=2,3,6,8,9,24
and each full split (21 comparisons), the maximum absolute likelihood
difference between the Rust-enabled NHSMM and LaMa was 5.12e-13.
Full-sequence log likelihoods were -384.870257529883 (335 train),
-221.659113280174 (168 validation) and -438.739386605504
(504 historical test). The tail end probability ranged from
1.93e-5 to 0.998 across splits. This checks the trained checkpoint's
forward likelihood using exported component probabilities; it does not
independently reimplement the neural heads or establish forecast quality.
AWA jobs: NHSMM export `d8e8271a198a4a02bf99bb4aa1648e42`;
LaMa comparison `0ce0204347ee4361b9bb657ca4b397e3`
and difference calculation `d0e4fe2df3b941249f1b09f4351dabad`.

## Preliminary one-hour risk forecast (2026-11-25)

The validation-selected dynamic D+ checkpoint at update 180 above was
reproduced from seed 17 and assessed on the same verified BTCUSDT 1h
train (335), validation (168) and historical-test (504) splits. This
is an observational research check, with no new fit or test tuning.
The forecast issued before observing standardized return z_t was
P(|z_t| > 1.508653), where the threshold is the train-only 90th
percentile of |z|. The expanded state-age filter propagated the
exported duration PMFs, age-dependent transitions and context-dependent
D+ tail hazards. The predicted state mixture used the model's
Gaussian emission means/variances at t. Its sequential density
log scores summed to the package's full-sequence likelihood within
1.14e-13 absolute on each split.

For comparison, a Gaussian EWMA variance forecast used the train
mean and variance with fixed decay 0.94 and updated variance only
after each observation. A second baseline used the train Gaussian
mean and variance without updates. All three methods restarted at
each split, matching the model's splitwise context reset. Validation
selected the NHSMM checkpoint by likelihood; neither baseline was
selected by the historical test.

| Historical test (504 hours; 26 threshold events) | NHSMM | EWMA | Train Gaussian |
| --- | ---: | ---: | ---: |
| Mean density log score (higher is better) | -0.8705 | -1.0436 | -1.1466 |
| Event log loss (lower is better) | 0.2135 | 0.2624 | 0.2383 |
| Event AUC (higher is better) | 0.5979 | 0.5435 | 0.5000 |
| Event Brier score (lower is better) | 0.05205 | 0.05096 | 0.05529 |

The NHSMM wins on density score and event log loss here, while EWMA
has the lower Brier score. With just 26 test events, one chronological
split and one seed, this is not a robust predictive superiority or
trading-utility claim. The event threshold is a standardized return
magnitude, not a price-level risk or trading P&L. AWA export job
`da2298c029e748cf93946044e2d6ef3f`; forecast evaluation job
`eca2c613d7bd47018bed2a38e3b72515`.

## Rolling one-hour risk forecast with Rust training (2026-11-25)

An observational expanding-window evaluation reused the 1,007 verified
BTCUSDT 1h standardized returns above. Three chronological, disjoint
168-hour test windows followed training/validation cutoffs
335/503, 503/671 and 671/839 (exclusive indices in the concatenated
series); test index ranges were [503,671), [671,839), [839,1007).
Each fold used its own train-only mean, standard deviation and 90th
percentile of absolute standardized return. The event was exceeding
that threshold in the next 1h return. The forecasting filter used
only observations available before the scored return and reset at
each validation/test split as in the preceding single-split check.

For each fold, NHSMM K=3, D=8, causal dynamic D+ with context-dependent
tail was fit from scratch for 180 Adam updates using the opt-in Rust
likelihood/first-gradient path, seeds 17, 23 and 41, zero dropout,
constant temperature and fixed learning rate 0.001. Update count and
learning rate were held fixed for all nine runs. Gaussian EWMA used
decay 0.94 and train mean/variance; the static Gaussian used the same
train estimates. Neither baseline was tuned to the test windows.
A first attempt at learning rate 0.01 completed seed 17 in fold one,
then seed 23 reached exact D+ hazard h=1, a deliberately unsupported
Rust training derivative. That partial attempt was discarded and
**all nine runs restarted** at 0.001. No per-window/seed fallback or
checkpoint selection was applied.

| Test window (events) | NHSMM Brier | EWMA Brier | NHSMM event log loss | EWMA event log loss | NHSMM density log score | EWMA density log score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 (8/168) | 0.04791 | 0.04830 | 0.20869 | 0.22867 | -1.10497 | -1.07951 |
| 2 (6/168) | 0.04045 | 0.03475 | 0.19273 | 0.18793 | -1.01222 | -0.83369 |
| 3 (12/168) | 0.06880 | 0.06762 | 0.26751 | 0.25953 | -1.14863 | -1.06704 |

NHSMM entries are the mean of three seed-specific scores within each
window; EWMA is seed-invariant. Across the nine model runs, the
maximum absolute error between the independently reconstructed
sequential density score and the package's whole-sequence likelihood
was 2.84e-14. Mean NHSMM event AUC was 0.696, 0.477 and 0.707 by
window (EWMA 0.561, 0.604 and 0.584). Total fit time across nine
Rust runs was 75.24 s, excluding model creation and evaluation.

The repeated result does **not** establish a consistent NHSMM advantage
over EWMA. EWMA wins density log score in all three windows and Brier
in two; NHSMM wins event log loss in only the first. These are 26
events in total, all on one asset and one contiguous historical period.
The 504-hour period had already been inspected in the preceding
single-split check, so these windows are not a fresh independent
holdout. The smaller learning rate means these scores are not directly
comparable to the update-180 checkpoint at learning rate 0.01.
The evaluation script in the AWA workspace was
`hsmm_lama/rust_dynamic/rolling_forecast.py`
(SHA-256 `5fd7640fbc135ccce0178939ca458015d9cea690bb751f705c209e4e1c5f0ae7`).
AWA jobs: fold one `28422f4c24744067aee469518f08dc09`,
fold two `cc3dd2041b064f85aab94100c74deb0c`,
fold three `6d56c3859b9e493faa07052649170e71`.

## Fresh-period PyTorch forecast comparison (2026-11-25)

This replaces the split-reset forecast protocol in the two preliminary
risk sections above for assessing a live one-hour forecast. Their
numerical Rust/LaMa likelihood checks remain valid. Four previously
unused, nonoverlapping BTCUSDT 1h 28-day periods were fixed before
examining their scores: 2026-02-01..28, 2026-04-01..28,
2026-07-01..28 and 2026-10-01..28. The Nautilus
`adapter.kline_reader.KlineReader` at test-owner
`efc8d8f6febde003c31743536e481f3e636bfa42` verified 24
published-hash-matching bars on every UTC day, 672 contiguous
exclusive-close hourly bars per period and finite positive closes.
The dataset identity was
`1158b60ab26a76b5913acc4723051bda536833cda55857ef7f5b2c13b746f53b`.
The four clock/close SHA-256 digests, in period order, were
`7fda710402a1c642e85d04fc5f4d2c0c41b1688637d7a927d4fdd074430eb3f1`,
`6ff4ad87e132a2c962a040af4952bba612025af8517b68ecb31eb94a92f8a643`,
`e17dc95c140b306fdd1f01802bd2ddd19767c1d53d22948d92ee09e9d7dd86bb`
and `942257c5489c6e1a4294739f19346b3016b0586db6d89afcc674245ba59b3c92`.
Each period yielded 671 returns, `10000*diff(log(close))`, split into
335 train, 168 validation and 168 test observations. Each fold's
standardization and absolute-return event threshold (train 90th
percentile) used train only.

The evaluation used **the default PyTorch model path, not Rust**.
K=3 HMM (D=1), static causal D+ HSMM (D=8) and causal dynamic D+
NHSMM (D=8, context dimension one, contextual tail) each received
50 Adam updates at learning rate 0.01, seeds 17/23/41, one
initialization, zero dropout and constant temperature. No
validation/test-specific hyperparameter or checkpoint choice was
made. The dynamic tail-context weight began at 0.18, as in the
previous research fit. Emission context used the previous standardized
return, and outgoing-boundary context used the current return.
For each period, one state-age filter processed all 671 observations
continuously; no reset occurred at the train/validation/test cuts.
The EWMA baseline (fixed 0.94 decay, train mean and variance) also
carried its variance state across the same cuts. A static train
Gaussian was the second baseline. The event and probability were
scored at each test hour **before** the realized return was fed into
the filter. Across all 36 model fits, reconstructed sequential log
density matched the package's full-sequence likelihood within
3.41e-13 absolute.

| Test period (events/168) | HMM event log loss | Static HSMM | Dynamic NHSMM | EWMA | Train Gaussian |
| --- | ---: | ---: | ---: | ---: | ---: |
| February (6) | 0.18835 | **0.17095** | 0.17372 | 0.27444 | 0.19608 |
| April (7) | 0.19673 | **0.17466** | 0.18779 | 0.19853 | 0.21313 |
| July (7) | 0.19510 | **0.17810** | 0.17953 | 0.23091 | 0.20862 |
| October (15) | 0.29941 | 0.30027 | **0.29566** | 0.32748 | 0.30701 |
| Mean of periods | 0.21990 | **0.20600** | 0.20918 | 0.25784 | 0.23121 |

The three seed scores were averaged for each model and period;
baselines are seed-invariant. Mean Brier scores by method in the same
order were 0.05168, **0.04957**, 0.05077, 0.05198 and 0.05391.
Mean density log scores (higher is better) were -1.16699,
**-1.14820**, -1.15581, -1.27555 and -1.23707. Static HSMM
beat dynamic NHSMM in event log loss in three of four periods; dynamic
won in October. Within-period seed ranges were much narrower than
the variation across periods. There were only 35 test events total.
These are research observations, not proof of generalization across
assets or evidence of trading utility. Fixed 50-update comparisons
also do not establish that either model was optimally trained.

Source preparation script in the AWA workspace:
`hsmm_lama/fresh_prepare.py` (SHA-256
`38e0d5131d98dda8f2cb6859e9ce57434679eeaf6a024d923b25f4d7df3adf35`);
model evaluation: `hsmm_lama/fresh/evaluate.py` (SHA-256
`03d84a7d662a61ae08c59739c175d1a99ee9669fcac851e8e9ac825e428373ea`).
Each completed model/seed result is saved as
`hsmm_lama/fresh/result_period{1..4}_{hmm|hsmm|nhsmm}_seed{17|23|41}_u50.json`
in workspace storage. AWA source preparation job
`b2e1c8473bc54a69b4812e6f471017c2`; completion jobs for
the multi-period model runs include
`36be1cb308bb4146a8a0f2e6eb06eb4d`,
`24596bdcf98345da98a8b923969cc531`,
`0684590fb24a450097a13a1daea3ceef`,
`434253126670461b899d311774a432e5` and
`94a410c6efa949c08160586d4163be23`.

No CUDA, batching, Float32, h=1 tail-boundary gradient or second derivatives
are established. The canonical suite on the exact source snapshot:
1,698 passed, six skipped (three CUDA-unavailable, three opt-in Rust HMM library unavailable), zero failures; Nautilus test owner `efc8d8f6febde003c31743536e481f3e636bfa42`, AWA job `451b619d5880425e8b885d35a23e0b03`.
