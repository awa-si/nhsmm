# Testing and verification

The maintained automated contract is the pytest-discovered suite under `tests/`.

## AWA workspace resources

Repository workspace defaults are defined in `workspace.ini`:

```ini
[resources]
cpu = 4
memory_mb = 8192
storage_mb = 8192
pids = 512
tmp_mb = 1024
```

The 8 GiB storage allocation is sufficient for the complete CPU pytest suite when PyTorch is installed from the CPU wheel index. A generic `pip install -e .` under Python 3.14 currently resolves the CUDA-enabled PyTorch dependency set and can exceed the retained-size guard.

For a CPU-only AWA verification environment:

```bash
python -m venv .venv
.venv/bin/python scripts/install_cpu_dev.py
```

The bootstrap installs `torch>=2.2` from PyTorch's CPU wheel index first, then
runs the normal editable `.[dev]` install. Because the Torch requirement is
already satisfied, pip does not resolve the CUDA dependency set.

## Canonical suite

```bash
pytest -q
```

`pyproject.toml` limits discovery to `tests/test_*.py`. Focused tests are appropriate during development; shared training/runtime changes should also receive the full suite when it is available locally.

`tests/test_training.py` covers:

- optimizer coverage and causal-encoder update;
- default 40-iteration joint-training budget;
- independent restart initialization and encoder/duration-bias reset;
- deep best-run snapshots including `duration_logits_bias`;
- K-Means emission initialization on separated observations;
- end-to-end multi-restart optimization;
- scalar external-context training with non-degenerate distribution hidden width;
- transition-only refinement parameter isolation;
- refinement opt-in defaults;
- transition-specific context capacity without changing Initial/Duration/Emission bounds;
- caller train/eval mode restoration when optimization fails during input validation.

`tests/test_duration_prior_training.py` covers geometric D+ prior normalization, injection after independent restart initialization, and recovery of the optimized tail hazard after an update crosses either probability boundary.

`tests/test_emission_variance_initialization.py` checks the variance-aware initializer against a Gaussian density and global-variance fallback, plus parameter preservation and causal prefix invariance of the script-local emission-weight diagnostic. `tests/test_kmeans_cluster_variance.py` checks actual returned Gaussian covariances rather than assuming an exponential raw-parameter transform.

`tests/test_context_hardening.py` owns external-context shape, finiteness, variable-length, and width contracts. `tests/test_context_optin.py` owns internal-encoder opt-in and model-construction contracts. `tests/test_default_distributions.py` verifies fail-closed categorical labels/temperature controls plus frozen context-free neural-control parameters.

`tests/test_trading_feature_redundancy.py` checks training-only ZCA covariance normalization, pointwise prefix invariance, the Gaussian likelihood Jacobian identity against an independent density, and singular-covariance regularization for the script-local feature diagnostic.

`tests/test_trading_boundary_evidence.py` verifies the script-local causal edge update against independent probability-space enumeration, marginal argmax revision without a latent transition, observation-supported low-prior boundaries, episode self-transitions, and one-to-one boundary event matching.

`tests/test_trading_causal_decisions.py` checks the script-local prefix MAP endpoint and committed-state rules against independent exhaustive probability-space paths, finite and D+ support, prefix invariance, batch equivalence and impossible-support failure.

`tests/test_trading_observation_controls.py` verifies training-only diagnostic generator moments, Gaussian marginal statistics and AR(1) autocorrelation, prefix reproducibility, held-out feature mutation isolation and strict JSON missing-state reporting.

`tests/test_model_conform_duration.py` independently checks geometric-tail masses, duration means and survival sums, terminal right censoring, generator episode boundaries, frozen oracle Gaussian parameters, duration movement and tail-information coverage. It also verifies observed optimizer-step reporting, exclusion of final restart scoring, and rejection of incomplete fixed budgets.

## Complete-code audit — 2026-10-12

The audit started from `regime-v2` commit `b17a69a497d7391f0fc3025ded9b04c18dc5ef6c`. It covered the complete tracked Python inventory (130 files before the new regression module; 131 afterward) with AST parsing, repository-wide Ruff/Black checks and the canonical CPU pytest suite. Manual review focused on distribution parameterization, causal/non-causal inference, duration support, gradients, context/padding, initialization/restarts, streaming state and caches, artifacts, tuning, validation and diagnostic data/evaluation paths. This is bounded verification, not an exhaustive line-by-line proof of every script or a new statistical acceptance run.

The following production defects were reproduced with targeted counterexamples and corrected. Their regression owner is `tests/test_code_audit_regressions.py` (11 tests; padding and empty-row pooling are separate cases).

| Owner | Reproduced defect | Resulting behavior |
| --- | --- | --- |
| Artifact | CPU snapshots shared model tensor storage | Built artifacts own cloned CPU tensors |
| Context pooling | MHA pooled padded queries and produced NaN for all-masked rows | Only valid queries contribute; empty rows return finite zero context |
| Runtime | A rejected first step fixed the context mode | Context mode locks only after a successful step |
| Causal Viterbi | With D=1 and a tail, boundary scores overwrote continuation scores | Both paths compete, with matching predecessor pointers |
| Non-causal duration | Tail hazard 1 evaluated a singular continuation derivative in an inactive branch | Exact impossible continuation support remains; tested likelihood gradients are finite |
| Categorical | Zero probability had a finite log probability | Zero mass maps to negative infinity; entropy handles zero terms |
| Streaming encoder | Replaced LSTM parameters could reuse version counters and stale fused weights | Cache keys track parameter identity and mutation version; dtype-moved buffers retain the existing no-rebuild contract |
| Optimization | convergence_stop=False still stopped on a plateau | A successful run consumes its configured max_iter budget when stopping is disabled |
| Optimization config | duration_tail could change without rebuilding structural distributions | Incompatible override is rejected before optimization |
| Tuning | trials limited grid output only after full Cartesian materialization | Candidate enumeration stops at the requested limit |

Two suspected initializer defects were rejected during the full-suite contract check: legacy spread initialization deliberately preserves Gaussian effective variance softplus(0) and Student-t effective scale softplus(1). Those parameters and their established tests remain unchanged. They must not be interpreted as unit variance/scale. The optional variance-aware initializer has its own explicit effective-variance contract.

Historical diagnostic scripts were formatted and their static import/dead-variable/name violations corrected. Emission identifiability and initialization-ablation reports now read actual Gaussian variance as softplus(raw).clamp_min(min_covar), rather than exp(raw). Historical JSON results were not rewritten or rerun; older v1 diagnostics remain historical evidence.

### Observed verification

- AST parsing: all 131 Python files passed.
- Ruff: repository-wide check passed.
- Black: canonical package/tests/scripts check passed, 131 files unchanged.
- `git diff --check`: passed.
- Focused post-contract-correction gate: 24 passed.
- Canonical CPU suite: **787 passed, 3 skipped**, 98.90 seconds pytest time; the skips require CUDA.

The final suite includes the existing independent exhaustive-reference, gradient, padding, runtime-equivalence and training contracts. No benchmark/training experiment batch was started as part of this audit, beyond the small fixtures required by tests. CUDA execution, performance benchmarking and full post-fix statistical recovery acceptance remain unverified.

### Effect on earlier experiment interpretation

Before this audit, convergence_stop=False did not guarantee the requested iteration count. Earlier 30/40-iteration descriptions therefore identify maximum budgets unless actual completed iterations are separately recorded. Their stored metrics remain observations of the historical code, but they are not evidence for an exact fixed-length comparison. Follow-up training comparisons must use the corrected stopping behavior and record actual completed iterations. No package default or trading-usefulness claim is changed by this audit.

## HSMM research/correctness milestone gate

The independent HSMM correctness milestone is owned by this testing contract. Its acceptance harness should live under `tests/reference/` and remain deliberately independent from optimized production recursion.

The execution sequence and stop/go criteria are defined in [hsmm-correctness-work-gates.md](hsmm-correctness-work-gates.md).

Required reference coverage:

- exhaustive small-model joint path enumeration for likelihood and filtering;
- exact MAP/Viterbi reference from the same enumerated path set;
- explicit state-age and episode-boundary marginals;
- degenerate cases `K=1`, `D=1`, `T=1`, self-transitions, impossible support, and near-deterministic probabilities;
- Float64 gradient checks for initial, duration, transition, emission, and end-to-end log-likelihood;
- property-based invariants for normalization, causal prefixes, right padding, streaming equivalence, support preservation, and age reset on episode boundaries.

Reference code must not call `NHSMM.forward`, `NHSMM.decode`, `filter_step`, `filter_model_sequence`, `duration_log_hazard`, `HSMMFilterRuntime`, or private production recursion helpers. Shared data containers and raw distribution parameter extraction are allowed only when they do not reproduce production inference logic.

Numerical acceptance must distinguish mathematical equality from floating-point execution:

- Float64 reference comparisons use the tightest tolerance justified by observed error;
- Float32 batch/filter/runtime comparisons use a measured bounded tolerance recorded with the test;
- impossible support remains exact where the contract requires `-inf`/zero mass.

Statistical recovery harnesses remain a separate acceptance layer and do not substitute for this mathematical gate.

## Static checks

```bash
ruff check nhsmm tests scripts
black --check nhsmm tests scripts
```

Final Gate-10 package acceptance on 2026-10-07 used the documented CPU bootstrap on Python 3.12 and passed:

- `ruff check nhsmm tests scripts`;
- `black --check nhsmm tests scripts` — 90 files unchanged;
- `git diff --check`;
- `python -m pytest -q` — `712 passed, 3 skipped`, with the three skips CUDA-only.

## Controlled duration-context acceptance

```bash
python scripts/validate_duration_context.py --output /tmp/nhsmm-duration-context.json
```

The harness explicitly opts into the internal ContextEncoder because duration context is derived from observations. Independent seed runs are process-parallel by default with up to four single-threaded workers; use `--workers 1` for serial diagnostics.

Re-run after material changes to training objective, restart semantics, duration parameterization, context encoder, optimizer coverage, causal filtering, or survival semantics.

## Controlled duration-tail acceptance

```bash
python scripts/validate_duration_tail.py --workers 4 --output /tmp/nhsmm-duration-tail.json
```

This gate compares the opt-in bounded `D+` tail against the historical finite-support model on matched controlled data. The true-tail scenario places substantial duration mass beyond `D=8` and evaluates held-out log likelihood plus episode-end survival calibration at horizons `1,3,5,8`. The finite-support control contains no duration above `D` and requires the learned tail hazard to return to the compatibility boundary `h_tail≈1` without material metric drift.

Re-run after changes to duration-tail parameterization, hazard semantics, optimization coverage, causal filtering/runtime, or survival forecasting.

## Controlled latent-state acceptance

```bash
python scripts/validate_state_recovery.py \
  --scenario all \
  --workers 1 \
  --output /tmp/nhsmm-state-recovery.json
```

The benchmark evaluates state identity only up to permutation and explicitly uses `emission_init_mode="kmeans"`; package default remains `spread`.

## Controlled learning/prediction acceptance

    python scripts/validate_learning_prediction.py --scenario all --seeds 401,402,403,404,405 --max-iter 40 --workers 4 --output /tmp/nhsmm-learning-prediction.json

This is the canonical empirical proof that optimize() changes learned state and that the fitted model improves held-out likelihood and predicts identifiable latent states on disjoint OOS sequences.

The harness now separates two questions:

1. **learning_pass** — parameter movement, held-out likelihood gain, permutation-invariant OOS state recovery, ARI, provenance, and variable-length prediction;
2. **usefulness_pass** — boundary timing, regime run lengths, transition recovery, occupancy recovery, posterior calibration, collapse behavior, and train/OOS stability.

Usefulness metrics include exact boundary F1, a causal ±1-timestep boundary F1 for weak-signal interpretation, run-length relative error, transition-matrix MAE, occupancy L1, posterior ECE/Brier, and health diagnostics.

Latest 5-seed evidence:

| scenario | OOS accuracy | ARI | boundary F1 | boundary F1 ±1 | run-length rel. error | transition MAE | occupancy L1 | posterior ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| strong | 0.9989 | 0.9965 | 0.9890 | 0.9944 | 0.0108 | 0.0029 | 0.0022 | 0.0032 |
| moderate | 0.9633 | 0.8936 | 0.7568 | 0.9205 | 0.0213 | 0.0096 | 0.0222 | 0.0309 |
| weak | 0.7856 | 0.4642 | 0.3222 | 0.6000 | 0.0652 | 0.0788 | 0.1733 | 0.1687 |
| null | 0.3589 | 0.0000 | 0.0000 | 0.0000 | 17.8000 | 0.3333 | 1.2822 | 0.6238 |

Strong and moderate pass both learning and usefulness cleanly. Weak passes a deliberately lower-confidence usefulness contract; exact boundary timing is explicitly weaker, while ±1-timestep timing, durations, transitions, occupancy, and calibration remain informative. Null is a negative control: learning-control checks pass, but usefulness is not applicable and no predictive state identity is claimed.

The hard state-identification acceptance explicitly uses emission_init_mode="kmeans". A separate default-spread probe is recorded in validation/README.md.

The harness configuration is now externalized through the public `ValidationConfig` contract. A complete JSON contract can be supplied with `--config`; `--dump-config` writes the fully resolved contract; repeated `--set` arguments tune fields such as `model.max_iter`, `data.steps`, `health.max_state_occupancy`, `boundary_tolerance`, or `scenario.moderate.max_median_ece` without code edits. See [configuration.md](configuration.md).


## Controlled transition-context acceptance

```bash
python scripts/validate_transition_context.py \
  --scenario all \
  --workers 1 \
  --output /tmp/nhsmm-transition-context.json
```

The transition harness uses known context-dependent transition matrices, permutation-invariant state alignment, strong/moderate effects, and a null control. It sets `max_duration=1` to isolate transition recovery. Maintained validation explicitly opts into `transition_context_max_delta=1.5`, `transition_refine_steps=20`, and `transition_refine_lr=0.03`; package defaults remain conservative and refinement remains opt-in.

Re-run after changes to transition context parameterization, context-network hidden dimensions, refinement likelihood, filtering boundary semantics, state recovery, or external-context handling.

Acceptance thresholds and latest observed results are recorded in [`validation.md`](validation.md).

## External HSMM reference cross-check

The external implementation cross-check is intentionally optional and is not part of the canonical pytest suite. It compares fixed-parameter stationary non-causal explicit-duration inference against `gaussian-hsmm==0.1.0`:

```bash
python scripts/validate_external_hsmm_reference.py \
  --cases 20 \
  --seed-base 1000 \
  --output /tmp/external-hsmm-reference.json
```

Run it in a dedicated reference environment with `gaussian-hsmm==0.1.0` available. The harness does not call the external model's `fit()` path and does not compare optimizers; it exercises only the external explicit-duration `score()` and `decode()` inference paths with parameters matched to NHSMM. A pass requires zero Viterbi mismatches and maximum absolute marginal log-likelihood difference <= `1e-12`.

## Runtime benchmark

```bash
python scripts/benchmark_runtime.py MODEL_ARTIFACT --steps 128 --warmup 16 --batch-size 1
```

Latency and Python-allocation measurements use separate passes; `tracemalloc` is not active during latency measurement. Absolute hosted-runner timing is evidence, not a pass/fail threshold.

## Integration smoke

`.github/workflows/smoke.yml` is a clean-runner integration diagnostic started manually with `workflow_dispatch`. Run the local/focused checks first and use the smoke workflow after the relevant local gate is green. Routine edit/test work remains local; GitHub Actions is not the primary edit/test loop.

## Downstream Nautilus usability check

Nautilus empirical checks are downstream validation, not package-core acceptance. The repository workspace profile declares:

```ini
[workspace]
data = nautilus
```

AWA mounts the dataset read-only at `/data/nautilus`; do not copy the dataset into the repository or workspace.

The maintained current state-count evidence is recorded in [`validation/README.md`](validation/README.md). Re-run that comparison after material model/training/decoder changes or changes to the Nautilus 18-D observation mapping.


## Step 1.1 — predictive neural acceptance contract, 2026-10-12

The first implementation step is split at its semantic boundary. This stage
freezes executable acceptance requirements before changing model/runtime
behavior or interpreting stored encoder artifacts under new timing semantics.
At this historical checkpoint production migration was pending; none of the strict expected failures below was
a claim that normalization has been fixed.

For causal internal observation-derived context, let c_t encode observations
through t and let c_-1 be a fixed zero start context:

| Quantity | Allowed internal information |
| --- | --- |
| Initial-state prior | c_-1, independent of the first observation |
| Emission for x_t | c_(t-1), excluding x_t |
| Boundary propagation from t-1 to t | c_(t-1), preserving existing boundary timing |
| Context cached after accepting x_t | c_t, for the following observation |

The implementation must separate emission context from the context retained for
duration/transition forecasts. Shifting every distribution's context would delay
the existing boundary semantics by an extra row and is not the intended repair.

Acceptance owner: `tests/test_predictive_neural_contract.py`.
It checks strict past-only emission context, observation-independent initial
context, zero-start incremental runtime emission, normalization of a one-row
density, normalization of a predictive density after a fixed prefix, and a
normalized held-fixed external-context control. Quadrature uses 28,001 points
over [-12,16] with absolute normalization tolerance 2e-5.

Two additional boundaries are explicit:
- Legacy v1 encoder artifacts must not be silently reinterpreted under changed
  emission/prior timing. A versioned predictive artifact contract and explicit
  rejection/retraining guidance are required. Encoderless legacy compatibility
  is to be preserved.
- Bidirectional/noncausal observation-derived encoder scores must not be exposed
  as a normalized generative likelihood merely by lagging an emission row.
  Generative likelihood/training must reject that unresolved interpretation;
  any retained retrospective conditional-score API needs explicit semantics.
  Encoderless noncausal HSMMs are a separate supported model class.

Observed native gate: **19 passed, 7 strict expected failures** in the combined
new-contract, code-audit and duration-diagnostic suite. The seven expected
failures reproduce current gaps; strict XPASS forces removal of a pending mark
when its production repair is integrated. Ruff and Black passed. No full-suite
or statistical-acceptance claim is made for this preparatory contract stage.

Planned at the Step 1.1 checkpoint: implement batch emission/prior timing, bounded runtime context
retention and artifact migration together; convert acceptance failures to passing
regressions, run affected and full canonical suites, update documentation and
push that completed production stage before advancing to self-reset/duration
identifiability work.


## Native Torch tail-boundary compatibility — 2026-10-12

Native PyTorch 2.14.1+cpu returns a zero clamp derivative at the exact feasible tail bounds EPS and 1. Existing projected-tail recovery tests therefore failed after moving from the earlier AWA runtime. Duration.tail_probability now uses explicit strict outside-domain branches: values and exact h=1 support remain unchanged, the derivative is one inside the feasible domain including its endpoints, and zero outside. Four new Float32/Float64 endpoint-gradient regressions and the maintained tail suites passed: **30 passed**. Ruff and Black passed. This is an isolated numerical prerequisite for the predictive-neural migration, not statistical recovery acceptance or completion of Step 1.2.


## Step 1.2 — predictive neural production migration, 2026-10-12

Batch emissions use the previous causal encoder row; the initial prior uses a
zero start context. Streaming caches the accepted current context for the next
emission and preserves boundary timing. Public likelihood/training rejects
noncausal internal observation-derived context. Artifact v2 declares timing;
v1 encoder artifacts are rejected, while v1 encoderless loading remains covered.
Retrospective forward scores remain available.

The predictive acceptance module now has **11 passing tests**, with no pending
xfails. Additional regressions cover training rejection before mutation,
encoderless legacy loading, and emission-context observation gradients.
Affected artifact/context/runtime/training suites: **80 passed**.
Broader native CPU verification: **802 passed, 3 CUDA skips, 1 deselected**.
Ruff passed; Black reported **110 files unchanged** in the materialized
package, test and script closure. Native runtime: Python 3.12.14,
PyTorch 2.14.1+cpu, NumPy 2.3.5.

The excluded test is
tests/test_trading_benchmark_metrics.py::test_legacy_generation_preserves_frozen_numerical_reference.
Its exact historical data/reference/trading-regimes-v1/trading_regimes.npz
could not be retrieved through the GitHub connector (empty Contents response;
binary Blob decoding unsupported). Required Git blob:
1f13d48fde153f7a6aa81cd2883d50f8b2960b41; manifest SHA256:
370e6e9e87e52d779ea208111eeb2e03b0bf21f8b7e88859fcc8c19ac2739fb4.
Reconstructions had different hashes and were not substituted. The test remains
unchanged. This is not a fully green canonical-suite claim.

The broader gate used single-thread OMP/MKL/OPENBLAS and
python -m pytest -q -p no:cacheprovider -k
'not legacy_generation_preserves_frozen_numerical_reference'.

The corrected structural diagnostic integrates the former neural counterexample
to **1.0** on 28,001 and 56,001 points; its held-fixed control is also 1.0.
Self-reset nonidentifiability remains unchanged. See
[results](validation/results/predictive-neural-step12-20261012.json).
The tail-boundary prerequisite is commit
56c485aaa69fc766596798bd42b737894fa79a55.
No new statistical fit acceptance or trading-performance claim is made.
Next: controlled encoderless duration/self-reset baseline as its own documented,
pushed stage.

## Step 2 — controlled duration/self-reset diagnostic, 2026-10-12

`tests/test_duration_reset_diagnostic.py` verifies that a zero-reset generator's
episode boundaries coincide with observed state switches and that the default
generator remains bitwise unchanged when its existing matrix is explicit. It
checks short-censor bucket allocation, geometric tail exposure without a false
terminal end, monotone labeled EM likelihood, and independent NumPy/model
likelihood agreement for finite and D+ laws across all four transition arms.
Exact zero diagonal support, age tying, fixed duration gate, known transitions,
frozen nuisance parameters and actual update counts are checked.

Initial native gate: **12 passed** across the new module and
`tests/test_model_conform_duration.py`. AST parsing and Ruff passed; Black
reported **112 files unchanged** in the materialized closure. The final focused gate including the audit regressions reported **23 passed**. Only diagnostic
scripts, tests and evidence/docs change; production model/defaults do not.
The broader 802-pass gate from Step 1.2 is historical evidence, not a new full
suite run. Its unavailable frozen NPZ test remains unresolved.

Bounded execution command, with OMP/MKL/OPENBLAS threads set to 1:

```bash
python -m scripts.diagnose_duration_resets --seeds 901,902,903 --iterations 80 --base-commit 6eeac114228cda26ff969b0ec207a4bd54902e91 --output docs/validation/results/duration-reset-step2-native-20261012.json
```

The independent NumPy recursion verifies fixed fitted parameters; the labeled
censored EM has privileged episode boundaries. Neither is an independent
unsupervised training baseline. Full statistical acceptance and adequate tail
exposure remain separate requirements. Results and interpretation belong to
[validation.md](validation.md#controlled-duration-and-self-reset-baseline--2026-10-12).

## Step 3 — fixed tail exposure and run support, 2026-10-12

`tests/test_tail_coverage_diagnostic.py` checks tail ends versus censored survival
exposure, fail-closed coverage requirements, absence of a hazard estimate when
there is no exposure, preservation of the true PMF during hazard-only marginal
training, and positive D+ support above the finite duration bound with exact zero
self-reset support. The three fixed 128-sequence datasets are independently
generated in tests and must all meet the 30-end/60-trial per-state requirements.

Initial focused native gate: **18 passed** across tail-coverage, reset-capacity
and model-conform diagnostic tests. AST/Ruff pass and Black reports **114 files
unchanged** in the materialized package/test/script closure. No production
model/default changes; the optional known-PMF fit control is script-only.
The historical missing frozen NPZ verification remains unresolved.

Each seed is executed sequentially with a 300-second bound, one worker and
OMP/MKL/OPENBLAS threads 1:

```bash
python -m scripts.diagnose_tail_coverage --seed 901 --iterations 80 --base-commit fbbff77e177453bed120a4fa6947c5787db23f45 --output docs/validation/results/tail-coverage-step3-seed901-native-20261012.json
```

Repeat for seeds 902 and 903 with distinct result filenames. Source SHA256s,
array hashes, completed updates and independent likelihood errors are recorded.
Methods, results and limits are owned by
[validation.md](validation.md#fixed-tail-exposure-and-run-duration-support--2026-10-12).


Final Step 3 gate including code-audit regressions: **29 passed**. Every seed
finished its four arms at 80 updates within the per-seed bound. Source hashes,
preserved six-sequence prefixes against immutable Step 2 evidence, and all
update counts were checked. The largest independent NumPy/model likelihood
difference was 4.219e-15 per row. This verifies the recorded bounded experiment,
not full canonical-suite or statistical acceptance.

## Step 4 — independent marginal duration EM, 2026-10-12

`tests/test_duration_em_diagnostic.py` compares NumPy forward/backward sufficient
statistics to complete generative path enumeration at bucket widths 1, 2 and 3.
It covers parallel geometric-survival/self-reset edges, initial bucket draws,
right-censored terminal episodes and the single-row no-trial case. It also
checks normalized PMFs, feasible hazards, monotone EM likelihood and exact
agreement with the public model likelihood after importing fitted parameters.

Initial native gate: **5 new tests passed**. AST and repository-closure Ruff
passed; Black reported **116 files unchanged**. This adds a research diagnostic
and tests only; production training and defaults are unchanged.

Sequential per-seed runs have 300-second bounds, one worker and OMP/MKL/OPENBLAS
threads 1:

```bash
python -m scripts.diagnose_duration_em --seed 901 --base-commit a93b6c0ccf7d8df2ae456ef05fdaa9ff4963c918 --output docs/validation/results/duration-em-step4-seed901-native-20261012.json
```

Repeat for seeds 902 and 903 with distinct output filenames. Each run verifies
frozen Step 3 dataset hashes, equal initialization likelihood, and reproduction
of the prior Adam-80 held-out score. It records an 81-entry EM likelihood history,
80/80/400 completed updates and fit timings. Known emissions, initial law and
transition matrix remain privileged controls; no labels enter either fit.
Methods and measured results are owned by
[validation.md](validation.md#independent-duration-em-versus-gradient-fitting--2026-10-12).


Final Step 4 focused gate, including preceding duration diagnostics and audit
regressions: **34 passed**. AST/Ruff pass; Black reports 116 unchanged files in
the materialized closure. All three seed batches completed 80/80/400 updates
within their per-seed bounds. Initial likelihood matching, prior Adam-80
reproduction, 81-entry monotone EM histories and every source/data/report hash
were verified. Maximum independent held-out likelihood discrepancy:
4.219e-15 per row. No production change or new full canonical-suite claim;
the historical frozen-NPZ verification remains open.


## Step 5 — controlled emission dependence, 2026-10-12

tests/test_emission_dependence_diagnostic.py checks stationary residual
marginals/correlation, all-sequence prefix reproducibility, independent Gaussian
density, IID collapse, exhaustive state-path likelihood/posterior agreement,
causal prefix/future mutation invariance, train-only nuisance fitting and
nonuniform finite/D+ agreement with public filtering. The public-input comparison
uses identical Float32-rounded observations because NHSMM canonicalizes inputs;
recurrence assertions retain 1e-12 tolerance.

Final native focused gate: **24 passed** (seven new checks, six maintained
observation controls, eleven audit regressions). AST/Ruff pass; Black reports
118 unchanged files in the materialized closure. Three seeds and 48 comparisons
completed within a 180-second bound, one worker and OMP/MKL/OPENBLAS threads 1.
Source provenance was verified. Reproduction:

    python -m scripts.diagnose_emission_dependence --base-commit 22bb15e3abfe994d6acf354bd117f833e98007f1 --output docs/validation/results/emission-dependence-step5-native-20261012.json

The initial public-input fixtures compared different Float64 versus canonical
Float32 values and failed by about 1e-8. They were corrected without loosening
tolerance. The diagnostic was rerun after all focused checks passed.
No production/default change or new full canonical-suite claim; the historical
frozen-NPZ check remains open. Methods, results and limits:
[validation.md](validation.md#controlled-emission-autocorrelation-and-covariance--2026-10-12).

## Step 6 — maintained Trading feature residuals, 2026-10-12

tests/test_trading_residual_diagnostic.py checks episode-safe lag pairing even
when regimes recur, train-only fitting under held-out mutations, omission of
the first split/episode row from AR/VAR innovations, independent IID and AR
controls, and explicit null statistics for constant features in strict JSON.
Initial focused gate: five tests passed. AST/Ruff pass; Black reports 120
unchanged materialized Python files. Production code/defaults are unchanged.

The bounded diagnostic uses current native v2 generation at seeds
424242/424243/424244, 60,000 rows each, start price 50,000 and episode-end
transition context. It uses contiguous 60/20/20 chronological splits, not
the benchmark's sampled windows. Training state labels define oracle residual
means and one-step AR/VAR controls; test labels are diagnostic conditioning only.
No likelihood, state-recovery or live predictor acceptance claim is made.
This generated v2 data does not replace the unavailable historical v1 NPZ.

Reproduction with one worker and OMP/MKL/OPENBLAS threads 1:

    python -m scripts.diagnose_trading_residuals --base-commit 800a6e6098f443bc4af946584ecb52bd51764fe3 --output docs/validation/results/trading-residual-step6-native-20261012.json

The execution bound is 180 seconds. Source hashes and feature/label/episode
hashes are recorded. Methods and measured evidence belong to
[validation.md](validation.md#maintained-trading-feature-residuals--2026-10-12).


Final Step 6 native gate: **32 passed** across the new residual tests,
maintained observation/redundancy controls, edge-emission checks and audit
regressions. AST/Ruff pass; Black reports 120 unchanged files in the materialized
closure. All three native v2 generations and train/validation/test summaries
completed within the run bound. Source provenance was verified. This adds
research diagnostics/evidence only; the full canonical suite was not rerun and
the unavailable legacy NPZ check remains open.

## Step 7 — common-target feature/emission pilot, 2026-10-12

tests/test_feature_contract_pilot.py verifies training-permutation probability
mapping, causal next-return design excluding the final future target from inputs,
frozen train-only calibration, identical target pair counts, raw-unit Gaussian
score agreement with an independent Normal density, finite explicit Float64
log-score flooring with zero-probability counts, matched Gaussian/Student-t base
cluster variances, and real production optimize smoke paths for both families.
Initial focused gate: eight tests passed. AST/Ruff pass; Black reports 122
unchanged files in the materialized closure.

Sequential per-seed runs have 300-second bounds, one worker and OMP/MKL/OPENBLAS
threads 1. Seeds 901/902/903 share the frozen v2 dataset seed 424242 and fixed
training/test windows. Every model must record 40 completed updates excluding
restart rescoring. The prior Step 6 report and generated feature hashes are
verified; source hashes are recorded.

    python -m scripts.diagnose_trading_feature_contract --seed 901 --base-commit 836ada51dad9cca04d7420d228afbe93791d01db --output docs/validation/results/feature-contract-step7-seed901-native-20261012.json

Repeat for 902 and 903 with distinct result filenames. Initial diagnostic
integration exposed Float32 metric-floor underflow and the public restriction
of kmeans_cluster_variance to Gaussian. Both were corrected in script-only
scoring/initialization controls and protected by full optimizer smoke tests
before rerunning the accepted batches. Production initialization/defaults and
the historical unavailable NPZ verification are unchanged. Methods/results:
[validation.md](validation.md#matched-trading-feature-and-emission-contract-pilot--2026-10-12).


Final Step 7 focused gate: **35 passed** across feature-contract, redundancy,
observation, model-conform duration and audit regressions. AST/Ruff pass; Black
reports 122 unchanged files. All 18 accepted models recorded exactly 40 updates,
with identical 704 state rows and 700 next-return targets. Source/frozen dataset
provenance was verified. No production change or new full canonical-suite
claim; the historical frozen-NPZ verification remains unresolved.

## Step 8 — direct observable baselines and data replication, 2026-10-12

tests/test_trading_information_baselines.py checks fixed causal lag columns,
prefix invariance under future mutations, exclusion of each final future target
from the design, rejection of negative lags, frozen mean/variance coefficients
under test-target mutation, raw-unit density agreement with an independent
Normal distribution, constant-posterior equivalence to the direct forecast,
and direct raw-log scoring under exponentiation underflow and state permutation.

Before model execution, the focused gate passed **40 tests** across the new
baseline tests, Step 7 family/optimizer tests, feature redundancy, observation
controls, model-conform duration and code-audit regressions. AST/Ruff/Black
passed; Black reported 124 unchanged materialized Python files. The recycled
runtime required restoration of declared dependencies and the editable package;
execution uses Python 3.12.14, Torch 2.14.1+cpu, NumPy 2.3.5, pytest 9.1.1,
Black 26.10.1 and Ruff 0.17.0. No packaging/lock/default changes were made.

Each generator-seed run has a 300-second bound, one worker and OMP/MKL/OPENBLAS
threads 1. Runs are sequential. Each Gaussian/Student-t fit must record exactly
40 production updates. Source SHA256 hashes and the corresponding committed
Step 6 feature hash are checked or recorded; remote input identities were
verified against the immutable Step 7 head.

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 300 python -m scripts.diagnose_trading_information_baselines --generator-seed 424242 --base-commit c2528258a66c5f65399030b711ff5ffab1ba3c96 --output docs/validation/results/information-baselines-step8-data424242-native-20261012.json

Repeat for generator seeds 424243/424244 with matching output filenames.
Methods, numerical limits and measured results belong to
[validation.md](validation.md#direct-observable-baselines-and-data-replication--2026-10-12).
The full canonical suite is not rerun; the unavailable historical v1 NPZ check
remains unresolved. These are separate native v2 generations, not replacements
for that fixture.

Final Step 8 evidence: all three bounded sequential runs completed; six fits
record exactly 40 updates, each with 704 state rows and identical 700 forecast
pairs across heads. All source and frozen feature hashes were verified. The
initial 40-test gate remains the applicable code verification; no source changed
after execution. No new full canonical-suite or historical-NPZ acceptance claim.

## Step 9 — calibration and optimization budget controls, 2026-10-12

tests/test_trading_calibration_budget.py verifies disjoint and complete
window folds, invariance of held-out mean predictions to their own block's
target mutations, deterministic nested ridge selection and refit, frozen
parameters under external target mutations, raw-unit score agreement with
an independent Normal distribution, and exact intermediate-weight equivalence
to separately stopped production fits for both Gaussian and Student-t.
The initial focused gate passed **45 tests** across this file, Step 8/7,
redundancy, observation, duration and audit-regression controls. AST/Ruff/Black
passed before execution; Black reported 126 unchanged materialized Python files.

Three native generator-seed batches run sequentially, one worker, with
OMP/MKL/OPENBLAS threads 1 and a 600-second subprocess bound per dataset.
Every family must record 120 production updates. The script captures the
40-update weights on the same trajectory, verifies them against the immutable
Step 8 state/fixed-head metrics, and records all pre-update training likelihoods.
Current dependency/report Git identities were verified through the connector;
source SHA256 and prior report/dataset hashes accompany each result.

    python -m scripts.diagnose_trading_calibration_budget --generator-seed 424242 --base-commit d98678d2b9b3446181ba89e9e91a096a9c072435 --output docs/validation/results/calibration-budget-step9-data424242-native-20261012.json

Repeat for 424243/424244 with distinct filenames and the same single-thread
environment. No external test or validation segment is used for ridge selection.
Head-fold representation sharing, non-walk-forward folds and remaining
in-sample variance smearing are explicitly disclosed in
[validation.md](validation.md#trading-calibration-and-optimization-budget-controls--2026-10-12).
No full canonical-suite or historical-NPZ acceptance claim.

Final Step 9 evidence: all three sequential bounded batches completed;
six trajectories have exactly 120 production updates. All twelve 40/120
snapshots retain 704 state rows and 700 common future-return pairs. Every
40-update checkpoint matched immutable Step 8 scores. Source, prior-report
and dataset hashes were verified. No source changed after the initial
45-test/AST/Ruff/Black gate. Production defaults and canonical-suite/NPZ
verification limits are unchanged.

## Step 10 — age-free HMM decision control, 2026-10-12

tests/test_trading_hmm_control.py verifies the independent NumPy HMM recursion
against complete prefix state-path enumeration, exact reconstruction of HSMM
emission/initial restart parameters, their exact transfer to the D1 HMM restart,
real three-update Gaussian/Student-t optimize paths, D1 duration normalization,
and production filtering/public likelihood agreement with independent recursion.
The focused gate passed **48 tests** across this file and the Step 9/8/7,
redundancy, observation, duration and audit controls. AST/Ruff/Black passed
before execution; Black reported 128 unchanged materialized Python files.

Three dataset batches execute sequentially, one worker, with OMP/MKL/OPENBLAS
threads 1 and a 300-second subprocess bound per dataset. Six HMM fits must
record 120 updates; twelve 40/120 snapshots have 704 state rows and 700 common
forecast targets. Existing Step 9 HSMM reports supply frozen comparisons.
Current source/report Git identities were verified through the connector.

    python -m scripts.diagnose_trading_hmm_control --generator-seed 424242 --base-commit 64e48f648bfd3a87b500330d3b99806258885d89 --output docs/validation/results/hmm-control-step10-data424242-native-20261012.json

Repeat for 424243/424244 with matching filenames and the single-thread
environment. All three bounded runs completed. Source, prior-report and
dataset hashes, completed update counts and shared target counts were verified.
All 24 train/test snapshot checks against independent NumPy recursion passed:
maximum log-posterior discrepancy 5.84e-5, maximum likelihood discrepancy per
row 1.96e-6, within fixed Float32 tolerances 1e-4/2e-5. No source changed after
the initial verification gate. No full canonical-suite or frozen-NPZ claim.
Methods and decision:
[validation.md](validation.md#age-free-hmm-decision-control--2026-10-12).

## Step 11 — Trading ground-truth and HSMM contract audit, 2026-10-12

New tests cover closed-form gamma2/Weibull CDF checks, clipped PMF support and
normalization, exact finite-D core hazard agreement, geometric-projection
normalization/mean/overflow, ledger corruption detection, latent/observed
remaining-life distinction, additive v2 censor labels, unchanged v1 schema
and exclusion of partial first episodes from complete-window counts.

65 focused tests passed across the new audit, dataset, boundary, causal
decisions, HMM, calibration, information, feature-contract, duration and audit
regressions. Six benchmark-metric tests also passed; the unavailable historical
legacy NPZ check was deselected. AST/Ruff/Black passed; Black reported 130
unchanged Python files. A 3000-row pre-change legacy control confirms identical
array and metadata hashes after the v2-only addition. It does not resolve the
historical frozen NPZ limitation.

The audit uses one worker, OMP/MKL/OPENBLAS threads 1 and a 180-second bound,
three native 60000-row datasets and 20000 independent duration draws/state.
The initial diagnostic was superseded by a complete rerun after the v2-only
label extension; only the final source-hash-matched report is committed.

    python -m scripts.diagnose_trading_ground_truth --base-commit d1686859e2fc4efb0eb58f2441d23bba56238760 --output docs/validation/results/ground-truth-step11-native-20261012.json

Methods, measured compatibility gaps and the revised isolated-HSMM gate:
[validation.md](validation.md#trading-ground-truth-and-hsmm-contract-audit--2026-10-12).
No new full canonical-suite or frozen-NPZ acceptance claim.

## Step 12 — isolated finite-support HSMM verification, 2026-10-12

tests/test_isolated_trading_hsmm.py covers a hand-solvable censored duration
MLE, survival rather than completed-mass scoring, independent segment recursion
against enumerated state-age paths, exact duration support and zero renewal
diagonal, public NHSMM likelihood/filter agreement, real duration-gradient
updates with frozen nuisance parameters, and overlapping Gaussian observations
with nontrivial causal posteriors. Final focused gate: 70 passed across this
file plus ground-truth, dataset, boundary, causal-decision, HMM, calibration,
information, feature-contract, duration and audit controls. AST/Ruff/Black pass;
Black reports 132 unchanged materialized Python files.

One worker, OMP/MKL/OPENBLAS threads 1 and a 300-second execution bound:

    python -m scripts.verify_isolated_trading_hsmm --base-commit 1028ec240c53ec797f7216f5d7e32e3cab0ef841 --output docs/validation/results/isolated-hsmm-step12-native-20261012.json

The final three-seed run follows the expanded ambiguity-control code gate.
All three seeds complete 1200 duration-only updates, and all nine public
likelihood/filter comparisons pass independent references. Source hashes
and all predeclared result gates are verified. The labelled learning stage
does not call production optimize or establish end-to-end unsupervised recovery.
Research support masks are not verified as a runtime/artifact contract.
Methods and scope:
[validation.md](validation.md#isolated-finite-support-hsmm-verification--2026-10-12).
No new full canonical-suite or historical-NPZ acceptance claim.

## Step 13 — observation-only marginal duration learning, 2026-10-12

tests/test_unlabelled_trading_hsmm.py checks public marginal likelihood against
an independent segment reference, production duration autograd against central
differences under ambiguous observations, actual observation-only parameter
updates and frozen nuisance equality, label/observation storage isolation, and
evaluation-only terminal right censoring. Together with Step 12 tests:8 passed.
AST/Ruff/Black pass over35 verified materialized/new Python files.

One worker, OMP/MKL/OPENBLAS threads1 and a900-second execution bound:

    python -m scripts.verify_unlabelled_trading_hsmm --base-commit 43e6789c378e2d559839112d25a67340d9800f0d --output docs/validation/results/unlabelled-hsmm-step13-native-20261012.json

The design fixes three seeds, observations-only Adam80/batch2 and known nuisance
parameters before held-out scoring. Learning progress and fine PMF TV recovery
are separate report gates. This focused execution does not rerun the full
canonical suite, historical NPZ gate, Trading features or joint optimize.

Final execution:240 updates completed,128 unique training sequences visited
per seed, nuisance parameters unchanged. Independent numerical gates pass:
gradient discrepancy<=1.91e-9, likelihood discrepancy<=1.05e-11 and state-probability
discrepancy<=6.33e-14. Report source hashes and update counts verified. Learning
progress passes seeds901/903 and fails seed902 (negative held-out improvement);
the stricter each-PMF-TV<=0.12 criterion fails all three seeds. These failures
remain in the report and are not converted into a green statistical gate.

The native runtime was restored from39 connector-verified immutable blobs;
declared dependencies and an editable package were installed successfully.
Execution versions: Python3.12.14, Torch2.14.1+cpu, NumPy2.5.4. Materialization
is sparse; unrelated retained scratch files were not part of this verification.
The preliminary600-row run was interrupted after its last reported40 updates
for seed901 due backward-pass cost; only the subsequent300-row execution is
accepted as completed run evidence. Methods, limits and measured results:
[validation.md](validation.md#observation-only-marginal-duration-learning--2026-10-12).

## Step 14a — independent unlabelled EM mechanics, 2026-10-12

tests/test_unlabelled_duration_em.py owns expected duration-count enumeration,
terminal survival/censor allocation, one-row zero duration information,
all-logit production gradient identity, independent segment likelihood,
finite-difference scores and EM objective monotonicity. Together with Step12/13:
12 tests pass. AST/Ruff/Black pass over37 verified materialized/new Python files.
Current sparse scope contains42 connector-verified immutable inputs plus the
new script/test. The restored native environment installs the declared CPU
dependencies and editable package successfully; Python3.12.14, Torch2.14.1+cpu,
NumPy2.5.4. The long paired-size comparison follows this completed code gate.

One worker, OMP/MKL/OPENBLAS threads1,1800-second bound:

    python -m scripts.compare_unlabelled_duration_em --base-commit ba81eb1997c3ccdf409ba1fc95f16beeaaae3def --output docs/validation/results/unlabelled-em-step14-native-20261012.json

No new full canonical-suite or historical-NPZ acceptance claim. Methods:
[validation.md](validation.md#independent-unlabelled-em-and-episode-exposure--2026-10-12).

Step14b completes all six arms:30 EM updates/arm,180 updates total. All six
marginal likelihood histories are monotonic within the declared tolerance.
Step13 train/held-out observation hashes, full prior source/report hashes,
Adam training likelihood and uniform held-out likelihood are replay-verified;
paired first128 observations are exactly identical. The final report contains
all three seeds, both learned PMFs, objective histories and nonzero residual
scores. Numerical gates pass: all3120 duration-score coordinates<=2.27e-14,
public/segment learned-law LL error<=1.12e-11, state-probability error<=1.10e-13.
Code is unchanged since the12-test/AST/Ruff/Black gate at Step14a.

Statistical result: small EM beats Adam training likelihood but worsens held-out
likelihood in every seed.8x sample exposure improves every state/seed PMF TV and
held-out likelihood is positive versus uniform in all seeds. The unchanged
each-PMF-TV<=0.12 gate still fails all three seeds; no statistical threshold
is relaxed and no global-MLE convergence claim is made. Source hashes and all
run/numerical gates are verified separately from these retained failures.

## Step15a — compact duration family and nested selection,2026-10-13

tests/test_compact_duration_em.py owns exact block partitions and clipped endpoint
atoms, invalid support/width rejection, hand-derived constrained Q M-step and its
zero grouped score, width1 unrestricted equivalence, support/normalization/constant
density, monotonic observation-only constrained EM, grouped finite-difference
scores and disjoint inner train/validation likelihood selection.18 focused tests
pass with Step12–14 controls. AST/Ruff/Black pass over39 verified materialized/new
Python files.45 connector-verified current inputs include both prior raw reports.
The unchanged native environment uses Python3.12.14,Torch2.14.1+cpu,NumPy2.5.4.

One worker,OMP/MKL/OPENBLAS threads1,1800-second bound:

    python -m scripts.compare_compact_duration_em --base-commit d3e6e9ad983ac4f1558f21333c68b49c304819fd --output docs/validation/results/compact-duration-step15-native-20261013.json

Widths4/8/16 and inner20/outer30 updates are fixed before outer evaluation.
Selection is confined to the small training dataset's96/32 split; endpoints
remain separately learned atoms and known support remains an oracle condition.
No new full canonical-suite or historical-NPZ acceptance claim. Methods:
[validation.md](validation.md#compact-duration-family-with-training-only-selection--2026-10-13).

Step15b completes9 inner fits and6 outer refits,360 EM updates total. All15
histories are monotonic within tolerance. Replay/source/input-report hashes,
disjoint96/32 indices, both inner subset hashes, exact argmax selection,
paired128/1024 observations, support and constant block density are verified.
Code is unchanged since the18-test/AST/Ruff/Black gate at Step15a. Learned-law
numerical checks pass:3120 raw duration-logit gradients<=3.01e-14, six public/reference
LL errors<=1.14e-11 and prefix state-probability errors<=8.66e-14.

All three selections choose width16(53 free parameters versus757). The1024-arm
passes positive held-out gain and each-PMF-TV<=0.12 in3/3 seeds; observed
TV0.05292–0.11169. The128-arm remains0/3 full passes and seed903 remains
held-out-negative. All six compact arms improve held-out likelihood over the
unrestricted Step14 EM while reducing training likelihood. These are separate
statistical findings, not waived thresholds or production-default changes.
Known per-state support remains an oracle condition; unknown-support/public
optimizer, joint Trading and runtime/artifact acceptance remain open. No new
full canonical-suite or historical-NPZ claim.

## Oracle-support switch retained, default off — 2026-10-13

The compact CLI defaults to --support-mode full (all ages1..260/state).
Explicit --support-mode oracle preserves Step15's per-state-bound control.
The historical Step15 command above ran at immutable
d3e6e9ad983ac4f1558f21333c68b49c304819fd with that version's oracle default.
With current code an oracle rerun requires the explicit flag; for example:

    python -m scripts.compare_compact_duration_em --support-mode oracle --base-commit <verified-current-commit> --output docs/validation/results/compact-duration-oracle-control-native-20261013.json

Original reports/source hashes remain
historical evidence tied to their recorded commits, not generic-support passes.

Four additional regressions verify the CLI/default and explicit oracle mask,
full-mode support construction without ground-truth access, public gradient/
likelihood agreement with nonzero mass outside true support, report-mode overwrite
protection, and finite monotonic full-support fitting. With prior focused tests:
22 passed. AST/Ruff/Black pass over39 materialized Python files.45 immutable
inputs were connector-verified; native dependencies/editable package restored.

Report schema v2 records support mode, oracle-support flag, per-state bounds,
matching uniform baseline and the previous unrestricted reference's oracle mode.
Likelihood/filter and gradient diagnostics use the same explicit active mask
as training. Legacy diagnostic callers retain their original oracle default.
The existing v1 compact reports are interpreted as oracle evidence; a full-mode
run must use a separate output path. CLI help and support modes are smoke-verified.
This is switch/short-run correctness, not a new full generic recovery acceptance.

## Step17a — bounded resumable full-support execution,2026-10-13

The full-support resource smoke is slower than the earlier oracle control.
A preliminary run at8c7865c743c8465e74d0fa5ab6c6d763679af3ba was interrupted
after its last reported small-seed901 update30, before a completed seed/result
was saved. Its partial work is not included in final statistical evidence.
A CPU denormal-mode comparison had identical likelihood/counts but little speed
gain; it was not adopted and numerical computation remains unchanged.

--stop-after-seed exits after one new complete seed is written. --resume loads
only reports with matching schema, source/input hashes, software/runtime metadata,
base commit and full support contract. Invalid/duplicate seeds and incomplete
inner/outer histories are rejected. Completed seeds are skipped, including
the all-completed no-op. The entire batch still consists of9 inner20-update
fits and6 outer30-update refits over901/902/903. No threshold/budget changes.

The prior combined23-test gate and final12-test compact module cover24 distinct
focused tests; resume metadata/completeness rejection and the segmented CLI
save/skip/no-op behavior are covered. AST/Ruff/Black pass over39 files.48 current
immutable inputs are connector-verified. Native environment is unchanged.
Each bounded segment uses one worker,threads1, and a2400-second bound; after
the first segment, append --resume while retaining --stop-after-seed and the
same --base-commit and output path. Numerical results are checked and pushed
at complete-seed checkpoints, with partial versus full acceptance explicit.

## Step17b — full-support seed901 verified checkpoint,2026-10-13

The saved v2 report at
docs/validation/results/compact-duration-full-support-native-20261013.json
contained exactly one complete seed901 at this checkpoint, at source commit
ca73c96cddf373677be0711882981e618e89e9a6. Three inner histories contain20 updates
each and both outer histories30 each:120 actual M-steps over five fits.
Resume completeness, current nine source hashes and both input-report hashes,
128/1024 observation replay and paired-prefix hashes, held-out replay,96/32
disjoint selection and inner hashes, selected validation argmax, monotonicity,
full support1..260, normalized/nonnegative block-constant PMFs and independently
recomputed TV/held-out/full-gate values all pass. No code changed since Step17a;
its24 focused cases/static verification remain applicable.

All1040 checked raw duration coordinates agree with independent posterior-count
gradients within2.05e-14. Public/reference LL errors are at most1.10e-11 and
state-probability errors at most7.29e-14. Both statistical recovery gates fail;
numerical verification does not imply recovery. Seeds902/903 and full batch
acceptance remain pending. Resume retains the original ca73c96 source identity,
even when subsequent commits update only report/documentation.

## Step17c — second full-support seed checkpoint,2026-10-13

At this checkpoint the report contained seeds901/902,240 actual updates and
ten monotonic fits. The Step17b source/input hash, replay/paired-prefix, inner
split/selection/hash, compact PMF and independently recomputed metric/gate checks
pass for both seeds. Seed902 checks all1040 raw duration coordinates: maximum
gradient error1.94e-14, public/reference LL error1.23e-11 and state-probability
error8.45e-14. Source code and numerical mode remain unchanged at ca73c96;
the previous24 focused cases/static checks remain applicable. Both902 arms
fail the statistical recovery gate despite positive held-out gain. Seed903
remains pending. This checkpoint accepts two complete numerical runs only.

## Step17d — complete full-support batch and representation bound,2026-10-13

The final v2 report contains seeds901/902/903 with360 actual M-steps over15
complete monotonic fits. All nine source hashes and both input-report hashes
match the frozen ca73c96 source identity. Each seed passes complete-history
checks,128/1024 paired data replay, held-out replay,96/32 disjoint inner
selection and hashes, selected validation argmax, full-support bounds,
nonnegative/normalized block-constant PMFs, and independently recomputed TV,
held-out gains and unchanged acceptance gates. The all-complete CLI resume
skips all three seeds without new fitting. No source/algorithm changes were
made during this batch; Step17a's24 focused cases/static checks remain valid.

All3120 raw duration coordinates agree with independent posterior-count scores
within2.05e-14. Across six public/reference comparisons, maximum LL error is
1.23e-11 and maximum state-probability error8.45e-14. All numerical checks pass;
all six statistical recovery gates fail. These are separate conclusions.

The separate family-bias JSON retains six support/width combinations. Current
source hashes, normalized forward-KL projections, their block-mass equality,
TV values, block-median lower bounds and chaotic block decomposition are
verified. Independent scalar candidate checks confirm the median minimizer.
Full-support width16 chaotic TV is bounded below by0.136925>0.12; the feasible
width8 projection passes every state-TV threshold, without asserting learned
recovery or changing the completed likelihood-selection experiment.
Production/canonical suite, historical NPZ and Trading acceptance are not
extended by these research checks.

## Step18a — exact adaptive block optimizer,2026-10-13

scripts/compare_adaptive_duration_em.py adds full-support training-derived
contiguous duration blocks. The DP is compared against every partition of a
six-age row at penalties0/0.7/8; additional cases check count-derived boundaries,
positive mass across all ages, zero-exposure behavior and invalid inputs.
An actual short observation fit checks monotonic penalized evidence plus public
likelihood/filter and all-logit gradient agreement. Selection isolation uses
disjoint inner observations and a deterministic stronger-penalty tie rule.

Nine new cases pass. The initial combined run passed32 of33 cases and failed a
new strict-improvement assertion: with two short sequences, positive penalty
correctly retained the initial single block. The real-fit improvement case
was corrected to penalty0, and all nine new cases then passed. The24 unchanged
existing cases passed in the combined run:33 distinct verified cases in total.
Both added files pass AST, exact-M-step smoke, Ruff and Black checks.
This is a script-only research scope; the canonical production suite is unchanged.

Before the pilot, fixed its budget to seed1901,64/16 training/held-out sequences,
300 steps,48/16 inner split, factors0.5/1/2 times log(sequence count),six updates
per candidate and ten per outer arm. Syntax/smoke precede the bounded single-
worker job. Candidate/refit/held-out data are fresh relative to901/902/903.
The pilot will evaluate unchanged TV0.12 and positive held-out gain separately
from numerical acceptance; it cannot provide large-sample/multi-seed acceptance.

## Step18b — adaptive pilot verification,2026-10-13

docs/validation/results/adaptive-duration-pilot-native-20261013.json records
the bounded seed1901 pilot at source commit324de0e100b44ecd3a85fad21b91170c589b56d1.
All ten source hashes, fresh64/16 observation replay,48/16 disjoint inner split
and hashes, likelihood selection,38 completed update indices and objective
monotonicity pass. Recomputed PMF TV and held-out/full gates match the report.
Adaptive PMFs remain positive/normalized and block-constant, with complete
disjoint coverage of every age1..260. The final penalized objective is separately
recomputed from LL, PMF, penalty and block count. Fixed-control LL is monotonic.
Maximum independent posterior-gradient error is5.08e-14, segment-reference LL
error6.71e-12, public recursion LL error1.14e-12 and state-probability error1.20e-13.
Both statistical gates fail, and the adaptive fit is worse than the fixed
control on held-out likelihood and three of four state TVs. Source remains
unchanged from the verified nine new cases plus24 existing focused cases.
No further broad/canonical/Trading acceptance is asserted.

## Step19a — matched-budget warm-start pilot,2026-10-13

The adaptive research fitter now accepts a copied positive normalized initial
PMF and validated contiguous block partition covering every age1..260. Existing
default behavior is unchanged. New checks cover warm-state immutability, invalid
support/partition rejection, actual warm-fit objective monotonicity and shared
inner warm-up with matched candidate update budgets and split isolation.
AST and an actual one-step warm/pruning smoke pass before longer execution;
Ruff/Black pass for the three changed/new Python files. The combined five-module
focused suite passes all35 cases, including the eleven adaptive cases.

The predeclared pilot uses fresh seed1902,64/16 training/held-out sequences of
300 rows and one48/16 inner split. Factors0.125/0.5/2 times log(training sequence
count) are compared separately from uniform starts (nine updates) and shared
three-update unpenalized warm starts (six additional updates). Selection uses
validation marginal LL only, with stronger factor on an exact tie. Outer arms
receive twelve total updates each: uniform12, warm3+9, fixed-width16 control12.
Shared inner warm-up reduces actual work to48 inner plus36 outer M-steps=84;
all individual adaptive paths have equal total budgets. The original1901 pilot
remains historical evidence at324de0e, not a current-source hash assertion.

## Step19b — paired warm-start pilot verification,2026-10-13

docs/validation/results/adaptive-warmstart-pilot-native-20261013.json records
the complete seed1902 experiment at1c5301c62c209a374239fa58aa8caf8bdb7db9af.
All eleven current source hashes,64/16 observation replay,48/16 inner split and
hashes, six factor/mode candidates, selected-factor argmax and all84 actual
M-steps pass. Both warm histories contain three updates; uniform candidates
have nine and warm candidates six, with matching total path budgets. Initial
warm continuation LL matches the shared preceding warm result in inner fits
and refit. Uniform/warm/fixed outer histories contain12/9/12 updates, respectively,
and the warm outer path additionally has its separately verified three-step
phase. Monotonicity is checked within each fixed-objective phase, not across
the change from warm penalty0 to the selected nonzero penalty.

All PMF positivity/normalization and full-age block coverage/density checks,
recomputed final penalized scores, TV, held-out gains and unchanged full gates
pass. Three public/all-logit checks cover3120 coordinates: maximum gradient
error4.86e-14, public recursion LL error3.64e-12 and state-probability error1.28e-13.
All three recovery gates fail. No source changes occurred during the pilot;
the combined35-case focused suite and AST/smoke/Ruff/Black evidence remain
applicable. No current canonical-suite, historical NPZ or Trading acceptance
is implied by this paired single-seed pilot.

## Step20a — resumable sample-size replication,2026-10-13

The adaptive research fitter supports checkpoint callbacks and guarded phase
history continuation. A real observation fit interrupted at update5 resumes
to exactly the uninterrupted final PMF; altered saved evidence is rejected.
The sample-size driver validates source/runtime/base/contract identity and
ordered completed arms. Its mocked CLI interrupts the1024 fitting phase,
preserves the completed128 arm, resumes the saved six-item history and skips
all completed work on the final no-op. Synthetic mocked results are test data,
not statistical evidence. All38 cases pass across the five focused modules;
AST and a real checkpoint/resume smoke pass. Ruff/Black cover all three changed
or added Python files. No production package code is changed.

Predeclared execution: fresh seed1903,128 and1024 paired300-row sequences,
the same24 fresh held-out sequences, fixed factor2, pseudocount1e-6, warm3 plus
penalized27 updates per arm. No renewed width/penalty selection uses these
outer observations. A phase snapshot writes to a temporary file then replaces
the JSON atomically. Resume re-scores its last saved PMF and verifies LL/objective
within1e-7 before proceeding; this may repeat an E-step, not an M-step. Source
and numerical/runtime metadata must remain identical; docs/report-only commits
can advance the branch while --base-commit retains the frozen code identity.
Native execution uses one worker/threads1 and a2400-second bound. Checkpoints
are local until explicitly pushed at verified report gates; they alone do not
guarantee survival of workspace loss. Statistical acceptance remains pending.

## Step20b — complete128-arm checkpoint,2026-10-13

At this checkpoint the sample-size snapshot, with source identity
421ef280fafce80f4ae1cedd3e30df303b2c36de, contained one complete128 arm plus
an active1024 warm-state update0. Complete
small-arm source hashes,128 training and24 held-out replay, paired-prefix hash,
warm0..3 and fit0..27 histories, phase continuity, monotonic fixed objectives,
positive/normalized full-age block PMFs, recomputed final score, TV and held-out
gates all pass. Maximum gradient error1.85e-14, segment-reference LL error1.14e-12,
public recursion LL error2.28e-13 and state-probability error7.59e-14 pass the
unchanged numerical gates. Small full recovery fails. The active large state
is execution/recovery data, not completed large-arm numerical/statistical
acceptance. The running local report may advance beyond this immutable snapshot;
resume retains --base-commit421ef28 even after docs/report-only commits.

## Step20c — large-fit execution checkpoint,2026-10-13

The immutable report snapshot advances the active1024 phase to warm3 plus
fit10, preserving the complete128 result. Source hashes,1024 data replay,
paired128-prefix identity, contiguous phase histories, warm-to-fit evidence
continuity, monotonic fixed objectives, positive normalized full-age PMFs,
block coverage/density and independently recomputed active penalized score
are verified. The active fit has5/5/9/7 blocks. This is an execution/recovery
checkpoint only; no large-arm held-out/truth or public final-gradient acceptance
is inferred. The running local file may advance after snapshot capture. Restoring
this snapshot resumes with the same421ef28 source identity and re-scores fit10
before applying its remaining17 M-steps.

## Step20d — complete paired sample-size result,2026-10-13

The final adaptive sample-size report contains both128/1024 arms and active=null,
with source identity421ef28 unchanged. All60 actual M-step indices, eleven
source hashes, both training replays, paired128-prefix identity, identical24-
sequence held-out replay, warm/fit continuity and fixed-objective monotonicity
pass. Positive normalized full-age block PMFs, independently recomputed final
scores, TVs, held-out gains and full gates agree with the report. Completed
CLI resume skips both arms without fitting.38 focused cases and static/smoke
checks remain applicable; no algorithm/source change occurred during the experiment.

The two public gradient checks cover2080 duration coordinates: maximum error
1.85e-14. Maximum segment-reference LL error1.48e-12, public recursion LL error
2.28e-13 and state-probability error7.59e-14 pass the original gates.128 full
recovery fails;1024 full recovery passes for this one fresh paired seed.

The separate learned-family-bias report is bound to the final raw report SHA256.
Its evaluation-only block-median bounds and normalized forward-KL projections
are recomputed from the learned partitions and exact true PMF. Every small-arm
lower bound exceeds0.12; all large-arm feasible projection TVs are below0.12.
These are representation diagnostics, not new training/selection data or
minimum-normalized-TV claims. Multi-seed/current-canonical/production/Trading
and historical NPZ acceptance are not extended by this single paired result.

## Step21a — fresh-seed large-only replication,2026-10-13

The local runtime recycled after the prior completed result.53 required paths
were restored/read through the GitHub connector at3aeb2ff and their Git-blob
identities verified (12 restored files,41 already matching). Declared runtime/
test/build tools were reinstalled; Python3.12.14, NumPy2.5.4 and Torch2.14.1+cpu
match the previous experiment. Editable package install uses the sparse-snapshot
SCM version override0.0.0 after declared dependencies are satisfied.

The sample-size driver now accepts --seed (1903/1904/1905) and --large-only. The
default remains the original paired1903 experiment. Resume follows the requested
ordered arm list and rejects other seeds/contracts or a small active arm when
large-only is requested. CLI interruption/resume/completed-no-op tests cover
both paired and large-only modes, with a separate cross-seed/arm rejection case.
All40 focused cases pass in the restored environment. AST and actual-fit smoke
precede jobs; AST/Ruff/Black pass over42 scoped files.

The new replication contract fixes seeds1904/1905 with1024 training and24 held-
out300-row sequences per seed, one worker/threads1, warm3 plus fit27, factor2,
prior1e-6 and unchanged numerical/TV/held-out gates. Seeds are executed serially
with2400-second bounds and separate guarded phase/arm JSON reports; completed
seeds are verified/documented/pushed individually. No128 refits or renewed
penalty selection are performed. Large-only source hashes differ from421ef28
only through the driver extension; old reports keep their immutable identities.

## Step21b — seed1904 execution snapshot,2026-10-13

At this checkpoint source identity8fcb7ca was unchanged. The snapshot stored
warm3 and fit0..10 for the active1024-only seed1904 arm, with5/4/9/6 blocks.
All eleven source hashes,1024 training replay, phase/update completeness,
warm-to-fit LL continuity, monotonic fixed objectives, positive normalized PMFs,
full-age partition/density invariants and independently recomputed last penalized
score pass. No outer observations/truth are evaluated at this partial gate.
The snapshot is execution/recovery evidence, not a complete statistical or
public final-gradient result. Resume retains --seed1904, --large-only, source
8fcb7ca and the same software contract; seed1905 remains unstarted.

## Step21c — seed1904 completed and verified,2026-10-13

The final1904 report has active=null and one complete1024 arm at8fcb7ca.
All30 M-step indices, eleven source hashes, training/first128/held-out replay,
warm-fit continuity and fixed-objective monotonicity pass. Positive normalized
full-age block PMFs, independently recomputed final scores, TV/gain/full gates
and public numerical checks pass. Complete CLI resume skips the arm. Maximum
all-logit gradient error7.55e-15, segment-reference LL error1.82e-12, public
recursion LL error1.05e-11 and state-probability error1.38e-13 meet the gates.
Statistical recovery fails on bear TV0.128586; all other state TVs pass.

The separate1904 family-bias report records its raw-input SHA256, learned blocks,
median TV bounds and normalized forward-KL projection TVs. Bear's bound0.126560
exceeds0.12, proving failure for this learned partition independently of fitted
mass updates. This post-fit diagnosis does not alter the method or gate.1905
was still pending at this checkpoint under the same frozen code/contract.

## Step21d — seed1905 execution snapshot,2026-10-13

At this checkpoint the second replication at8fcb7ca reached warm3+fit10 with5/4/9/5
blocks. Eleven source hashes,1024 training replay, complete phase indices,
warm-to-fit LL continuity, monotonic fixed objectives, positive normalized full-
age PMFs, contiguous block coverage/density and independently recomputed active
penalized score pass. The snapshot is durable recovery data only; no1905 outer
truth/held-out or public final-gradient acceptance is asserted. Seed1904 remains
fully verified with failed bear recovery, independent of this partial run.

## Step21e — complete replication and combined verification,2026-10-13

The final1905 report has active=null and one complete1024 arm. All30 updates,
eleven current source hashes, training/first128/held-out replay, phase continuity
and monotonic fixed objectives pass. Positive normalized full-age block PMFs,
recomputed objective/TV/held-out/full gates and public numerical checks pass;
complete resume is a no-op. Maximum gradient error3.52e-14, segment-reference
LL error3.87e-12, public recursion LL error8.87e-12 and state-probability error
7.00e-14 meet the unchanged gates. Range/chaotic TVs fail recovery.

The SHA-bound1905 family diagnostic verifies median-TV bounds and normalized
forward-KL projections. Chaos's bound0.126638 exceeds0.12. Range's bound0.112068
and feasible projection0.115803 are below0.12, yet fitted TV0.125884 fails:
mass estimation/optimization can also contribute beyond representational bias.

The combined summary records immutable input-report SHA256s for1903/1904/1905,
all90 large-arm updates and method/software equivalence. Ten algorithm/core
source hashes match across the old/new driver versions; only the seed/large-only
CLI driver source differs. Python/NumPy/Torch and all method contract parameters
match. Independently recomputed state TVs and full gates give pass1/fail2;
maximum gradient/public-LL/state-probability errors are3.52e-14/1.05e-11/1.38e-13.
Forty focused cases and42-file static/smoke checks remain applicable. Numerical
verification passes; robust three-seed recovery is not accepted. Production,
joint Trading, current canonical-suite and historical NPZ acceptance are unchanged.

## Step22b — large-selection driver and restored native preflight,2026-10-13

New research driver/test owners: scripts/compare_adaptive_large_selection.py and
 tests/test_adaptive_large_selection.py. Atomic active snapshots bind seed,
protocol, software/source hashes, phase dataset hashes, penalty and update
budget. Completed phases form an ordered prefix and are re-scored without
extra M-steps; PMF/partition/history/objective checks use the existing fitter.
Selection is durably saved before outer data is generated. A factor2 winner
reuses its selected refit; otherwise a matched control is independently fitted.
--max-phases bounds new completed phases; completed evaluation is a no-op.

Seven new focused cases cover real checkpoint interruption at phase start/end
with identical resumed PMFs, phase limits, changed data/contract/PMF rejection,
validation-only scoring, exact-tie stronger-factor selection, control reuse and
separate control fitting, early outer-data/truth rejection, completed no-op,
and short real-fit/public numerical evaluation. The existing40 cases remain
covered. Fifty connector-restored inputs pass Git-blob identity verification;
one dependency reopen restored the old test's warmstart helper. Declared CPU
Torch/NumPy/test/static/build dependencies and editable install succeed.
Python3.12.14/NumPy2.5.4/Torch2.14.1+cpu match the prior experiment;44-file AST,
Ruff and Black pass. These are implementation/preflight checks, not fresh large
statistical recovery or production/joint Trading acceptance.

## Step22c — first large inner warm checkpoint,2026-10-13

At immutable source3966159902a48bfc2c1fad6a22c375a15cd71dee, seed1906
finishes three unpenalized inner-warm updates on768 of1024 training sequences.
The bounded --max-phases1 invocation exits0 at the completed phase boundary.
All13 source hashes, full protocol/software/seed equality, training and disjoint
768/256 split replay hashes, indices0..3 and monotonic fixed objective pass.
Independent fit_adaptive terminal replay validates positive normalized full-age
PMF/partition coverage/densities and reproduces saved LL/objective within the
existing1e-7 resume tolerance without any new M-step. Selection/evaluation remain
null and active is null. No truth/outer statistics are evaluated for this stage.
Raw checkpoint SHA256: 80d9c9556c1f81c929cbc15c67ff28ff2e801da9a871eeeb35dee07e78b191db.
The final unified preflight suite has47 passed in12.07s;44-file static checks
pass. Large predictive/recovery acceptance remains pending.

## Step22d — candidate_0 active execution snapshot,2026-10-13

The immutable raw snapshot captures update10/27 of factor0.125*log(768).
Thirteen current source hashes and unchanged prior warm/split/source/contract
identity pass. Active phase key, budget, complete history indices, monotonic
fixed objective, positive finite normalized full-age PMFs and contiguous
block/density coverage pass. Blocks67/73/57/36. Snapshot SHA256:
df35bace71808d4650f9fb58da3ca275030d0666e7b2be1bb53db24df67592ad. No independent terminal score, completed-candidate
validation or outer recovery claim is made at this checkpoint. Fitting continues.

## Step22e — completed candidate_0 and independent verification,2026-10-13

The bounded invocation exits0 with inner_warm and candidate_0 complete, active
null and selection/evaluation null. All27 updates (history0..27) use penalty
0.830473716643459=0.125*log(768); completed actual work is30 M-steps.
The original warm record and source/contract/software/data/split identity are
unchanged. All13 current source hashes and1024 training/768+256 split replay
hashes pass. Full positive normalized PMF/partition and monotonic fixed-objective
history checks pass. Independent terminal E-step replay reproduces saved
LL-499451.2684951689 and objective-499641.4594240784 within1e-7, with no new
M-step. Separately recomputed inner validation LL is-166868.3480591295 on256
observations; no labels/truth or outer data participate. Blocks67/73/57/36.
The SHA-bound phase-verification JSON records these values and the input report
SHA256eb7f047f6d431a4a23ab38eb6196ffdd71938338570df806898c85ea7ac9b416. No public large recovery or factor-selection conclusion
is asserted until remaining candidates/refits/evaluation finish. Prior47-case
and44-file preflight evidence remains applicable; sources are unchanged.

## Step22g — completed candidate_1 and independent verification,2026-10-13

The bounded invocation exits0 with warm/candidate_0/candidate_1 complete, active
null and selection/evaluation null. Candidate_1's27 updates (history0..27)
use penalty3.321894866573836=0.5*log(768); total completed work57 M-steps.
Prior completed phase records and all source/contract/software/data/split
identities are exactly preserved. All13 current source hashes,1024 training
and768/256 split replay, fixed-objective monotonicity, full positive normalized
PMF/partition consistency and terminal E-step replay pass. Saved LL/objective
are-499629.0984813526/-499811.8135767134, reproduced within1e-7 without new
M-steps. Blocks18/13/19/9. Independently scored inner validation LL on256
sequences is-166776.27502647758: +92.07303265191149 nats versus candidate_0 on the
identical validation observations. This is a predictive score comparison, not
fine-PMF recovery, significance or final factor selection. The third candidate
and outer refits/evaluation remain pending. SHA-bound verification JSON records
input hash7f7cb99c8efeede4917917d84fc844d28b0e3eb4a8dfc2f3c0c3cd8d26ad9bb2; prior47-case/44-file preflight remains applicable.

## Step22j — completed seed1906 large-selection result,2026-10-13

The bounded batch exits0 after candidate_2, full_warm and selected_refit; the
selected factor2 reuses its identical fit as control. All114 actual M-steps
complete. Independent verifier replays all13 source/software/contract identities,
1024 training and disjoint768/256 split hashes, all candidate validation scores
and the fixed selection rule, all phase budgets/indices/objective/PMF/partition
checks and warm-to-fit LL continuity. New candidate_2/full_warm/refit terminal
E-steps independently reproduce saved LL/objectives within1e-7; prior warm and
candidate_0/1 records remain identical to their verified checkpoints.

Outer24-sequence replay/hash, uniform baseline, fitted LL, PMF TVs/means/RMSE,
positive predictive gain and full gates are independently recomputed. Public
all-logit gradient, segment likelihood and filtering checks pass: max gradient
1.4433e-15, public recursion LL4.5475e-12 and state-probability1.0148e-13.
Completed CLI resume exits0 without new fitting/evaluation. SHA-bound final
verification records input report hash7f3c68610d2dc4fac25d507a8829dce43f198f9ad747a523c1d4304dc85599f5.
Post-fit learned-partition TV bounds/projections are evaluation only; they do
not change learning/selection/gates. Numerical verification passes while range/
chaotic fail fine-PMF recovery. Prior47 tests/44-file static checks remain
applicable; sources are unchanged. Remaining1907/1908 and production/joint
Trading/historical NPZ acceptance are not asserted.

## Step22k — two fresh inner warm starts after control-plane reload,2026-10-13

Current global instructions/workflow/coding, NHSMM project instructions/workflow
and regime-v2 AGENTS are fully reread through the GitHub connector. The explicit
active regime-v2 branch persists over default develop headers. Current13-source
hashes match completed1906; the existing47-case/44-file static/real-smoke evidence
remains applicable. Environment is still Python3.12.14/NumPy2.5.4/Torch2.14.1+cpu.

The seed1907 and1908 --max-phases1 invocations each exit0 at completed inner_warm,
with active/selection/evaluation null. Each has768 fit and256 validation sequences
from1024 independently generated300-row observations; all data and split hashes
are independently replayed. Their training hashes differ. Source/software/seed/
protocol identity, history0..3, positive finite normalized full-age PMFs, block
coverage/densities and monotonic fixed objective pass. Terminal fit_adaptive
E-step replay reproduces each saved LL/objective within1e-7 without new M-steps.

| Seed | Blocks bull/bear/range/chaotic | Terminal LL | Terminal objective | Raw SHA256 |
| --- | --- | ---: | ---: | --- |
|1907|260/253/233/186|-499953.9123828681|-499953.9201610947|7658a145d8c31e55ea720566d8ac1e9550e5e79074247a2e8428f1d321acbab5|
|1908|254/254/245/153|-500322.4408234874|-500322.4485786696|21ca554f6834e61ce381de4879ee01faca9498c3cff7e2033ddc9b100a74dd6e|

These six completed M-steps are warm execution/recovery evidence only. No outer
observations/truth, factor selection, final numerical or statistical recovery
acceptance is asserted for these seeds. Compute concurrency stays1; independent
connector reads and checkpoint reads are bounded parallel. Source3966159, all
factors/budgets/gates, known nuisance controls and oracle OFF are unchanged.

## Step22m — seed1907 first candidate completion,2026-10-13

A lost internal session reference was distinguished from remote movement: the
branch still pointed to e172589. All54 required local inputs/checkpoints were
compared against current connector blob identities and passed; Python3.12.14/
NumPy2.5.4/Torch2.14.1+cpu persisted. No source or dependency changes invalidate
the existing static/real-smoke evidence.

Seed1907's bounded invocation exits0 with inner_warm/candidate_0 complete and
active/selection/evaluation null. All27 phase updates at0.125*log(768) pass
source/data/split/history/PMF/partition/objective checks. The warm record remains
identical. Terminal E-step replay reproduces LL-499961.7578440498 and objective
-500156.10102691303 within1e-7 without new M-steps. Independent256-sequence
inner validation LL is-166754.64328071033; blocks76/63/65/34; total30 M-steps.
SHA-bound phase verification uses input hashbf46e82188f8b3ccae5471055fbe0c0994bd4606e8461365ab290f3fb7c0f9b3. Selection, outer
statistics and public final numerical/recovery gates are still pending.

## Step22o — seed1908 completion and two-seed candidate join,2026-10-13

Seed1908's bounded invocation exits0 with warm/candidate_0 complete, all27
updates and active/selection/evaluation null. Source/software/contract and
1024-data/768+256 split replay, positive normalized full-age PMFs/partitions,
phase indices, monotonic fixed objective and warm-to-fit continuity pass.
The original warm record is preserved. Independent terminal E-step replay
reproduces LL-500336.32880270877/objective-500538.9761965589 within1e-7 without
new M-steps; independent inner256-sequence validation LL is-167330.094205874.
Blocks68/80/63/37. SHA-bound phase report uses input hash42f05224570252ffd12430ba36387b05da731f03015aae5a2a2d3656b4e2fefc.

The batch join verifies equal source/software/contract and distinct training
hashes for1907/1908. Both27-update first candidates pass,54 new updates and60
total including prior warm starts. The candidate-batch JSON binds both exact
raw report hashes and verification metrics. No absolute cross-seed LL ranking,
factor choice, outer statistics or public final recovery result follows. Source
and environment are unchanged, so prior47-case/44-file preflight remains valid.

## Step22p — recovered native runtime preflight,2026-10-13

The prior virtual environment was lost;54 retained source/doc/test/checkpoint
files match current connector blob identities at057717d. A transient exec
transport interruption cancelled the initial network approval without a
returned decision. The offline cache lacked Torch. After transport recovered,
normal declared CPU dependencies install successfully; no policy escalation
or alternate runtime is used. Editable install succeeds. Restored versions are
Python3.12.14/NumPy2.5.4/Torch2.14.1+cpu, matching checkpoint identity.
Forty-seven focused cases including real fit/resume/public numerical smoke
pass in48.53s;44-file AST/Ruff/Black pass. Both candidate_0 results remain intact.
Next: candidate_1 factor0.5 for1907/1908 with unchanged3966159 and oracle OFF.

## Step22r — seed1907 second candidate completed,2026-10-13

The bounded invocation exits0 at complete candidate_1,27 updates with penalty
3.321894866573836,57 total actual M-steps. Prior completed phase records are preserved.
All13 source/software/contract identities,1024 training and768/256 split replay,
phase indices, positive normalized full-age PMFs/partitions, fixed-objective
monotonicity and warm-to-fit continuity pass. Independent terminal E-step
reproduces LL-500167.31410301256/objective-500290.23463600525 within1e-7 without
new M-steps. Independently scored validation LL-166628.70723221602 improves
on candidate_0 by125.93604849430267 nats using identical256 sequences.
Phase evidence binds raw SHA256b061d60828e9af4bc9ebf122565cc648ead559e136581febfcab080fbeb2d37d. No factor selection or outer
recovery/public final acceptance is claimed yet.

## Step22t — second inner candidate batch independently verified


Seeds1907/1908 each finish27 candidate_1 updates at factor0.5 (54 new,
114 cumulative M-steps). Current source/software/contract equivalence and
distinct training datasets are checked. Each source identity, regenerated
training/split hash, preserved warm/candidate_0 record, ordered phase/budget,
monotone objective, positive normalized4x260 PMF and contiguous constant-density
partition passes. Terminal training E-step replays agree within1e-7; both
candidate_1 and candidate_0 are independently scored on the same per-seed256
inner validation sequences. The47 focused tests and44-file static checks from
Step22p apply to the unchanged source; no additional M-step is performed by
verification. Both main reports are SHA256-bound by individual phase evidence
and results/adaptive-large-selection-candidate1-batch-native-20261013.json.
No outer evaluation, factor selection, true-PMF recovery, production or legacy
NPZ acceptance is claimed. Oracle remains OFF.

## Step23-1907-active

Seed1907 candidate_2 factor2 checkpoint: 5/27 updates,
blocks5/4/8/5. Source/software/contract/data/split/prior phase preservation,
history/objective/full-age PMF and partition checks pass. This snapshot is
recovery evidence; terminal independent replay and factor choice remain pending.
Native fitting continues at unchanged source3966159; oracle OFF.

## Step23-1907-complete

Seed1907 candidate_2 factor2 completes27 updates (84 total M-steps).
Source/software/contract/data/split/history/objective/full-age PMF/partition and
independent terminal training E-step replay pass. All prior phases are preserved
exactly. All three candidates are independently scored on the same256 inner
validation sequences and the selection matches the maximum with stronger-factor
tie break. Factor2 validation LL-166599.38091861576, chosen factor2.0.
Blocks5/4/8/5. Main report SHA256172a2ce05a5d12d8f8c7ee9cc269fc1f208c00ea9317619edfa5420c660f6f61.
Selection is complete; outer evaluation remains null and full-data refit pending.
Frozen source3966159, method, budgets, gates and oracle OFF remain unchanged.

## Step23-1908-complete

Seed1908 candidate_2 factor2 completes27 updates (84 total M-steps).
Source/software/contract/data/split/history/objective/full-age PMF/partition and
independent terminal training E-step replay pass. All prior phases are preserved
exactly. All three candidates are independently scored on the same256 inner
validation sequences and the selection matches the maximum with stronger-factor
tie break. Factor2 validation LL-167164.1952762566, chosen factor2.0.
Blocks5/4/8/5. Main report SHA2568051085a684c3967fe17c83a9b303a9c7746e3af030623a56eaa2f30194f5f5d.
Selection is complete; outer evaluation remains null and full-data refit pending.
Frozen source3966159, method, budgets, gates and oracle OFF remain unchanged.

## Step23-batch — inner candidate selection complete

Both seeds1907/1908 finish candidate_2 factor2:27 updates each,54 new
M-steps and168 cumulative including prior inner phases. All independent terminal
replays, regenerated datasets/splits, source/software/contract checks and prior
phase preservation pass. All three candidate validation scores and chosen factors
are independently reproduced. Selection is complete and saved before outer data;
outer evaluation remains null. Next full_warm (3 updates on1024 sequences) and
selected_refit (27), with frozen source3966159/method/budgets/gates and oracle OFF.
SHA-bound paired evidence: results/adaptive-large-selection-candidate2-batch-native-20261013.json.

| Seed | Factor2 inner validation LL | Gain over0.125 (nats) | Selected factor | Blocks bull/bear/range/chaotic |
| --- | ---: | ---: | ---: | --- |
| 1907 | -166599.38091861576 | +155.26236209457 | 2.0 | 5/4/8/5 |
| 1908 | -167164.19527625659 | +165.89892961740 | 2.0 | 5/4/8/5 |

Factor comparisons use the same within-seed256 validation observations; absolute likelihoods do not rank seeds. Selection does not establish the true-PMF recovery or outer predictive gates.

## Step24-1907-active

Seed1907 full_warm completes3 updates on1024 sequences; selected_refit
factor2 reaches5/27, blocks5/4/8/6.
All13 source/software/contract/data/split identities, prior inner phase records
and selected factor are preserved. Full-age PMF/partition/history/objective and
warm/refit continuity checks pass. This is recovery evidence; terminal replays,
outer evaluation and recovery/numerical gates remain pending. Oracle OFF.

## Step24-1907-complete

Seed1907 completes full_warm3 plus selected_refit27 on1024 sequences,
114 total actual M-steps. Selected factor2 and its matched frozen-factor control
reuse the identical fit. All13 source/software/contract/data/split/phase/history/
objective/PMF/partition checks, independent warm/refit terminal E-step replay,
regenerated outer evaluation and public likelihood/gradient/filtering checks
pass. Prior inner records and selection remain identical; completed CLI resume
is a byte-preserving no-op. Outer gain0.00592318680781116 nats/row;
PMF TVs[0.10871667616130973, 0.11004879204839987, 0.12596853943506775, 0.11614778987526281]; mean RMSE0.8229498555275837.
Full recovery gateFAIL; failed states['range'].
Learned blocks[5, 4, 8, 6]; post-fit partition TV lower bounds[0.10845232583663918, 0.1092965235192142, 0.12497076145532046, 0.11436916048165213];
feasible forward-KL projection TVs[0.11094778037100295, 0.1146360225418479, 0.12722818351149318, 0.11685519102449432].
Verification is SHA-bound toe785101743f7cfb86ca146dbf3409969aef8e9ce5af97e8841e1ec2a6e0f2110.
Post-fit diagnostics do not enter fitting/selection or change the gates. Oracle
remains OFF; no production/joint-neural/legacy NPZ acceptance is claimed.

## Step24-1908-active

Seed1908 full_warm completes3 updates on1024 sequences; selected_refit
factor2 reaches5/27, blocks6/4/8/5.
All13 source/software/contract/data/split identities, prior inner phase records
and selected factor are preserved. Full-age PMF/partition/history/objective and
warm/refit continuity checks pass. This is recovery evidence; terminal replays,
outer evaluation and recovery/numerical gates remain pending. Oracle OFF.
