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
