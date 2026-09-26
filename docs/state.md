# NHSMM State

Tracks repository readiness before NHSMM is evaluated as a temporal-model candidate for `awa-si/nautilus`.

## Target

Determine whether a causal NHSMM provides robust, permutation-invariant, out-of-sample incremental information about:

- latent episode duration;
- transition dynamics;
- near-term change hazard;

relative to:

1. no temporal model; and
2. Nautilus' current Gaussian HMM, including current persistence/duration shaping.

Latent states remain neutral (`0..K-1`) during the core evaluation. Do not map them to Nautilus H1 structure classes.

## Current status

**Phase:** evaluation-harness readiness.

The P0 causal/runtime/artifact blockers identified before the historical A/B/C evaluation are closed.

For `causal=True`, NHSMM uses a dynamic causal boundary-time hazard contract: the information set `F_t` determines the duration/end hazard for boundary `t -> t+1`. Filter/runtime and causal `forward()`/Viterbi use the same `(latent_state, episode_age)` ordering and do not consume future observations.

The canonical `DefaultEncoder` runtime is incremental. It retains only the bounded convolution history plus LSTM hidden/cell state, so each accepted observation is encoded once. Custom causal encoders without the streaming contract remain on the explicit correctness-first prefix fallback. The canonical one-timestep path now evaluates its single causal convolution window as the exactly equivalent flattened linear transform and evaluates the single-layer LSTM recurrence directly from the existing `nn.LSTM` parameters, avoiding generic sequence dispatch without changing model parameters or artifact state.

The incremental inference hot path is profiled and optimized. Gaussian emissions use the exact analytical diagonal-Gaussian log density instead of constructing a `MultivariateNormal` per bar, while runtime filtering uses an internal kernel for already-normalized model-produced duration/transition scores. That internal kernel now constructs its already-normalized posterior state without repeating the public `HSMMFilterState` validation pass; the public constructor and `filter_step()` remain fully validating external/reference contracts.

Production artifact v1 persists resolved `ModelConfig`, canonical `DefaultEncoder` configuration, redundant schema metadata, and the complete model/distribution `state_dict`. Loading is fail-closed and strictly validates artifact version, feature/state/duration dimensions, causal mode, distribution types, encoder metadata, and model state compatibility.

The maintained automated test contract is now the normal pytest-discovered `tests/test_*.py` suite. Historical tests and experiments targeting removed constructors or obsolete evaluation semantics have been removed rather than used to restore stale APIs. `.github/workflows/smoke.yml` runs the full discovered suite and the artifact-loaded CPU runtime benchmark on a clean runner. Testing policy is documented in `docs/testing.md`.

`expected remaining duration` is intentionally not part of the public runtime contract. Survival/end-within-horizon probabilities remain the canonical duration outputs; a scalar expectation is only an optional derived diagnostic under an explicit frozen-current-`F_t` assumption.

## Completed contracts

### Causal model foundation

- [x] `ModelConfig.causal` controls causal model construction.
- [x] Causal encoder mode is unidirectional with left-only convolution padding.
- [x] Initial causal context uses the first causal timestep rather than future-pooled context.
- [x] Custom encoders used with `causal=True` must explicitly declare causal behavior.
- [x] Caller controls runtime device through `NHSMM(..., device=...)`.

### Causal filtering and decoding

- [x] `HSMMFilterState` represents normalized `P(z_t, age_t | F_t)` as `[B,K,D]`.
- [x] Duration index `0 == d=1`; age index `0 == age=1` across forward/Viterbi/filter/runtime.
- [x] Duration PMFs convert to end/continue hazards without NaN propagation on impossible ages.
- [x] Continuation increments age without transition; transition occurs only at an episode boundary.
- [x] Self-transition starts a new episode at age one.
- [x] Initial segments are transition-independent in forward/Viterbi.
- [x] Duration-dependent Viterbi transitions use predecessor-segment duration.
- [x] Causal `forward()`/Viterbi use the same dynamic hazard ordering as filter/runtime.
- [x] Malformed/non-finite state, duration, transition, dtype, and device inputs fail closed.

### Causal invariants

- [x] Posterior normalization.
- [x] Deterministic/truncated duration support.
- [x] Prefix invariance under future-observation changes.
- [x] Right-padding/mask invariance for valid causal prefixes.
- [x] Repeated runtime `step()` matches model-bound filtering within tolerance.
- [x] Duplicate/older timestamps are rejected before state mutation.
- [x] Timestamp mode cannot change without `reset()`.

### Temporal outputs

- [x] Current latent-state posterior.
- [x] Current state-age posterior.
- [x] One-step active-episode end probability.
- [x] Configurable-horizon survival probability.
- [x] Configurable-horizon end-within probability.
- [x] Boundary transition joint mass `[B,K,K]`.
- [x] Next-episode state joint mass `[B,K]`.
- [x] Next-state prior before `x_{t+1}` `[B,K]`.
- [x] State-change probability `[B]` distinct from episode-end probability.
- [x] Expected remaining duration reviewed and excluded from the public runtime API.

### Inference/runtime

- [x] Model-bound filtering requires eval mode and uses inference mode.
- [x] Mutable runtime state is separate from model parameters.
- [x] Production inference preparation/loading path uses strict `state_dict` loading and default parameter freezing.
- [x] Canonical `DefaultEncoder` has bounded incremental CNN/LSTM state.
- [x] Incremental DefaultEncoder convolution reuses the existing Conv1d weights as an exactly equivalent one-window linear transform.
- [x] Incremental DefaultEncoder LSTM reuses the existing `nn.LSTM` parameters in an exactly equivalent one-step recurrence without adding artifact parameters.
- [x] Gaussian runtime emission scoring avoids per-step `MultivariateNormal` construction and is numerically equivalent to the diagonal covariance model.
- [x] Runtime skips redundant duration/transition normalization only for canonical model-produced normalized scores; the public validated filter path remains unchanged semantically.
- [x] Internal normalized runtime filtering skips the redundant second `HSMMFilterState` validation only after explicitly normalizing the posterior; public state construction remains fail-closed and fully validated.
- [x] Artifact-loaded CPU latency/allocation benchmark exists and verifies bounded runtime-state size.
- [x] Benchmark latency and allocation passes are separated so `tracemalloc` does not contaminate latency measurements.

### Artifact contract

- [x] Artifact versioning (`artifact_version == 1`).
- [x] Resolved `ModelConfig` persisted.
- [x] Canonical `DefaultEncoder` configuration persisted.
- [x] Causal/non-causal mode persisted and cross-validated.
- [x] Distribution/model state persisted through complete `state_dict`.
- [x] Feature/state/duration/distribution/encoder schema validated on load.
- [x] Unknown or incompatible artifacts fail closed.

### Repository/test contract

- [x] Standard pytest discovery is restricted to `tests/test_*.py` through `pyproject.toml`.
- [x] Current filtering/general contracts migrated into discovered tests.
- [x] Obsolete historical model/distribution/context tests removed.
- [x] Obsolete tuning and OHLCV experiment scripts/docs removed.
- [x] Smoke CI runs the full discovered pytest suite rather than a hand-maintained file list.
- [x] Test/verification policy documented in `docs/testing.md` and linked from README/CONTRIBUTING.
- [x] Package maturity classifier aligned with the documented pre-1.0 research stage (`Development Status :: 3 - Alpha`).

## Verification

Observed evidence relevant to the current contract:

- `tests/test_default_distributions.py`: `10 passed in 2.16s` during focused distribution hardening.
- Duration-indexing run `36248632077`: `5 passed in 1.48s` after forward/Viterbi indexing fixes.
- Smoke run `36251179309`: causal dynamic-hazard integration passed after replacing the stale finite-impossibility sentinel contract.
- Smoke run `36251977802`: right-padding/mask invariance passed.
- Smoke run `36252656161`: strict inference loading and round-trip inference equivalence passed.
- Smoke run `36253244202`: incremental encoder/runtime integration passed; prior focused suite reported `27 passed`.
- Smoke run `36254069155`: artifact v1 plus production benchmark integration passed; prior focused suite reported `31 passed`.
- Smoke run `36255390273` on commit `284468dcc62f00f3a77d7fe1f130425bd0fee52d`: success; full canonical discovery reported `56 passed in 2.27s` with no pytest warnings.
- The historical artifact-loaded CPU benchmark from run `36255390273` reported p50 `2.781659 ms`, p95 `2.857897 ms`, mean `2.7841281 ms`, Python traced peak `26,530` bytes, and constant runtime state `151` tensor elements. That latency loop ran under `tracemalloc`; it is retained as historical evidence but is not a clean latency baseline.
- Local sparse connector-workspace profiling on the same host/runtime shape (`K=3`, `F=4`, `D=5`, batch 1) measured the pre-optimization hot path at approximately mean `1.64 ms`, p50 `1.12 ms`, p95 `1.89 ms`; after direct diagonal-Gaussian scoring plus the normalized runtime filter kernel, repeated measurements were approximately mean `1.00-1.06 ms`, p50 `0.70-0.75 ms`, p95 `1.01-1.17 ms`.
- The internal normalized runtime filter kernel was compared against public `filter_step()` over 200 randomized normalized duration/transition cases, including impossible-duration support; maximum observed absolute posterior difference was `9.54e-7`.
- Smoke run `36257892827`: success; full canonical discovery reported `56 passed in 2.42s`. The artifact-loaded CPU benchmark after separating latency from allocation tracing reported p50 `1.105176 ms`, p95 `1.170783 ms`, mean `1.113136 ms`, Python traced peak `14,619` bytes, and constant runtime state `151` tensor elements.
- Local profiling of the subsequent one-timestep encoder fast paths measured approximately mean `0.785 ms`, p50 `0.548 ms`, p95 `0.837 ms` on the same sparse-workspace runtime shape. Streamed encoder outputs remained equivalent to full causal encoder outputs with maximum observed absolute difference `2.83e-7` across multiple seeds, batch sizes, and sequence lengths.
- Smoke run `36260442193`: success; full canonical discovery reported `56 passed in 2.24s`. The artifact-loaded CPU benchmark with the incremental Conv/LSTM fast paths reported p50 `0.987014 ms`, p95 `1.043253 ms`, mean `0.990100 ms`, Python traced peak `14,619` bytes, and constant runtime state `151` tensor elements.
- Local same-host profiling of the normalized filter kernel before/after removing its redundant second state validation measured approximately p50 `0.102 ms` -> `0.077 ms` and mean `0.113 ms` -> `0.084 ms`, about a 25% kernel-level reduction. The added regression test compares the internal normalized path directly with public `filter_step()` and verifies normalization within `1e-6` tolerance.
- **Smoke run `36261713973`: success; full canonical discovery reported `57 passed in 2.21s`.** Its artifact-loaded benchmark reported p50 `0.936711 ms`, p95 `0.980768 ms`, mean `0.939742 ms`, Python traced peak `14,619` bytes, and constant runtime state `151` tensor elements. This hosted-runner result supports the direction of the optimization but is not the direct A/B evidence because runner environment varies; the same-host local filter-kernel measurement is the direct performance comparison.

## Repository housekeeping

- Broad `torch>=2.2` clean-runner installation currently resolves a large CUDA distribution despite CPU execution; GitHub Actions should remain an integration diagnostic, not the routine edit/test loop.
- Static checks remain part of the repository verification contract: `ruff check nhsmm tests scripts` and `black --check nhsmm tests scripts`. They were not executed in the performance smoke because the workflow installs the runtime package plus pytest, not the dev extra.

## Nautilus evaluation gate

The NHSMM-side P0 gate is now satisfied. The next work is the Nautilus-facing evaluation harness, not additional latent-state semantic mapping.

Intended comparison:

```text
A0  no temporal model
B1  Nautilus raw HMM forward filter
B2  HMM + current persistence/duration shaping
C   causal NHSMM
```

All variants must use identical causally available observations and chronological evaluation windows.

Primary evaluation:

- predictive OOS log score / likelihood where comparable;
- duration/survival calibration;
- change-hazard Brier score and log loss at fixed horizons;
- downstream incremental information after controlling for current observations, HMM posterior uncertainty, and existing HMM age/persistence information.

H1 structure accuracy, Hungarian matching, state-plot plausibility, and in-sample likelihood alone are not acceptance criteria.

## Next slices

1. Define and implement the Nautilus-facing evaluation harness around artifact-loaded causal NHSMM inference without mapping latent states to H1 classes.
2. Run the chronological A0/B1/B2/C comparison only after the harness, feature contract, and evaluation-window contract are fixed.
