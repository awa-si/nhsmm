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

**Phase:** pre-test integration readiness.

The causal `(latent_state, episode_age)` filter is wired to canonical NHSMM initial, duration, transition, emission, and causal-context outputs. A correctness-first online runtime advances the HSMM posterior through `step()` while keeping mutable runtime state separate from model parameters.

The current model-bound and online-runtime contracts are verified against the real canonical `NHSMM` in a clean repository checkout. Prefix invariance, right-padding invariance for valid prefixes, and repeated `step()` equivalence with `filter_model_sequence(...)` passed.

Multi-horizon survival/end-within-horizon forecasts are implemented from the current `(state, age)` posterior and current causal duration distribution. Transition forecasts expose episode-boundary mass, next-episode state mass, next-state prior, and state-change probability without mapping latent states to semantic H1 classes. These temporal-output contracts are clean-runner verified.

For `causal=True`, duration context is explicitly defined as a **dynamic causal hazard** contract: the current information set `F_t` determines the duration/end hazard used for boundary `t -> t+1`. The duration law is not frozen at episode start. Filter/runtime and causal `forward()`/Viterbi now use the same age/hazard ordering. The non-causal retrospective segment DP remains a separate inference path.

The runtime currently retains and re-encodes the causal observation prefix because the encoder does not yet expose incremental hidden/convolution state. This is causally correct and suitable for contract testing, but it is not the final bounded-cost hot path.

## Completed

### Causal model foundation

- [x] `forward()` permits only duration `d=1` at `t=0`.
- [x] `ModelConfig.causal` exists.
- [x] Causal encoder mode is unidirectional and uses left-only convolution padding.
- [x] Causal initial context uses the first causal timestep rather than future-pooled context.
- [x] Custom encoders used with `causal=True` must explicitly declare causal behavior.
- [x] Caller controls runtime device via `NHSMM(..., device=...)`.

### Causal filter core

- [x] `HSMMFilterState` represents normalized posterior `P(z_t, age_t | F_t)` as `[B,K,D]`.
- [x] Duration PMFs are converted to end/continue hazards without NaN propagation on impossible ages.
- [x] Continuation increments age without transition.
- [x] Transition occurs only after an episode boundary.
- [x] Self-transition starts a new episode at age one.
- [x] Current state posterior and age posterior are exposed.
- [x] One-step current-episode end probability is exposed.
- [x] Malformed/non-finite state, duration, transition, dtype, and device inputs fail closed.

### Model-bound filtering

- [x] `filter_model_sequence(...)` uses canonical `_build_sequence_set(...)` emissions/context.
- [x] Initial/duration/transition scores come from the current model distributions.
- [x] Boundary `t-1 -> t` uses duration/transition information from `F_{t-1}`.
- [x] `x_t` enters only through the emission after boundary propagation.
- [x] Filtering requires causal config, initialized distributions, and eval mode.
- [x] Distribution output shapes are validated.
- [x] Padded timesteps remain outside normalized trace mass.

### Online runtime

- [x] `HSMMFilterRuntime.step(...)` provides a causal online API.
- [x] Mutable `HSMMRuntimeState` is separate from trainable model parameters.
- [x] Runtime carries the previous boundary duration/transition scores so each accepted observation advances the HSMM posterior exactly once.
- [x] Timestamped mode requires strictly increasing timestamps.
- [x] Duplicate or older timestamps are rejected before runtime mutation.
- [x] Timestamp mode cannot silently switch after initialization; `reset()` is required.
- [x] Runtime accepts one timestep at a time with fixed batch size.
- [x] Current reference runtime re-encodes retained prefix until the encoder gets an incremental-state contract.
- [x] `forecast_survival(...)` exposes configurable-horizon current-episode survival/end risk from `F_t`.
- [x] `forecast_transition()` exposes one-step episode-boundary and latent-state transition quantities from `F_t`.

### Distribution hardening

- [x] `Categorical.rsample()` calls `nnF.gumbel_softmax(...)` directly; the previous package-init alias coupling is no longer required.
- [x] `Categorical.sample(...)` and `IndependentStudentT.rsample()` preserve sample/batch shape contracts.
- [x] Duration and transition structural masks are hard zero-support masks after normalization.
- [x] Duration index `i` denotes total duration `i + 1`; transition duration axes use the same convention.
- [x] `Transition(max_duration=None)` and `max_duration=1` are supported consistently.
- [x] Gaussian and Student-t emission construction/log-probability contracts are covered by focused tests.
- [x] `nhsmm/distributions/README.md` documents the current `default.py` shapes and semantics.
- [x] `tests/test_default_distributions.py` covers the focused distribution contract.
- [x] Causal duration-context semantics are explicitly documented as dynamic boundary-time hazard semantics under `F_t`.

## P0 blockers before historical A/B/C evaluation

### Temporal outputs

- [x] Current latent-state posterior.
- [x] Current state-age posterior.
- [x] One-step active-episode end probability.
- [x] Survival probabilities at configurable future horizons.
- [x] Probability current episode ends within configurable future horizons.
- [x] Transition outputs exposed and clean-runner verified for downstream evaluation:
  - boundary transition joint mass `[B,K,K]`;
  - next-episode state joint mass `[B,K]`;
  - next-state prior before `x_{t+1}` `[B,K]`;
  - state-change probability `[B]` distinct from episode-end probability.
- [ ] Expected remaining duration, if retained after contract review.

### Causal invariants

- [x] Core posterior normalization.
- [x] Deterministic/truncated duration support without NaN propagation.
- [x] Malformed probability rows rejected before posterior contamination.
- [x] Real-model prefix invariance: changing only observations after `t` leaves filtered output through `t` unchanged.
- [x] Real-model streaming equivalence: repeated `step()` matches `filter_model_sequence(...)` within tolerance.
- [x] Duplicate timestamps are rejected without advancing or replacing runtime state.
- [x] Forward/Viterbi/filter/runtime use one canonical duration convention: duration index `0 == d=1`, age index `0 == age=1`.
- [x] Initial segments are transition-independent in forward/Viterbi recursion.
- [x] Duration-dependent Viterbi transitions are indexed by the predecessor segment duration.
- [x] Duration-context semantics resolved: `causal=True` uses dynamic causal boundary-time hazard under `F_t`; no episode-start freezing.
- [x] Causal `forward()`/Viterbi use the same dynamic age/hazard boundary ordering as filter/runtime.
- [x] Right-padding/mask invariance for valid causal context, emission, forward-posterior, and filter prefixes.

### Inference/runtime

- [x] Model-bound filtering refuses training mode and runs under `torch.inference_mode()`.
- [x] Mutable online state is separate from model parameters.
- [ ] Explicit production inference preparation/loading path.
- [ ] Incremental encoder state so the live path no longer re-encodes full prefix.
- [ ] CPU latency/allocation benchmark after semantics stabilize.

### Artifact contract

- [ ] Version artifacts.
- [ ] Persist `ModelConfig` and encoder configuration.
- [ ] Persist causal/non-causal mode.
- [ ] Persist distribution/model state consistently.
- [ ] Validate feature/state/duration/schema metadata on load.
- [ ] Fail closed on incompatible artifacts.

## Verification

Observed:

- GitHub Actions smoke run `36234225103`: success; original general functional smoke passed.
- `tests/filtering.py` local hardening run: `8 passed`.
- `tests/test_categorical_rsample.py` local targeted run: `1 passed`.
- Lightweight local model-bound filter harness: passed.
- Lightweight local online-runtime harness: exact batch/streaming equivalence; duplicate timestamp rejected without mutation.
- Clean-runner smoke run `36237525927`: success; model-filter/runtime tests reported `6 passed in 1.68s`.
- Clean-runner smoke run `36244181797`: success; `tests/test_model_filter.py`, `tests/test_runtime.py`, and `tests/test_survival.py` reported `11 passed in 1.32s`.
- Clean-runner smoke run `36244797811`: `16 passed, 1 failed`; the only failure was a transition test-fixture axis-indexing mistake, not an implementation failure. The fixture was corrected to index `[batch, source_state, duration, destination_state]` explicitly.
- Clean-runner smoke run `36244956483` on commit `4d80000a0d6f8398746ea65aaa6c621c782ad238`: success; `tests/test_model_filter.py`, `tests/test_runtime.py`, `tests/test_survival.py`, `tests/test_transitions.py`, and `tests/test_runtime_transition.py` reported `17 passed in 1.84s`.
- Focused `default.py` distribution contract run: `tests/test_default_distributions.py` reported `10 passed in 2.16s`.
- Duration-indexing run `36248255531`: `1 failed, 4 passed`; the strengthened test exposed that a first segment with `d>1` incorrectly depended on transition logits.
- Duration-indexing run `36248632077`: success after recursion fixes; `tests/test_duration_indexing.py` reported `5 passed in 1.48s`.
- Duration-context contract review established dynamic causal hazard semantics under `F_t` for boundary `t -> t+1`.
- Focused local causal-hazard harness verified normalized-forward/filter equivalence, deterministic duration-two ages, Viterbi against brute-force dynamic-hazard paths, and finite gradients through the causal recursion.
- Clean-runner smoke run `36250935387`: failed only because `tests/general.py` still asserted the old finite `NEG_INF` sentinel for impossible t=0 ages; the new causal recursion intentionally uses exact `-inf` support.
- Clean-runner smoke run `36251179309` on commit `9e4cf795c80c4a2edd2e9d19648d13613e4a099c`: success; general functional smoke passed and the causal inference suite reported `20 passed, 1 warning in 1.83s`. The warning is limited to a brute-force test helper converting a grad-enabled score to `float`.
- Clean-runner smoke run `36251977802` on commit `56bece5ceddd8d6611a011e8d25b2aaf60ccefa0`: success; general functional smoke and causal temporal output tests passed with the added right-padding invariance regression. The temporary workflow trigger was restored immediately after the run started.

## Repository drift / housekeeping

- Most historical tests are not named for standard pytest discovery.
- `tests/neural.py`, `tests/ctx_encoder.py`, `scripts/tune.py`, `scripts/tune_gaussian.py`, and related historical consumers still reference removed/obsolete APIs or evaluation semantics.
- Do not restore obsolete APIs merely to satisfy those files; migrate or remove them after the canonical runtime contract stabilizes.
- `pyproject.toml` still says `Development Status :: 4 - Beta` while project guidance describes pre-1.0 research status.
- Broad `torch>=2.2` clean-runner installation currently downloads a large CUDA distribution despite CPU execution; do not use Actions for routine small checks.
- `.github/workflows/smoke.yml` is a clean-runner integration diagnostic; routine small checks should stay local.

## Nautilus evaluation gate

Do not start the main historical comparison until the remaining P0 causal/runtime/output items above are complete.

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

1. Decide whether expected remaining duration belongs in the public runtime contract.
2. Add incremental encoder state for bounded-cost live inference.
3. Clean test discovery and migrate/delete stale legacy tests/scripts.
4. Implement artifact/version/loading contracts before Nautilus production integration.
