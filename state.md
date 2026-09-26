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

`expected remaining duration` is intentionally not part of the public runtime contract. Under dynamic causal hazard semantics, a scalar expectation would require freezing the current duration law into the future and would therefore be only a lossy derived summary of the canonical survival/end-within-horizon curve. It may be computed as an optional diagnostic under that explicit frozen-current-`F_t` assumption, but it is not a primary model output.

Production inference now has an explicit preparation/loading path separate from artifact deserialization. `prepare_inference(...)` requires initialized distributions, rejects non-finite model parameters, switches the model to eval mode, and freezes parameters by default. `load_inference_model(...)` constructs through the canonical `ModelConfig -> NHSMM` path, initializes canonical distributions, strictly loads a caller-deserialized `state_dict`, can require causal mode, and returns only an inference-prepared model.

The canonical causal `DefaultEncoder` now exposes bounded incremental state: the last `cnn_kernel - 1` raw observations plus LSTM hidden/cell state. `HSMMFilterRuntime.step(...)` uses that state so each new bar is encoded exactly once and no longer re-encodes or retains the full prefix on the default production path. Custom causal encoders that do not expose the incremental-state API remain on the explicit correctness-first full-prefix fallback.

Production artifact v1 is now explicit and fail-closed. It persists resolved `ModelConfig`, canonical `DefaultEncoder` configuration, redundant feature/state/duration/causal/distribution schema metadata, and the full model/distribution `state_dict`. Loading uses `torch.load(..., weights_only=True)`, validates metadata before model construction, strictly loads state, and rejects unsupported versions, schema mismatches, encoder mismatches, and non-canonical custom encoders. Artifact v1 intentionally supports only the reconstructible canonical `DefaultEncoder`.

The production CPU runtime benchmark now executes against a model that has been saved and reloaded through artifact v1. The clean-runner reference measurement for `K=3`, `F=4`, `D=5`, batch size 1 was p50 `2.718 ms`, p95 `2.820 ms`, mean `2.730 ms` over 128 measured steps after 16 warmup steps. Python traced peak allocation was `27,200` bytes and bounded runtime state remained constant at `151` tensor elements. These numbers are reference measurements, not acceptance thresholds.

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
- [x] Canonical `DefaultEncoder` runtime uses bounded causal CNN/LSTM state and encodes each accepted observation once; custom encoders without the stream API use the explicit full-prefix fallback.
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
- [x] Expected remaining duration contract reviewed: excluded from the public runtime API; optional derived diagnostic only under an explicit frozen-current-`F_t` assumption.

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
- [x] Explicit production inference preparation/loading path with strict `state_dict` loading, causal-mode guard, eval preparation, and default parameter freezing.
- [x] Incremental encoder state for the canonical `DefaultEncoder`; live default runtime no longer re-encodes the full prefix.
- [x] CPU latency/allocation benchmark on the artifact-loaded bounded incremental runtime.

### Artifact contract

- [x] Version artifacts (`artifact_version == 1`).
- [x] Persist resolved `ModelConfig` and canonical `DefaultEncoder` configuration.
- [x] Persist causal/non-causal mode redundantly in config, encoder metadata, and schema metadata.
- [x] Persist distribution/model state consistently through the full `state_dict`.
- [x] Validate feature/state/duration/schema and encoder metadata on load.
- [x] Fail closed on unknown versions, incompatible schema/config/encoder metadata, malformed state mappings, and strict state-load mismatches.

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
- Expected remaining duration review: excluded from the public runtime contract because under dynamic causal hazard semantics it would require freezing the current duration law into future boundaries and would duplicate the canonical survival curve as a lossy scalar summary.
- Local inference helper syntax validation: `nhsmm/inference.py` and `tests/test_inference.py` compile successfully.
- Clean-runner smoke run `36252656161` on commit `7568672a0f2b774717bf2c5d8b8ad77131e9ffe8`: success; import smoke, general functional smoke, and causal temporal/inference tests all passed, including strict-load rejection and round-trip inference-output equivalence.
- Local incremental-encoder harness: causal `DefaultEncoder.forward(...)` and repeated `stream_step(...)` produced identical per-timestep contexts with max absolute difference `0.0`.
- Clean-runner smoke run `36253244202` on commit `9fc9f98f64be5ffe9058d8237a7ee99409732d32`: success; general functional smoke passed and the causal temporal/inference/incremental-runtime suite reported `27 passed, 1 warning in 1.99s`. The warning remains limited to the existing brute-force test helper converting a grad-enabled score to `float`.
- Local production artifact/benchmark module syntax validation passed for `nhsmm/artifact.py`, `nhsmm/benchmark.py`, `scripts/benchmark_runtime.py`, and their focused tests.
- Clean-runner smoke run `36253907613` on commit `c5875c9ea826194ebc19aab3563612e3c4824e67`: success; artifact v1 round-trip/fail-closed tests and bounded-state benchmark contract passed with the existing causal/runtime suite.
- Clean-runner smoke run `36254069155` on commit `cb8fa8ec89c122196ba76ffcd8044dd47c48ccc2`: success; `31 passed, 1 warning in 1.98s`. The artifact-loaded CPU benchmark reported p50 `2.7177515 ms`, p95 `2.819508 ms`, mean `2.7303501 ms`, Python peak `27,200` bytes, and constant runtime state `151` tensor elements over 128 measured steps after 16 warmup steps.

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

1. Clean test discovery and migrate/delete stale legacy tests/scripts.
2. Define the Nautilus-facing evaluation harness around artifact-loaded causal NHSMM inference without mapping latent states to H1 classes.
