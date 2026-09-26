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

The causal `(latent_state, episode_age)` filter is wired to canonical NHSMM initial, duration, transition, emission, and causal-context outputs. A correctness-first online runtime now advances the HSMM posterior through `step()` while keeping mutable runtime state separate from model parameters.

The runtime currently retains and re-encodes the causal observation prefix because the encoder does not yet expose incremental hidden/convolution state. This is causally correct and suitable for contract testing, but it is not the final bounded-cost hot path.

Multi-horizon survival/change-hazard outputs remain incomplete.

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

### Distribution hardening

- [x] Canonical package import fixes the `Categorical.rsample()` functional alias defect.
- [x] A pytest-discoverable categorical rsample test covers soft output, hard one-hot output, and gradient flow.
- [ ] Remove the remaining non-local `F = nnF` package-init coupling with a line-preserving direct patch.

## P0 blockers before historical A/B/C evaluation

### Temporal outputs

- [x] Current latent-state posterior.
- [x] Current state-age posterior.
- [x] One-step active-episode end probability.
- [ ] Survival probabilities at configurable future horizons.
- [ ] Probability current episode ends within configurable future horizons.
- [ ] Transition outputs exposed in the runtime result contract for downstream evaluation.
- [ ] Expected remaining duration, if retained after contract review.

### Causal invariants

- [x] Core posterior normalization.
- [x] Deterministic/truncated duration support without NaN propagation.
- [x] Malformed probability rows rejected before posterior contamination.
- [ ] Real-model prefix invariance observed passing in the current environment.
- [ ] Real-model streaming equivalence observed passing: repeated `step()` vs `filter_model_sequence(...)`.
- [ ] Mask/padding invariance for valid prefixes.
- [ ] Forward/Viterbi/filter/runtime duration indexing verified as one canonical convention.

A lightweight local runtime harness has already shown exact streaming/batch equivalence and duplicate-timestamp non-mutation. The repository real-model tests must still be executed against the current package checkout before these two invariants are marked complete.

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

- GitHub Actions smoke run `36234225103`: success.
- `tests/filtering.py` local hardening run: `8 passed`.
- `tests/test_categorical_rsample.py` local targeted run: `1 passed`.
- Lightweight local model-bound filter harness: passed.
- Lightweight local online-runtime harness: exact batch/streaming equivalence; duplicate timestamp rejected without mutation.

Present but not yet observed passing in a current full NHSMM checkout:

- `tests/test_model_filter.py`
- `tests/test_runtime.py`

Do not claim these real-model pytest files pass until actually executed.

## Repository drift / housekeeping

- Most historical tests are not named for standard pytest discovery.
- `tests/neural.py`, `tests/ctx_encoder.py`, `scripts/tune.py`, `scripts/tune_gaussian.py`, and related historical consumers still reference removed/obsolete APIs or evaluation semantics.
- Do not restore obsolete APIs merely to satisfy those files; migrate or remove them after the canonical runtime contract stabilizes.
- `pyproject.toml` still says `Development Status :: 4 - Beta` while project guidance describes pre-1.0 research status.
- Broad `torch>=2.2` installation caused the first CI smoke to download a large CUDA distribution despite CPU execution.
- `.github/workflows/smoke.yml` remains a clean-runner diagnostic; routine small checks should stay local.

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

1. Execute current real-model filtering/runtime tests in a current local checkout.
2. Add multi-horizon survival/change-hazard outputs to the causal runtime contract.
3. Verify one canonical duration indexing convention across forward/Viterbi/filter/runtime.
4. Add incremental encoder state for bounded-cost live inference.
5. Clean test discovery and migrate/delete stale legacy tests/scripts.
6. Implement artifact/version/loading contracts before Nautilus production integration.
