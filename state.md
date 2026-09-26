# NHSMM State

This file tracks the minimum repository state required before NHSMM is evaluated as a temporal-model candidate for `awa-si/nautilus`.

## Target

Determine whether a causal NHSMM provides robust, permutation-invariant, out-of-sample incremental information about:

- latent episode duration;
- transition dynamics;
- near-term change hazard;

relative to both:

1. no temporal model; and
2. Nautilus' current Gaussian HMM, including its existing persistence/duration shaping.

NHSMM latent states must remain semantically neutral (`0..K-1`) for this evaluation. They must not be mapped to Nautilus H1 structure classes as part of the core experiment.

## Current status

**Phase:** pre-test integration readiness

The repository is not yet ready for the Nautilus historical A/B/C evaluation. Core HSMM sequence inference exists, but a causal online HSMM filtering contract and first-class survival/change-hazard outputs are still missing.

## Completed

### HSMM duration boundary

- [x] `forward()` only permits duration `d=1` at `t=0`.
- [x] The initial duration boundary is now consistent with the existing Viterbi endpoint constraint `d <= t + 1`.

Commit: `3042bbc96fe2bff83153b1a6aab0f03ac4e3477a`

### Causal encoder mode

- [x] `ModelConfig.causal` exists.
- [x] `DefaultEncoder(causal=True)` disables bidirectional recurrence.
- [x] Causal convolution uses left-only padding rather than symmetric future-visible padding.
- [x] `NHSMM` propagates the causal contract to the default encoder.
- [x] In causal mode, initial-state context is derived from the first causal context rather than full-sequence pooled context.
- [x] Custom encoders used with `causal=True` must explicitly declare causal behavior.

Commits:

- `4255cb9024ee3052872588ec3ba57e78a373bc12`
- `5525cf92315f933f0d8a56071d0b834757033cc5`
- `3b2aea996d6c0c790b845324adae4ca31c409446`

### Device ownership

- [x] `NHSMM(..., device=...)` allows the caller to choose the runtime device.
- [x] A supplied `ContextEncoder` is moved to the selected device.

Commit: `ba0848f771de1da8f337fa0047029a189ca7b144`

## Pre-test blockers

### P0 — causal HSMM filtering

- [ ] Define an online filtering state that represents the posterior over current latent state and current episode age/residual duration without forcing the active segment to terminate at the current prefix boundary.
- [ ] Add an incremental `step()` or equivalent causal filtering API.
- [ ] Ensure repeated processing of the same observation/timestamp cannot advance filter state twice once timestamp semantics are introduced.
- [ ] Keep filtering distinct from retrospective smoothing and Viterbi decoding.

The existing sequence `forward()` is an endpoint/segment recursion and must not be wrapped directly as a live `step()` implementation without deriving the correct active-segment filtering recursion.

### P0 — temporal outputs

- [ ] Expose current latent-state posterior.
- [ ] Expose posterior state-age or an equivalent state-duration sufficient statistic.
- [ ] Expose survival probabilities for configurable future horizons.
- [ ] Expose change hazard / probability that the current episode ends within configurable future horizons.
- [ ] Expose transition probabilities needed for downstream evaluation.
- [ ] Define expected remaining duration if it is mathematically well-defined under the selected duration contract.

These outputs must not require semantic state labels.

### P0 — causal invariants

- [ ] Prefix invariance: output at `t` must be unchanged when observations after `t` are modified.
- [ ] Streaming equivalence: repeated causal `step()` over a sequence must agree with equivalent causal batch filtering within tolerance.
- [ ] Mask/padding invariance for valid prefixes.
- [ ] Forward/Viterbi/filtering duration support must use the same duration indexing and boundary conventions.

### P1 — artifact contract

- [ ] Version persisted artifacts.
- [ ] Persist `ModelConfig` and encoder configuration.
- [ ] Persist causal/non-causal mode explicitly.
- [ ] Persist model/distribution state consistently.
- [ ] Validate `n_features`, `n_states`, `max_duration`, dtype/device expectations, and relevant schema metadata on load.
- [ ] Fail closed on incompatible artifacts rather than silently adapting model structure.

### P1 — inference mode

- [ ] Provide an explicit inference preparation path (`eval()` plus inference/no-grad semantics at the integration boundary).
- [ ] Ensure dropout/training behavior cannot affect production filtering.
- [ ] Keep mutable online filter state separate from trainable model parameters and offline training state.

## General functional smoke test

`tests/general.py` tracks the small deterministic contracts introduced during pre-test preparation. It is intentionally not a full statistical/model-quality test suite.

Current intended checks:

- explicit CPU device ownership;
- causal encoder is unidirectional;
- causal convolution preserves sequence length without right/future padding;
- causal prefix outputs do not change when only future observations change;
- HSMM `forward()` permits only duration `1` at `t=0`.

The test file is repository scaffolding until it is actually executed. Do not report it as passing until execution has occurred and the result was observed.

## Nautilus evaluation gate

Do not start the main historical model comparison until all P0 items above are complete.

The intended comparison is:

```text
A0  no temporal model
B1  Nautilus raw HMM forward filter
B2  HMM + current persistence/duration shaping
C   causal NHSMM
```

All variants must use the same causally available observations and chronological evaluation windows.

Primary evaluation families:

- predictive out-of-sample log score / likelihood where directly comparable;
- duration/survival calibration;
- change-hazard Brier score and log loss at fixed horizons;
- incremental downstream information after controlling for current observations, HMM posterior uncertainty, and existing HMM age/persistence information.

H1 structure accuracy, Hungarian state matching, visual state plausibility, or in-sample likelihood alone are not acceptance criteria.

## Later work

After causal correctness and the first research comparison are established:

- optimize online filtering to bounded `O(K * D)`-class state where practical;
- benchmark CPU inference latency and allocations;
- harden serialization/runtime compatibility;
- clean or migrate stale legacy tests/scripts;
- update public documentation from measured behavior rather than capability claims.
