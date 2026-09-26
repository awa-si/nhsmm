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

**Last repository scan:** `develop` through `3a34c8f4e0f5adcc4cad0e12478aeb21902b6ebf`.

A standalone causal HSMM filtering core exists over `(latent_state, episode_age)` and has been hardened against malformed/non-finite probability inputs and tensor-contract mismatches. It is not yet wired into `NHSMM` as a public/incremental runtime API, and multi-horizon survival/change-hazard outputs are still missing.

The current repository contains both the canonical `NHSMM`/`ModelConfig` implementation and historical scripts/tests that target removed APIs. Those historical consumers are repository drift and are not part of the current model contract.

## Completed

### HSMM duration boundary

- [x] `forward()` only permits duration `d=1` at `t=0`.
- [x] The initial duration boundary is consistent with the existing Viterbi endpoint constraint `d <= t + 1`.

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

### Causal HSMM filter core

- [x] `HSMMFilterState` represents the normalized posterior over current latent state and current episode age as `[B,K,D]`.
- [x] Duration PMFs are converted to conditional end/continue hazards with impossible ages kept at zero probability rather than producing NaNs.
- [x] Continuation increments episode age without a transition.
- [x] Episode termination applies the duration hazard and only then the duration-conditioned transition matrix.
- [x] Self-transition starts a new episode of the same latent state with age reset to one.
- [x] Current state posterior and age posterior are directly available from the filter state.
- [x] One-step probability that the current episode ends before the next observation is available.
- [x] Deterministic-duration, boundary-transition, normalization, and numerical-support contracts were exercised locally on CPU.
- [x] Filter state rejects invalid rank, non-floating tensors, NaN/+inf, empty dimensions, and zero total mass.
- [x] Duration and transition inputs reject all-impossible probability rows rather than allowing NaN normalization.
- [x] Filter updates enforce dtype/device compatibility across posterior, emissions, durations, and transitions.

Commits:

- `5998c639408990075b1dc55e28f0c73a1a60f5d0`
- `d63981a8d17a551c54f127cb3bbdf7162138100e`
- `39e726c481c49c5fceb734597029d5e003eecd17`
- `3ef1d655dffdacdabe111c01cda79a1195e71983`

### Categorical rsample hardening

- [x] The missing functional symbol used by `Categorical.rsample()` is bound to the module's canonical `torch.nn.functional as nnF` import during `nhsmm.distributions` package initialization.
- [x] A current pytest-discoverable test covers soft Gumbel-Softmax output, gradient flow, and hard one-hot output.
- [x] The attempted whole-file Contents API rewrite was reverted to the exact original `default.py` blob after it introduced line-ending-only diff noise.

Commits:

- `0106c7e1f76b601281805c3c7bc5c03dd6d5e87e` — targeted test
- `eeebdceb8e1953df05546ec2240d89725238ddad` — restore exact original `default.py` blob after noisy rewrite
- `3a34c8f4e0f5adcc4cad0e12478aeb21902b6ebf` — functional alias binding

The runtime defect is resolved for canonical package imports. `default.py` still references `F` implicitly and therefore remains coupled to package initialization; remove that coupling later with a line-preserving direct patch rather than another full-file newline rewrite.

## Pre-test blockers

### P0 — causal HSMM filtering

- [x] Define an online filtering state over current latent state and current episode age without forcing the active segment to terminate at the current prefix boundary.
- [ ] Wire the filter core to NHSMM distribution/context outputs.
- [ ] Add an incremental `step()` or equivalent causal filtering API.
- [ ] Ensure repeated processing of the same observation/timestamp cannot advance filter state twice once timestamp semantics are introduced.
- [ ] Keep filtering distinct from retrospective smoothing and Viterbi decoding at the public API boundary.

The existing sequence `forward()` remains an endpoint/segment recursion. The causal filter is a separate recursion and must remain semantically distinct.

### P0 — temporal outputs

- [x] Expose current latent-state posterior at filter-core level.
- [x] Expose posterior state-age at filter-core level.
- [x] Expose one-step active-episode end probability at filter-core level.
- [ ] Expose survival probabilities for configurable future horizons.
- [ ] Expose change hazard / probability that the current episode ends within configurable future horizons.
- [ ] Expose transition probabilities needed for downstream evaluation through the NHSMM runtime API.
- [ ] Define expected remaining duration if it is mathematically well-defined under the selected duration contract.

These outputs must not require semantic state labels.

### P0 — causal invariants

- [x] Core filter posterior normalizes after repeated updates.
- [x] Core duration support handles deterministic/truncated support without NaN propagation.
- [x] Core filtering rejects malformed/non-finite probability rows before they can contaminate posterior state.
- [ ] Prefix invariance: full NHSMM filtered output at `t` must be unchanged when observations after `t` are modified.
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

## Repository scan findings

These items are verified repository state but are not evidence of NHSMM model quality.

### Canonical source surface

Current package core on `develop`:

- `nhsmm/config.py`
- `nhsmm/context.py`
- `nhsmm/convergence.py`
- `nhsmm/data.py`
- `nhsmm/encoder.py`
- `nhsmm/filtering.py`
- `nhsmm/distributions/default.py`
- `nhsmm/models/base.py`

`nhsmm/__init__.py` currently exports `ModelConfig`, `DistributionSet`, `DefaultEncoder`, `Convergence`, and `NHSMM`. The filtering helpers are currently internal/module-level and are not exported through the package root.

### Test discovery drift

Most historical/current test files are named without the normal `pytest` `test_*.py` / `*_test.py` pattern, including `tests/general.py` and `tests/filtering.py`.

Consequences:

- `python tests/general.py` and `python tests/filtering.py` are valid explicit smoke executions;
- targeted `python -m pytest tests/filtering.py` is valid and was used for the filter hardening run;
- `tests/test_categorical_rsample.py` is now collected by standard pytest discovery;
- a plain `pytest -v` still must not be assumed to represent the intended current suite because most canonical/legacy files remain outside standard discovery;
- test-file naming/discovery must be cleaned before `pytest -v` can be treated as the canonical full-suite command.

Historical test drift remains. For example, `tests/neural.py` imports removed modules/types such as `nhsmm.models.neural.NeuralHSMM`, `NHSMMConfig`, and `nhsmm.defaults.DTYPE`; `tests/ctx_encoder.py` also uses stale constructor arguments. These files must be migrated or removed rather than driving the canonical API backward.

### Script drift

`scripts/tune.py` still targets historical APIs including `NeuralHSMM`, `NHSMMConfig`, `nhsmm.utilities`, and `nhsmm.defaults`, and contains best-permutation/Hungarian state accuracy logic. `scripts/tune_gaussian.py` and other historical consumers require the same current-contract review before use.

Do not use these scripts for Nautilus evaluation until they are migrated to the canonical `ModelConfig` + `NHSMM` API and causal evaluation semantics.

### Distribution hardening status

The `Categorical.rsample()` NameError path found in the repository scan is operationally fixed for canonical package imports and covered by `tests/test_categorical_rsample.py`.

Remaining cleanup: `nhsmm/distributions/default.py` still calls `F.gumbel_softmax(...)` while importing the functional namespace as `nnF`; package initialization currently binds `F = nnF` in that module. This is safe for normal package imports but is non-local coupling. Replace the call directly with `nnF.gumbel_softmax(...)` once a line-preserving patch path is available.

### Packaging/status drift

`README.md` and project guidance describe NHSMM as pre-1.0 research-stage, while `pyproject.toml` still declares `Development Status :: 4 - Beta`.

`pyproject.toml` also uses broad lower-bounded dependencies such as `torch>=2.2`; the first GitHub smoke therefore installed the current CUDA-enabled PyTorch distribution and large CUDA dependency set even though the smoke explicitly ran the model on CPU. This is operationally expensive for frequent CI and should not be repeated for routine local checks.

### Workflow state

`.github/workflows/smoke.yml` remains in the repository from the first clean-run smoke. It supports manual `workflow_dispatch`; its push trigger is restricted to changes to the workflow file itself and the job additionally requires the bootstrap commit message, so ordinary `develop` source commits do not run it automatically.

Use local ChatGPT/container tests for routine small changes. Keep the workflow only as a clean-runner/integration diagnostic unless it no longer provides useful additional evidence.

## Verification

### General functional smoke

`tests/general.py` checks:

- explicit CPU device ownership;
- causal encoder is unidirectional;
- causal convolution preserves sequence length without right/future padding;
- causal prefix outputs do not change when only future observations change;
- HSMM `forward()` permits only duration `1` at `t=0`.

Observed GitHub Actions smoke run: `36234225103` — success.

### Causal filtering contracts

`tests/filtering.py` now checks:

- deterministic duration hazard conversion;
- episode-age progression and reset;
- transitions only at episode boundaries;
- repeated posterior normalization;
- rejection of zero-mass duration rows;
- rejection of zero-mass transition rows;
- dtype compatibility enforcement;
- rejection of NaN filter state.

Observed local ChatGPT runtime execution after hardening: `8 passed` on CPU. No GitHub Actions run was used for this slice.

### Categorical rsample contract

`tests/test_categorical_rsample.py` checks:

- soft Gumbel-Softmax output shape and normalization;
- finite gradient flow to logits;
- hard Gumbel-Softmax one-hot output.

Observed local ChatGPT runtime execution of the same package-import path: `1 passed` on CPU. No GitHub Actions run was used.

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

## Next slices

1. Wire `nhsmm/filtering.py` to the current NHSMM initial/duration/transition/emission/context outputs.
2. Add a causal incremental runtime API without reusing retrospective `forward()` semantics.
3. Add multi-horizon survival/change-hazard outputs and streaming-equivalence tests.
4. Clean test discovery so the intended current tests are collected by standard `pytest` commands.
5. Migrate or delete stale legacy tests/scripts once the canonical runtime path is stable.
6. Implement artifact/version/inference contracts before production Nautilus integration.
7. Remove the remaining non-local `Categorical.rsample()` alias coupling with a line-preserving direct patch.

## Later work

After causal correctness and the first research comparison are established:

- optimize online filtering to bounded `O(K * D)`-class state where practical;
- benchmark CPU inference latency and allocations;
- harden serialization/runtime compatibility;
- update public documentation from measured behavior rather than capability claims.
