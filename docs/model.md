# NHSMM model and runtime contract

This document describes the core model/runtime boundary of `nhsmm`.

Domain/framework integration belongs in the maintained host repository. `awa-si/nhsmm-interfaces` is deprecated and retained as public reference/history only.

## Model

The NHSMM separates four probabilistic components:

1. initial-state distribution;
2. transition distribution;
3. duration distribution;
4. emission distribution.

An optional, explicitly enabled context encoder can condition these components on sequence context. The default model has no internal encoder.

```text
observations
    |
optional context encoder
    |
context
    |
initial / transition / duration / emission
    |
HSMM inference
```

## Configuration

`ModelConfig` is the main configuration contract.

Typical construction:

```python
from nhsmm import ModelConfig, NHSMM

config = ModelConfig(
    n_states=3,
    n_features=5,
    max_duration=35,
    causal=True,
)

model = NHSMM(config=config)
model.initialize_distributions()

# use_context_encoder=False by default
```

`NHSMM` owns a resolved copy of the supplied `ModelConfig`; constructing a model, including construction with an explicitly injected encoder, does not mutate the caller's reusable config object.

A model intended for streaming use must be causal.

## Duration-tail training

The optional D+ tail stores its per-state end hazard directly as a probability. Joint optimization projects this parameter into [EPS, 1] after each optimizer update, before restart scoring and snapshots. This prevents an update beyond the feasible domain from entering the zero-gradient region of the inference clamp. The exact h_tail=1 finite-support boundary and artifact parameter representation are preserved. The inference clamp remains a guard for externally supplied parameters.

## Public runtime API

The supported streaming types are exported from the package root:

```python
from nhsmm import (
    HSMMFilterRuntime,
    HSMMFilterState,
    HSMMRuntimeState,
)
```

Avoid importing runtime/filter internals when the package-root contract is sufficient.

## HSMMFilterState

`HSMMFilterState.log_posterior` has shape:

```text
[B, K, D]
```

where:

- `B` = batch size;
- `K` = latent states;
- `D` = episode-age/duration support.

Semantics:

```text
log_posterior[b, k, a]
    = log P(z_t = k, age_t = a + 1 | x_0:t)
```

Convenience properties:

- `state_log_posterior` -> `[B,K]`;
- `state_posterior` -> `[B,K]`;
- `age_posterior` -> `[B,D]`.

The age posterior is not a duration forecast.

## HSMMFilterRuntime

`HSMMFilterRuntime` provides stateful causal filtering.

```python
runtime = HSMMFilterRuntime(model)

state = runtime.step(
    observation,
    context=None,
    timestamp=None,
)
```

`step()` returns an `HSMMFilterState`.

Accepted one-step observation shapes are:

```text
[F]
[B,F]
[B,1,F]
```

The feature dimension must equal `model.config.n_features`.

## Runtime invariants

### Causal model

The runtime requires `ModelConfig.causal=True`.

### Eval mode

The model must be in eval mode before causal filtering.

### Initialized distributions

Model distributions must already be initialized/loaded.

### Fixed batch size

After the first step, streaming batch size cannot change without `runtime.reset()`.

### Timestamp mode

A runtime session is either timestamped or untimestamped.

If the first step includes a timestamp:

- all later steps must include timestamps;
- timestamps must be strictly increasing.

If the first step omits a timestamp, timestamps cannot be enabled later without reset.

### Context mode

A runtime session uses either:

- the model's no-external-context path, which may be encoderless or use an opted-in internal encoder; or
- external caller-provided context.

The external-vs-non-external mode is fixed on the first successful step and cannot change until `runtime.reset()`. A rejected initialization step leaves the mode unset.

## Causal Float32 recursion

The causal batch forward recursion propagates a normalized `(state, age)` frontier at each timestep and carries cumulative log evidence separately. This preserves the public unnormalized `alpha`/log-likelihood semantics while avoiding repeated propagation of a growing sequence-level log offset in Float32. Model-bound filtering and runtime use the same normalized recursion semantics; the public `filter_step()` still validates and normalizes arbitrary caller-provided scores.

## No internal context — default

With `use_context_encoder=False` and no external context, observations are scored directly by the emission model and initial/duration/transition use their context-free parameters. No `ContextEncoder` is constructed. The canonical causal runtime remains bounded and retains only the latest accepted observation plus HSMM filter state.

## Internal context — opt-in

Set `use_context_encoder=True` in `ModelConfig` when learned observation-derived context is required.

With no external context:

```python
state = runtime.step(observation)
```

the causal model derives context through its configured encoder/runtime path. The initial prior uses zero context; emissions use the previous encoder row. Boundary duration/transition propagation preserves its existing previous-row timing. Batch and incremental streaming use the same information contract.

When the configured encoder exposes incremental streaming state, the runtime keeps bounded encoder state. Custom causal encoders without that contract may use the correctness-first retained-prefix fallback.

## Predictive neural likelihood

For causal internal context, c_t summarizes observations through t. The initial-state prior uses c_-1 = 0, independent of the first observation. Emission parameters for x_t use c_(t-1), excluding x_t. Boundary propagation from t-1 to t also uses c_(t-1); duration/transition contexts are not shifted again. Runtime stores c_t after accepting x_t for the next step. Custom causal encoder prefix fallback follows the same batch timing.

This removes emission self-conditioning. The former normalization counterexample now integrates to 1.0 on both maintained quadrature grids. This is bounded regression evidence, not proof for arbitrary custom distributions. External context is conditional on covariates held fixed while observations vary; their information content is the caller's responsibility.

Noncausal internal encoders retain retrospective forward/decode scores. Public generative log_likelihood and optimize reject them when no external context is supplied: lagging a bidirectional encoder does not remove future dependence. Encoderless noncausal models remain supported.

Artifacts use version 2 and declare internal_context_timing as predictive_past, retrospective_scores, or none. Version 1 encoder artifacts are rejected with retraining guidance because old observation-row timing cannot be silently reinterpreted. Version 1 encoderless artifacts remain loadable. Raw state_dict files carry no timing version: callers must establish provenance and retrain old encoder weights before predictive use. See [repair evidence](validation.md#predictive-neural-repair--2026-10-12).

## External context

External context is supplied explicitly:

```python
state = runtime.step(
    observation,
    context=context,
)
```

External context requires `ModelConfig.context_dim`; its dimensionality must equal `model.context_dim`. An internal encoder is not required. Variable-length external-context lists are validated against `context_dim` independently of the observation feature width.

External context remains the caller's responsibility for every step in that runtime session.

## Runtime reset

```python
runtime.reset()
```

Reset clears:

- filter/runtime state;
- context-mode lock;
- timestamp/session history.

It does not reinitialize model parameters.

Use reset when intentionally starting a new independent stream or changing runtime mode.

## Forecasts

After at least one observation has been processed:

```python
survival = runtime.forecast_survival(horizons)
transition = runtime.forecast_transition()
```

These forecasts use only the runtime's current information set.

They are separate from `HSMMFilterState.age_posterior`.

## Causal semantics

The streaming runtime is a filtering path, not a retrospective smoother.

Future observations are not consumed.

The filter tracks the posterior over current latent state and current episode age. State propagation follows explicit-duration HSMM boundary semantics; the current observation enters through the causal emission/context path for the current step.

Retrospective/non-causal model methods are distinct APIs and should not be interpreted as live estimates.

## Artifacts

Artifact helpers are exported publicly:

```python
from nhsmm import (
    build_artifact,
    load_artifact,
    save_artifact,
)
```

Inference helpers:

```python
from nhsmm import (
    load_inference_model,
    prepare_inference,
)
```

A loaded model intended for streaming must satisfy the same causal/runtime invariants.

`build_artifact()` returns an independent CPU tensor snapshot. Subsequent model parameter updates do not mutate the built artifact.

## Integration boundary

The NHSMM core intentionally stops at model/runtime outputs.

Framework/domain mapping belongs in the maintained host repository.

```text
external event
    |
host adapter
    |
nhsmm public runtime API
    |
HSMMFilterRuntime / HSMMFilterState
    |
downstream representation
```

The dependency direction is one-way: host adapters may depend on the public `nhsmm` package API; `nhsmm` must not depend on host/framework packages. For NautilusTrader, the active adapter owner is `awa-si/nautilus@main/adapter/nhsmm`.

Microsoft Qlib research mapping is implemented in the maintained host's
[`adapter/qlib`](https://github.com/awa-si/nautilus/tree/dev/qlib-nhsmm/adapter/qlib)
development branch. It consumes trusted causal artifacts and the public streaming
runtime, isolates instrument state, and builds real Qlib datasets with explicit
label-maturity purging. Fit-cutoff/observation-schema provenance and walk-forward
state alignment remain caller obligations. Qlib is an optional host dependency;
it does not replace NHSMM inference or become a model-core dependency.

The deprecated [awa-si/nhsmm-interfaces](https://github.com/awa-si/nhsmm-interfaces) repository remains available as public reference/history for earlier adapter and walk-forward contracts.

The core repository should not absorb:

- Nautilus/Freqtrade strategy semantics;
- trading execution/risk policy;
- research workflow/business policy;
- medical decisions;
- raw-document ingestion/OCR pipelines.

## Validation boundary

Context-effect validation remains separate from model training and runtime integration.

See [validation.md](validation.md).

## Related documentation

- [README](../README.md)
- [Validation API](validation.md)
- [Testing](testing.md)
- [Development state](state.md)
- [Release](release.md)
- [Deprecated NHSMM Interfaces reference](https://github.com/awa-si/nhsmm-interfaces)

## Distribution numerical boundaries

Continuous Gaussian/Student-t locations are finite real values, not categorical logits; no categorical score cap is applied to them. Independent Student-t parameters are broadcast before statistics and event reductions. Categorical labels broadcast across sample/batch dimensions; exact zero probabilities retain finite gradients for observed positive-probability classes.

Initial, duration and transition hard masks use exact negative infinity. Categorical logit construction accepts partial impossible support but rejects NaN, positive infinity and rows with no finite mass. Safe survival and sum reductions preserve impossible mass without undefined all-impossible backward paths. An impossible observed event can still have negative-infinite likelihood; numerical safety does not make it possible.

For duration soft_dmax, log_matrix remains the authoritative gated-model path. Generic forward/expected_probs gate semantics are unresolved (audit D07); do not substitute those helpers for gated model duration probabilities.

See `docs/validation/distribution-fixes-20261015.md` for regression coverage and explicit historical-suite limits.
