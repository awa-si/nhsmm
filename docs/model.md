# NHSMM model and runtime contract

This document describes the core model/runtime boundary of `nhsmm`.

Domain/framework integration belongs in [awa-si/nhsmm-interfaces](https://github.com/awa-si/nhsmm-interfaces).

## Model

The NHSMM separates four probabilistic components:

1. initial-state distribution;
2. transition distribution;
3. duration distribution;
4. emission distribution.

An optional context encoder can condition these components on sequence context.

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
```

A model intended for streaming use must be causal.

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

- internal model context; or
- external caller-provided context.

The mode is fixed on the first step and cannot change until `runtime.reset()`.

## Internal context

With no external context:

```python
state = runtime.step(observation)
```

the causal model derives context through its configured encoder/runtime path.

When the configured encoder exposes incremental streaming state, the runtime keeps bounded encoder state. Custom causal encoders without that contract may use the correctness-first retained-prefix fallback.

## External context

External context is supplied explicitly:

```python
state = runtime.step(
    observation,
    context=context,
)
```

Context dimensionality must equal `model.context_dim`.

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

## Integration boundary

The NHSMM core intentionally stops at model/runtime outputs.

Framework/domain mapping belongs in:

[awa-si/nhsmm-interfaces](https://github.com/awa-si/nhsmm-interfaces)

```text
external event
    |
nhsmm-interfaces adapter
    |
Observation / Context
    |
HSMMFilterRuntime
    |
HSMMFilterState
    |
StateEstimate / downstream representation
```

The dependency direction is one-way: `nhsmm-interfaces` may depend on the public `nhsmm` package API; `nhsmm` must not depend on `nhsmm-interfaces` or any framework package.

The interface repository owns host/domain contracts such as:

- `Adapter`;
- `NHSMMRuntimeAdapter`;
- `StructuredEventAdapter`;
- framework-specific adapters.

The core repository should not absorb:

- Nautilus/Freqtrade strategy semantics;
- trading execution/risk policy;
- research workflow/business policy;
- medical decisions;
- raw-document ingestion/OCR pipelines.

## Validation boundary

Context-effect validation remains separate from model training and runtime integration.

See [validation-api.md](validation-api.md).

## Related documentation

- [README](../README.md)
- [Validation API](validation-api.md)
- [Testing](testing.md)
- [Package validation](package-validation.md)
- [Development state](state.md)
- [Release](release.md)
- [NHSMM Interfaces](https://github.com/awa-si/nhsmm-interfaces)
