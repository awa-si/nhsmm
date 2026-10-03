# NHSMM — Neural Hidden Semi-Markov Models

PyTorch implementation of context-aware hidden semi-Markov models with explicit state durations.

> **Status:** pre-1.0 research/development package. Public APIs and contracts may still change.

[![PyPI](https://img.shields.io/pypi/v/nhsmm.svg)](https://pypi.org/project/nhsmm/) [![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0) [![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)

## Scope

NHSMM provides the model/core runtime layer:

- explicit-duration hidden semi-Markov inference;
- initial, transition, duration, and emission distributions;
- optional neural context encoding;
- causal streaming filtering;
- retrospective/batch inference;
- model artifacts and inference loading;
- context-effect validation utilities.

Domain/framework integration is intentionally kept outside this repository.

## Installation

```bash
pip install nhsmm
```

Optional dataframe helpers:

```bash
pip install "nhsmm[data]"
```

Development install:

```bash
git clone https://github.com/awa-si/nhsmm.git
cd nhsmm
pip install -e ".[dev]"
```

Python 3.12+ is required by the current package metadata.

## Basic model usage

```python
from nhsmm import ModelConfig, NHSMM

config = ModelConfig(
    n_states=3,
    n_features=5,
    max_duration=35,
)

model = NHSMM(config=config)
model.initialize_distributions()
```

`ModelConfig` is the main configuration contract.

## Architecture

```text
observations
    |
context encoder (optional)
    |
initial / transition / duration / emission distributions
    |
HSMM inference
    |
batch inference / causal filtering / forecasts
```

Main package areas:

```text
nhsmm/
├── config.py          # model configuration
├── context.py         # context routing and sequence containers
├── encoder.py         # neural context encoder
├── filtering.py       # filter-state semantics
├── inference.py       # inference helpers
├── runtime.py         # incremental causal runtime
├── survival.py        # survival forecasts
├── transitions.py     # transition forecasts
├── artifact.py        # model artifact IO
├── distributions/     # probabilistic components
├── models/            # model and training implementation
└── validation/        # context-effect validation API
```

## Public streaming runtime

The public runtime surface is exported directly from `nhsmm`:

```python
from nhsmm import (
    HSMMFilterRuntime,
    HSMMFilterState,
    HSMMRuntimeState,
)
```

Minimal streaming flow:

```python
runtime = HSMMFilterRuntime(model)

filter_state = runtime.step(
    observation,
    context=optional_context,
    timestamp=optional_timestamp,
)

state_posterior = filter_state.state_posterior
age_posterior = filter_state.age_posterior
```

Runtime rules:

- the model must be causal and in eval mode;
- one `step()` represents one streaming timestep;
- timestamp mode is fixed after the first step until `reset()`;
- internal-vs-external context mode is fixed after the first step until `reset()`;
- streaming batch size cannot change without reset;
- runtime state is separate from model parameters.

The runtime also exposes current-information survival and transition forecasts after at least one observation has been processed.

See [`docs/model.md`](docs/model.md) for the detailed model/runtime contract.

## Context modes

NHSMM supports two runtime context paths.

### Internal context

The configured causal encoder derives context from observations.

```python
runtime.step(observation)
```

### External context

The caller supplies context explicitly.

```python
runtime.step(
    observation,
    context=context,
)
```

A runtime session must remain in the selected mode until reset.

## Interfaces and adapters

Domain-facing integration is maintained in the separate repository:

**[awa-si/nhsmm-interfaces](https://github.com/awa-si/nhsmm-interfaces)**

The repository boundary is:

```text
host / domain system
        |
        v
nhsmm-interfaces
        |
        v
nhsmm public runtime API
        |
        v
NHSMM core
```

`nhsmm-interfaces` provides:

- canonical `Observation`, `Context`, and `StateEstimate` contracts;
- `Adapter`;
- `NHSMMRuntimeAdapter`;
- domain/framework adapters such as `StructuredEventAdapter`;
- integration guidance for systems such as Nautilus Trader and Freqtrade.

The core package does not contain trading policy, medical/research workflow policy, execution logic, or other domain decisions.

## Causal and retrospective paths

`ModelConfig(causal=True)` enables the causal encoder/runtime path.

The causal filter state is represented over latent state and episode age:

```text
P(z_t, age_t | x_0:t)
```

For causal filtering, information available at the current information boundary determines propagation; future observations are not consumed.

Retrospective/non-causal inference is a separate path and must not be interpreted as an online filtered estimate.

See [`docs/model.md`](docs/model.md).

## Validation API

The public context-effect facade is component-based:

```python
from nhsmm import (
    context_effect,
    evaluate_context_effect_replication,
    split_fit_context_effect_evidence,
)
```

Supported components:

```text
initial | duration | emission | transition
```

Validation is separate from model optimization. Evidence thresholds are caller policy rather than universal package defaults.

See [`docs/validation-api.md`](docs/validation-api.md).

## Artifacts and inference

Public artifact/inference helpers include:

```python
from nhsmm import (
    build_artifact,
    load_artifact,
    save_artifact,
    load_inference_model,
    prepare_inference,
)
```

Artifacts and runtime integration should use public exports rather than internal modules where a public contract exists.

## Development

```bash
pip install -e ".[dev]"
pytest -q
ruff check nhsmm tests scripts
black --check nhsmm tests scripts
```

Tests live under `tests/`.

## Documentation

- [`docs/model.md`](docs/model.md) — model and runtime contract
- [`docs/validation-api.md`](docs/validation-api.md) — validation API
- [`docs/testing.md`](docs/testing.md) — verification policy
- [`docs/package-validation.md`](docs/package-validation.md) — controlled package-level validation
- [`docs/state.md`](docs/state.md) — development-state record
- [`docs/release.md`](docs/release.md) — packaging/release process
- [`awa-si/nhsmm-interfaces`](https://github.com/awa-si/nhsmm-interfaces) — integration contracts and adapters

Validation documents describe controlled package behavior, not downstream domain performance.

## License

Apache License 2.0 © AWA.SI. See [LICENSE](./LICENSE).
