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
- context-effect validation utilities;
- reproducible model-health and validation snapshots;
- strict configuration contracts and a reusable configuration tuner.

Domain/framework integration is intentionally kept outside this repository.

## Installation

```bash
pip install nhsmm
```

Development install:

```bash
git clone https://github.com/awa-si/nhsmm.git
cd nhsmm
pip install -e ".[dev]"
```

For CPU-only development, install through the repository bootstrap so pip does
not resolve accelerator packages that are unnecessary on a CPU host:

```bash
python -m venv .venv
. .venv/bin/activate
python scripts/install_cpu_dev.py
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

# ContextEncoder is opt-in; the default model has no internal encoder.
```

`ModelConfig` is the runtime/training configuration contract. Validation policy is
kept separately in `ValidationConfig`; systematic search is provided by
`ConfigTuner`. See [`docs/configuration.md`](docs/configuration.md).

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
├── config.py          # model, health, and validation configuration contracts
├── context.py         # context routing and sequence containers
├── encoder.py         # neural context encoder
├── filtering.py       # filter-state semantics
├── inference.py       # inference helpers
├── runtime.py         # incremental causal runtime
├── survival.py        # survival forecasts
├── transitions.py     # transition forecasts
├── artifact.py        # model artifact IO
├── diagnostics.py     # model-health diagnostics
├── tuning.py          # validated configuration search
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

NHSMM supports three runtime context paths. `ContextEncoder` is opt-in.

### No context — default

`ModelConfig.use_context_encoder` defaults to `False`. No internal encoder is constructed; the probabilistic HSMM runs directly on its base distribution parameters and emission likelihoods. This is the smallest model surface for direct math, distribution, training, and runtime tests.

### Internal context — opt-in

Enable learned observation-derived context explicitly with `use_context_encoder=True`. Passing an explicit `encoder=` to `NHSMM(...)` is also treated as an opt-in and is recorded in the model config.

The configured causal encoder derives context from observations.

```python
runtime.step(observation)
```

### External context

The caller can supply context explicitly without enabling the internal encoder. Configure `context_dim` to the external context width.

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

### Cross-repository contract

`awa-si/nhsmm` is the source of truth for model/runtime behavior, artifacts, filtering, forecasting, and context-effect validation. `awa-si/nhsmm-interfaces` is the source of truth for host/framework mapping, adapter lifecycle, and canonical `Observation`/`Context`/`StateEstimate` contracts.

The interface repository must consume this package through public exports such as `HSMMFilterRuntime`, `load_artifact`, and the documented validation surface; it must not depend on model internals. Conversely, framework-specific adapter code does not belong in this core repository.

## Causal and retrospective paths

`ModelConfig(causal=True)` enables causal HSMM/runtime semantics; add `use_context_encoder=True` only when learned internal context is required.

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

For reproducible Train/OOS evidence, the package also exposes
`evaluate_validation_snapshot(...)`, `compare_validation_snapshots(...)`, and
`evaluate_model_health(...)`.

See [`docs/validation.md`](docs/validation.md).

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

CPU-only hosts should use `python scripts/install_cpu_dev.py` inside an
activated virtual environment before running the same checks.

Tests live under `tests/`.

## Documentation

- [`docs/model.md`](docs/model.md) — model and runtime contract
- [`docs/configuration.md`](docs/configuration.md) — configuration contracts
- [`docs/validation.md`](docs/validation.md) — validation API
- [`docs/testing.md`](docs/testing.md) — verification policy
- [`docs/state.md`](docs/state.md) — development-state record
- [`docs/release.md`](docs/release.md) — packaging/release process
- [`awa-si/nhsmm-interfaces`](https://github.com/awa-si/nhsmm-interfaces) — integration contracts and adapters

Validation documents describe controlled package behavior, not downstream domain performance.

## License

Apache License 2.0 © AWA.SI. See [LICENSE](./LICENSE).
