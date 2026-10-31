---
title: NHSMM — Neural Hidden Semi-Markov Models
description: Package overview, installation, APIs, and links to canonical model documentation.
type: guide
status: active
language: en
owner: NHSMM
created: 2025-10-10
updated: 2026-10-31
---

# NHSMM — Neural Hidden Semi-Markov Models

PyTorch implementation of context-aware hidden semi-Markov models with explicit state durations.

> **Status:** pre-1.0 research/development package. Public APIs and contracts may still change.

[![PyPI](https://img.shields.io/pypi/v/nhsmm.svg)](https://pypi.org/project/nhsmm/) [![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0) [![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)

## Scope

Default and supported development branch: v3.

NHSMM provides the model/core runtime layer:

- explicit-duration hidden semi-Markov inference;
- initial, transition, duration, and emission distributions;
- optional neural context encoding;
- causal streaming filtering;
- retrospective/batch inference;
- model artifacts and inference loading;
- strict configuration contracts.

Domain/framework integration is intentionally kept outside this repository.

## Installation

```bash
pip install nhsmm
```

Development install:

```bash
git clone --branch v3 https://github.com/awa-si/nhsmm.git
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
# Train dense/ragged sequences through the public initialization + optimizer path:
# model.fit(sequences)

# ContextEncoder is opt-in; the default model has no internal encoder.
```

`ModelConfig` is the runtime/training configuration contract. Validation policy is
owned by the host ML stack. Use `ModelConfig.with_overrides` for strict
model updates. Health/context validation policy now lives in Nautilus
adapter.qlib.nhsmm_validation. Experiment candidates, training tasks and recording
belong in the optional Nautilus Qlib host. See
[`docs/configuration.md`](docs/configuration.md).

See [the compact model lifecycle](docs/api.md) for fit, artifacts and causal streaming.

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
├── config.py          # model configuration contract
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

The dependency boundary remains:

```text
host / domain adapter
        |
        v
nhsmm public runtime API
        |
        v
NHSMM core
```

For NautilusTrader, the active integration owner is `awa-si/nautilus@main/adapter/nhsmm`.
Other hosts should likewise keep framework-specific mapping and lifecycle code outside this core package.

The core package does not contain trading policy, medical/research workflow policy, execution logic, or other domain decisions.

### Cross-repository contract

Host adapters must consume this package through public exports such as `HSMMFilterRuntime`, `load_artifact`, and ModelConfig; they must not depend on model internals. Conversely, framework-specific adapter code does not belong in this core repository.

## Causal and retrospective paths

`ModelConfig(causal=True)` enables causal HSMM/runtime semantics; add `use_context_encoder=True` only when learned internal context is required.

The causal filter state is represented over latent state and episode age:

```text
P(z_t, age_t | x_0:t)
```

For causal filtering, information available at the current information boundary determines propagation; future observations are not consumed.

Retrospective/non-causal inference is a separate path and must not be interpreted as an online filtered estimate.

See [`docs/model.md`](docs/model.md).

## Host validation API

The public context-effect facade is component-based:

```python
from adapter.qlib.nhsmm_validation import (
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

For reproducible Train/OOS evidence, the host also exposes
`evaluate_validation_snapshot(...)`, `compare_validation_snapshots(...)`, and
`evaluate_model_health(...)`.

See [`docs/validation.md`](https://github.com/awa-si/nautilus/tree/main/docs/ml-stack/nhsmm/evaluation.md).

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
ruff check nhsmm scripts
black --check nhsmm scripts
```

CPU-only hosts should use `python scripts/install_cpu_dev.py` inside an
activated virtual environment before running the same checks.

Model and mathematics tests are maintained in the [Nautilus ML stack](https://github.com/awa-si/nautilus/tree/main/tests/ml_stack/nhsmm). Run them from that checkout with this source installed editable; see [docs/testing.md](docs/testing.md).

## Documentation

- [`docs/model.md`](docs/model.md) — model and runtime contract
- [`docs/configuration.md`](docs/configuration.md) — configuration contracts
- [`docs/validation.md`](https://github.com/awa-si/nautilus/tree/main/docs/ml-stack/nhsmm/evaluation.md) — validation API
- [`docs/testing.md`](docs/testing.md) — verification policy
- [`docs/release.md`](docs/release.md) — packaging/release process
- [Nautilus ML stack](https://github.com/awa-si/nautilus/tree/main/adapter/qlib) — active research and integration

Validation documents describe controlled package behavior, not downstream domain performance.

## License

Apache License 2.0 © AWA.SI. See [LICENSE](./LICENSE).


Mathematical repair: noncausal public likelihoods now normalize complete-segmentation potentials; raw forward and MAP retain their score contract. Artifact v3 records this distinction and preserves floating precision. See [the current mathematical contract](docs/hsmm-mathematical-contract.md) and [repair evidence](docs/correctness-repair.md). Canonical model tests remain in Nautilus.

## Supervised state paths

The causal core supports observed state-path training and an ordinary HMM
specialization. See [supervised semantics and examples](docs/supervised.md).

## Explicit HMM / HSMM / NHSMM selection

Use ModelConfig.for_mode(mode, n_states=3, n_features=3) with mode set to
hmm, hsmm or nhsmm. Static modes construct no observation encoder or neural
context heads. The same supervised fit, artifact and streaming APIs apply.
See [mode contracts and examples](docs/model-modes.md).
