# NHSMM — Neural Hidden Semi-Markov Models

PyTorch implementation of a context-aware hidden semi-Markov model with explicit state durations.

> **Status:** pre-1.0 research/development package. Public APIs and internal contracts may still change.

[![PyPI](https://img.shields.io/pypi/v/nhsmm.svg)](https://pypi.org/project/nhsmm/) [![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0) [![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)

## Scope

The model separates four probabilistic components:

- initial-state distribution;
- transition distribution;
- duration distribution;
- emission distribution.

An optional neural context encoder can condition these components on sequence context.

Current configured emission families are Gaussian and Student-t. Current transition modes are `ergodic`, `semi`, and `left-to-right`.

## Installation

```bash
pip install nhsmm
```

Optional dataframe helpers use Polars and can be installed with:

```bash
pip install "nhsmm[data]"
```

For development:

```bash
git clone https://github.com/awa-si/nhsmm.git
cd nhsmm
pip install -e ".[dev]"
```

Python `3.12+` is required by the package metadata.

## Basic usage

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

`ModelConfig` is the main configuration contract. Historical constructor names or examples should not be treated as current API unless they are exported by the installed package.

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
likelihood / decoding / filtering / optimization
```

Main package areas:

```text
nhsmm/
├── config.py          # model configuration
├── context.py         # context routing and sequence containers
├── encoder.py         # neural context encoder
├── filtering.py       # filtering implementation
├── inference.py       # inference helpers
├── runtime.py         # incremental runtime
├── artifact.py        # model artifact IO
├── distributions/     # probabilistic components
├── models/            # model and training implementation
└── validation/        # context-effect validation API
```

## Causal and retrospective paths

`ModelConfig(causal=True)` enables the causal encoder/runtime path. The filter state is represented over `(latent_state, episode_age)` and does not consume future observations.

For causal filtering, information available at `F_t` determines the duration/end hazard for boundary `t -> t+1`.

Retrospective/non-causal inference is a separate path and should not be interpreted as an online filtered estimate.

See [`docs/model.md`](docs/model.md) for the model/runtime contract.

## Validation API

The public context-effect facade is component-based:

```python
from nhsmm import (
    context_effect,
    evaluate_context_effect_replication,
    split_fit_context_effect_evidence,
)
```

Supported component names:

```text
initial | duration | emission | transition
```

Validation is kept separate from model optimization. Evidence thresholds are supplied by the caller rather than embedded as universal defaults.

See [`docs/validation-api.md`](docs/validation-api.md).

## Development

Run the repository test/static-check set with:

```bash
pip install -e ".[dev]"
pytest -q
ruff check nhsmm tests scripts
black --check nhsmm tests scripts
```

Repository tests are under `tests/`. The current test and runtime policies are documented in [`docs/testing.md`](docs/testing.md).

Packaging/release notes are in [`docs/release.md`](docs/release.md).

## Documentation

- [`docs/model.md`](docs/model.md) — model/runtime contract
- [`docs/validation-api.md`](docs/validation-api.md) — validation API
- [`docs/testing.md`](docs/testing.md) — test and verification policy
- [`docs/package-validation.md`](docs/package-validation.md) — controlled package-level validation
- [`docs/state.md`](docs/state.md) — current development state
- [awa-si/nhsmm-interfaces](https://github.com/awa-si/nhsmm-interfaces) — external interface repository

Validation documents describe controlled package behavior only. They are not claims about downstream domain performance.

## License

Apache License 2.0 © AWA.SI. See [LICENSE](./LICENSE).
