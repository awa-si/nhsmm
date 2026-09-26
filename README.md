# NHSMM — Neural Hidden Semi-Markov Models

- **Repository:** [awa-si/nhsmm](https://github.com/awa-si/nhsmm)
- **Interfaces:** [awa-si/nhsmm-interfaces](https://github.com/awa-si/nhsmm-interfaces)
- **Documentation:** [NHSMM Wiki](https://github.com/awa-si/nhsmm/wiki)
- **Article:** [Unlocking Hidden Patterns in Time – Meet NHSMM](https://medium.com/@awa-si/unlocking-hidden-patterns-in-time-meet-nhsmm-the-neural-hidden-semi-markov-model-cd3f1e2428c2)

> **Pre-1.0 research stage.** NHSMM is actively evolving. Public APIs and internal model contracts may change before a stable `1.0.0` release.

NHSMM is a modular PyTorch library for context-aware latent-state sequence modeling with explicit state durations.

The current implementation separates four probabilistic components:

- initial state distribution;
- transition distribution;
- duration distribution;
- emission distribution.

A neural context encoder can condition these components on sequence context.

[![PyPI](https://img.shields.io/pypi/v/nhsmm.svg)](https://pypi.org/project/nhsmm/) [![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0) [![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)

## Current architecture

```text
Observed sequence
      ↓
Context encoder
      ↓
Initial / Transition / Duration / Emission parameterization
      ↓
HSMM inference
      ↓
Likelihood / decoding / training
```

Core implementation areas:

```text
nhsmm/
├── config.py          # ModelConfig and numerical constants
├── context.py         # context routing and sequence containers
├── encoder.py         # default neural context encoder
├── convergence.py     # convergence handling
├── data.py            # data utilities
├── distributions/     # initial, duration, transition, emission components
└── models/
    └── base.py        # NHSMM and DistributionSet
```

## Model configuration

`ModelConfig` is the canonical configuration contract.

Current configurable model dimensions include:

- `n_states` — latent-state count;
- `n_features` — observation feature count;
- `max_duration` — maximum explicit state duration;
- `context_dim` / `hidden_dim` — neural context dimensions;
- encoder pooling and convolution parameters;
- transition type;
- emission family;
- optimization and convergence parameters.

The current configured emission families are:

- Gaussian;
- Student-t.

The current configured transition modes are:

- `ergodic`;
- `semi`;
- `left-to-right`.

## Installation

From PyPI:

```bash
pip install nhsmm
```

For development:

```bash
git clone https://github.com/awa-si/nhsmm.git
cd nhsmm
pip install -e ".[dev]"
```

Python `3.9+` is required by the current package metadata.

## Basic construction

The current public API is configuration-driven:

```python
from nhsmm import ModelConfig, NHSMM

config = ModelConfig(
    n_states=3,
    n_features=5,
    max_duration=35,
)

model = NHSMM(config=config)
```

Do not rely on historical examples using `HSMM`, `NeuralHSMM`, `GaussianHSMM`, or older constructor signatures unless the current package exports explicitly provide them.

## Context-aware modeling

NHSMM can derive neural context from observed sequences and use it to parameterize latent-state distributions.

The implementation distinguishes:

- observed sequences;
- masks and sequence lengths;
- per-timestep context;
- canonical/global context;
- latent-state distribution parameters.

This separation is important for variable-length batching and for keeping the probabilistic model contract explicit.

## Temporal semantics

NHSMM supports retrospective sequence inference. When using NHSMM in forecasting, online inference, or trading systems, callers must preserve causal data flow themselves and distinguish filtering from full-sequence smoothing.

A result obtained using future observations must not be interpreted as a causal online state estimate.

## Development

Install development dependencies and run the relevant checks:

```bash
pip install -e ".[dev]"
pytest -v
ruff check nhsmm tests scripts
black --check nhsmm tests scripts
```

Some historical scripts and tests may target older APIs. Before changing the core model to satisfy one of those consumers, verify whether the consumer or the implementation reflects the intended current contract.

Repository control is split deliberately:

- [`agent.md`](./agent.md) defines AI behavior, reasoning, source resolution, and decision/completion gates for work in this repository;
- [`docs/agent-domain.md`](docs/agent-domain.md) owns the repository-wide probabilistic, tensor, causal, API-evolution, testing, and research contracts that the AI must load when material;
- [`docs/model.md`](docs/model.md) documents the model/project contract for human-facing repository guidance;
- the global Git/edit/CI workflow is owned by `awa-si/admin/workflow.md`.

`agent.md` is therefore not the substantive model specification. It routes AI work to the relevant contract owners.

## Documentation

- [AI-facing domain contract](docs/agent-domain.md)
- [Model / project governance](docs/model.md)
- [OHLCV experiment notes](docs/test_ohlcv.md)
- [Tier notes](docs/tier.md)

Documentation should describe implemented behavior. Performance, causal, production-readiness, or scalability claims require corresponding implementation or benchmark evidence.

## Project relationship

NHSMM is the open probabilistic-modeling foundation associated with the broader State Aware Engine (SAE) work. Domain/integration contracts are kept separate from the core model implementation.

## License

Apache License 2.0 © AWA.SI.

See [LICENSE](./LICENSE) for the full terms.
