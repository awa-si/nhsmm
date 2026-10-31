---
title: NHSMM — Neural Hidden Semi-Markov Models
description: Package overview, installation, public entry points, and canonical documentation.
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

The supported development branch is `v3`. NHSMM owns model math, distributions,
initialization and optimization, causal filtering and forecasts, configuration,
and core artifacts. Framework integrations, validation policy, datasets and
research workflows belong to the [Nautilus ML stack](https://github.com/awa-si/nautilus/tree/main/adapter/qlib).

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

For CPU-only development, use `python scripts/install_cpu_dev.py` inside an
activated virtual environment.

## Quick start

```python
from nhsmm import ModelConfig, NHSMM

model = NHSMM(ModelConfig(
    n_states=3,
    n_features=5,
    max_duration=35,
))
```

Use [the compact lifecycle](docs/api.md) for fitting, artifacts and causal
streaming. Context, model modes, supervised state paths and mathematical
semantics are documented in their canonical owners below.

## Documentation

- [Model and runtime contract](docs/model.md)
- [HSMM mathematical contract](docs/hsmm-mathematical-contract.md)
- [Bounded duration-tail contract](docs/duration-tail-contract.md)
- [Configuration](docs/configuration.md)
- [HMM, HSMM and NHSMM modes](docs/model-modes.md)
- [Supervised state paths](docs/supervised.md)
- [Verification policy and source-bound evidence](docs/testing.md)
- [CPU qualification and operational limits](docs/production.md)
- [Release procedure](docs/release.md)
- [Research hypothesis and evidence boundary](hypothesis.md)
- [Nautilus validation API](https://github.com/awa-si/nautilus/tree/main/docs/ml-stack/nhsmm/evaluation.md)

Validation documents describe controlled package behavior, not downstream
trading performance.

## License

Apache License 2.0 © AWA.SI. See [LICENSE](./LICENSE).
