# Contributing to NHSMM

Thank you for contributing to NHSMM (Neural Hidden Semi-Markov Models).

Contributions may include bug fixes, documentation, numerical-stability improvements, tests, performance work, examples, and research-driven extensions.

## Before you change code

Use the repository control layers according to their ownership:

1. Read [`agent.md`](./agent.md) for AI-facing behavior, reasoning, source resolution, and decision/completion gates.
2. Read [`docs/agent-domain.md`](docs/agent-domain.md) when the change touches probabilistic semantics, tensor contracts, causality, API evolution, testing, or research methodology.
3. Read [`docs/model.md`](docs/model.md) and the current implementation when the change depends on model/project behavior.
4. Apply the global repository workflow from `awa-si/admin/workflow.md` for editing, verification, CI, and write-back.

The current implementation and discovered `tests/test_*.py` suite are authoritative. Do not reintroduce historical constructors or compatibility paths solely to satisfy removed consumers.

`agent.md` is not a substitute for the substantive model/domain contracts; it routes AI work to the relevant owners.

## Getting started

```bash
git clone https://github.com/awa-si/nhsmm.git
cd nhsmm
pip install -e ".[dev]"
```

Create work from the current `develop` branch unless the repository workflow explicitly changes.

Keep changes focused and update dependent contracts together when behavior changes.

## Development checks

Run the smallest checks that materially validate your change, then expand where necessary.

```bash
pytest -q
ruff check nhsmm tests scripts
black --check nhsmm tests scripts
```

Test discovery and verification policy is documented in [`docs/testing.md`](docs/testing.md).

For shared inference, distribution, context, configuration, or training changes, run the broader relevant test set.

Never report a test, benchmark, training run, or empirical result as successful unless it was actually executed and observed.

## Model changes

NHSMM is an explicit-duration probabilistic sequence model. The canonical repository-wide invariants for model changes live in [`docs/agent-domain.md`](docs/agent-domain.md). Changes must preserve or intentionally redefine the affected mathematical contract.

Check, as applicable:

- tensor dimensions and broadcasting;
- normalization axes;
- initial, transition, duration, and emission semantics;
- duration boundaries and indexing;
- sequence masks and variable-length behavior;
- dtype/device handling;
- numerical stability in log space;
- gradient flow;
- causal versus retrospective inference semantics.

Do not add compatibility aliases or duplicate model paths merely to preserve stale internal consumers. Migrate consumers to one canonical API unless backward compatibility is explicitly required.

## Public API and configuration

The canonical model construction is configuration-driven:

```python
from nhsmm import ModelConfig, NHSMM

config = ModelConfig(n_states=3, n_features=5)
model = NHSMM(config=config)
```

When changing `ModelConfig` or public package exports, update dependent source, scripts, tests, examples, and documentation coherently.

## Documentation

Documentation must describe implemented behavior.

Do not publish unsupported claims such as `production-ready`, `causal`, `GPU optimized`, `memory-efficient`, or `scalable` without evidence appropriate to the claim.

Examples must use imports and constructor signatures that exist in the current package.

## Repository workflow

Repository editing, workspace selection, GitHub Patch/Workspace routing, CI/Actions escalation, concurrency protection, and remote verification are governed by `awa-si/admin/workflow.md` rather than duplicated here.

For this repository, `develop` is the default branch and the repository-specific verification commands are documented in `agent.md` / `docs/agent-domain.md`.

## Pull requests

When a pull request is requested:

- explain the behavioral or architectural change precisely;
- keep unrelated refactors out of the PR;
- include the verification actually performed;
- call out unresolved assumptions or empirical questions explicitly.

## License

By contributing, you agree that your contributions are licensed under the repository's `LICENSE`.
