# Contributing to NHSMM

Thank you for contributing to NHSMM (Neural Hidden Semi-Markov Models).

Contributions may include bug fixes, documentation, numerical-stability improvements, tests, performance work, examples, and research-driven extensions.

## Before you change code

Read [`agent.md`](./agent.md). It defines the repository-specific engineering, probabilistic, causal, verification, and GitHub workflow rules.

The current implementation is authoritative. Historical scripts, tests, and documentation may contain obsolete API names or constructor patterns, so inspect the relevant package code before changing behavior.

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
pytest -v
ruff check nhsmm tests scripts
black --check nhsmm tests scripts
```

For shared inference, distribution, context, configuration, or training changes, run the broader relevant test set.

Never report a test, benchmark, training run, or empirical result as successful unless it was actually executed and observed.

## Model changes

NHSMM is an explicit-duration probabilistic sequence model. Changes must preserve or intentionally redefine the affected mathematical contract.

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

GitHub Patch is the preferred repository editing workflow when the plugin/skill is available. Use it for precise diffs, atomic multi-file changes, post-commit verification, and CI/status inspection when relevant.

Otherwise, use the closest available GitHub workflow while preserving the same rules:

- read before writing;
- minimal coherent changes;
- concise factual commits;
- direct work on `develop` unless another branch is requested;
- no force pushes;
- verify the resulting commit and changed files;
- do not create a pull request unless requested or direct writing is unavailable and a PR is an authorized fallback.

Do not use GitHub Actions or workflow modifications merely as a mechanism for editing repository content.

## Pull requests

When a pull request is requested:

- explain the behavioral or architectural change precisely;
- keep unrelated refactors out of the PR;
- include the verification actually performed;
- call out unresolved assumptions or empirical questions explicitly.

## License

By contributing, you agree that your contributions are licensed under the repository's `LICENSE`.
