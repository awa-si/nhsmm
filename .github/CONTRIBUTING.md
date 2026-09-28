# Contributing to NHSMM

NHSMM is pre-1.0 research software. Changes should preserve the documented probabilistic, causal, tensor-shape, and runtime semantics unless the change intentionally revises them.

## Development setup

```bash
git clone https://github.com/awa-si/nhsmm.git
cd nhsmm
python -m pip install -e ".[dev]"
```

The package supports Python 3.9+ according to `pyproject.toml`. CI currently exercises Python 3.11 for repository smoke and release checks.

## Before opening a pull request

Start with the smallest test set that covers the change. Run the full suite when shared model, inference, runtime, distribution, configuration, or validation behavior is affected.

```bash
python -m pytest -q
ruff check nhsmm tests scripts
black --check nhsmm tests scripts
```

Only report checks that were actually run. If a check is intentionally omitted, note that in the pull request.

See [`docs/testing.md`](../docs/testing.md) for the repository verification policy and [`docs/model.md`](../docs/model.md) for the current model/runtime contract.

## Model and API changes

Review the affected behavior for:

- tensor dimensions and broadcasting;
- probability normalization axes;
- initial, transition, duration, and emission semantics;
- duration indexing and episode-boundary behavior;
- masking and variable-length sequences;
- dtype and device handling;
- log-space numerical stability;
- gradient flow;
- causal versus retrospective inference;
- public imports, configuration fields, and serialized artifacts.

The canonical construction path is configuration-driven:

```python
from nhsmm import ModelConfig, NHSMM

config = ModelConfig(n_states=3, n_features=5)
model = NHSMM(config=config)
```

When changing `ModelConfig`, public exports, artifacts, or runtime behavior, update source, tests, examples, and documentation together.

## Documentation

Document implemented behavior rather than expected or aspirational behavior. Benchmarks and validation results should identify the setup they measure and should not be generalized to downstream domains without separate evidence.

## Pull requests

Keep pull requests focused. Include:

- what changed;
- which contract or behavior is affected;
- compatibility implications, if any;
- exact verification performed;
- unresolved assumptions or follow-up work.

Avoid unrelated refactors in the same pull request.

## License

Contributions are licensed under the repository's Apache-2.0 [`LICENSE`](../LICENSE).
