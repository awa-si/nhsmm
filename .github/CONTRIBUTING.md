# Contributing to NHSMM

NHSMM is pre-1.0 research software. Changes should preserve the current probabilistic, causal, tensor-shape, and runtime contracts unless a change explicitly redefines them.

## Before changing code

Use the repository control layers according to ownership:

1. Read [`agent.md`](../agent.md) for repository-specific AI behavior and completion gates.
2. Read [`docs/agent-domain.md`](../docs/agent-domain.md) when probabilistic semantics, tensor contracts, causality, API evolution, testing, or research methodology are material.
3. Read [`docs/model.md`](../docs/model.md) and the current implementation for model behavior.
4. Apply the global repository workflow from `awa-si/admin/workflow.md` for editing, verification, CI, and write-back.

The current implementation is authoritative. Do not restore obsolete APIs merely to satisfy historical examples or stale consumers.

## Development

```bash
git clone https://github.com/awa-si/nhsmm.git
cd nhsmm
pip install -e ".[dev]"
pytest -q
ruff check nhsmm tests scripts
black --check nhsmm tests scripts
```

Run the smallest checks that materially validate a change, then expand when shared model/runtime behavior is affected. Never report a test, benchmark, training run, or empirical result as successful unless it was actually executed and observed.

Maintained automated tests follow normal pytest discovery under `tests/test_*.py`; see [`docs/testing.md`](../docs/testing.md).

## Model and API changes

Check affected contracts for:

- tensor dimensions and broadcasting;
- normalization axes;
- initial, transition, duration, and emission semantics;
- duration boundaries and indexing;
- masking and variable-length behavior;
- dtype/device handling;
- log-space numerical stability;
- gradient flow;
- causal versus retrospective inference semantics.

The canonical public construction path is configuration-driven:

```python
from nhsmm import ModelConfig, NHSMM

config = ModelConfig(n_states=3, n_features=5)
model = NHSMM(config=config)
```

When changing `ModelConfig` or public exports, update dependent source, tests, examples, and documentation coherently.

## Documentation

Documentation must describe implemented behavior. Do not make performance, production-readiness, causal, scalability, or hardware claims without appropriate implementation or benchmark evidence.

## Pull requests

Keep changes focused. Describe the behavioral or architectural change, verification actually performed, and unresolved assumptions or empirical questions. Avoid unrelated refactors.

## License

Contributions are licensed under the repository's Apache-2.0 [`LICENSE`](../LICENSE).
