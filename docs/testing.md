# Testing and verification

The maintained automated test contract is the normal pytest-discovered suite under `tests/`.

## Canonical test suite

```bash
pytest -q
```

`pyproject.toml` limits discovery to `tests/test_*.py`. New maintained regression or contract tests must follow that naming convention. Script-style test files outside discovery are not part of the repository test contract.

During development, focused pytest invocations are appropriate for the affected contract. Shared inference, distributions, context, configuration, training, artifact, or runtime changes should also be verified with the full discovered suite before completion.

## Static checks

```bash
ruff check nhsmm tests scripts
black --check nhsmm tests scripts
```

## Runtime benchmark

The production-reference CPU runtime benchmark operates on a saved NHSMM artifact:

```bash
python scripts/benchmark_runtime.py MODEL_ARTIFACT --steps 128 --warmup 16 --batch-size 1
```

The benchmark reports step-latency statistics, Python heap peak, and bounded runtime-state tensor size. Absolute hosted-runner timings are measurements, not pass/fail thresholds.

## Integration smoke

`.github/workflows/smoke.yml` is a clean-runner integration diagnostic. It installs the package, imports the public API, runs the full discovered pytest suite, and executes the artifact-loaded CPU runtime benchmark.

Routine small checks should be performed locally. GitHub Actions is not the edit/test loop.

## Legacy consumers

Historical tests and experiment scripts that targeted removed constructors or model semantics were removed rather than used to restore obsolete APIs. The current package implementation, `ModelConfig -> NHSMM` construction contract, and discovered `test_*.py` suite are authoritative.
