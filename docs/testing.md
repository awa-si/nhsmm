# Testing and verification

The maintained automated test contract is the normal pytest-discovered suite under `tests/`.

## Canonical test suite

```bash
pytest -q
```

`pyproject.toml` limits discovery to `tests/test_*.py`. New maintained regression or contract tests must follow that naming convention. Script-style test files outside discovery are not part of the repository test contract.

During development, focused pytest invocations are appropriate for the affected contract. Shared inference, distributions, context, configuration, training, artifact, or runtime changes should also be verified with the full discovered suite before completion.

The training regression in `tests/test_training.py` verifies two package-level contracts:

- `NHSMM.optimize()` includes trainable causal-encoder parameters and actually updates at least one encoder parameter;
- the default training iteration budget is `40`.

## Static checks

```bash
ruff check nhsmm tests scripts
black --check nhsmm tests scripts
```

## Controlled duration-context acceptance

Expensive package-level empirical validation is separate from routine pytest. The synthetic duration-context acceptance harness uses known ground truth, independent train/evaluation sequences, multiple seeds, strong/moderate context effects, and a null-context negative control.

```bash
python scripts/validate_duration_context.py --output /tmp/nhsmm-duration-context.json
```

The maintained acceptance thresholds and the latest validated evidence are documented in [`package-validation.md`](package-validation.md). Re-run this harness after material changes to the training objective, duration parameterization, context encoder, optimizer coverage, causal filter, or survival semantics.

## Runtime benchmark

The production-reference CPU runtime benchmark operates on a saved NHSMM artifact:

```bash
python scripts/benchmark_runtime.py MODEL_ARTIFACT --steps 128 --warmup 16 --batch-size 1
```

The benchmark reports step-latency statistics, Python heap peak, and bounded runtime-state tensor size. Latency and Python-allocation measurements use separate passes: `tracemalloc` is not active during the latency pass because allocation tracing materially perturbs one-step timing. Benchmark inputs are generated before the measured loops so RNG/allocation setup is not included in step latency.

Absolute hosted-runner timings are measurements, not pass/fail thresholds. Performance work should use a local editable install and before/after measurements on the same host; the clean runner is only a final integration reference.

## Performance profiling workflow

Performance optimization is local-first and measurement-driven.

1. Materialize the current `develop` state into a local workspace, preferably `/tmp/nhsmm`.
2. Prefer an editable install. When build isolation cannot access package indexes, `pip install -e . --no-deps --no-build-isolation` is acceptable only when the local environment already contains the required dependencies; it is not a clean dependency-install validation.
3. Measure the complete steady-state `HSMMFilterRuntime.step()` path before changing code.
4. Attribute time to the major runtime components before selecting a target: encoder, emission scoring, duration/transition scoring, normalized filtering, and runtime state/validation handling.
5. Compare before/after on the same host with the same model dimensions, batch size, warmup, number of iterations, device, dtype, and inference mode.
6. Report at least mean, p50, and p95 when the measurement is stable enough to support them.
7. Keep latency and allocation measurements in separate passes.
8. Validate optimized private kernels against the public/reference path within the repository tolerance contract.
9. Do not claim a speedup from hosted-runner comparisons across different runners.
10. Run the full discovered pytest suite before completion; use the smoke workflow only as the final clean-runner integration check.

Microbenchmarks are diagnostic evidence, not sufficient justification by themselves. A micro-optimization that improves an isolated kernel but does not improve its real caller should be rejected. Preserve probabilistic semantics, temporal causality, tensor-shape contracts, public APIs, artifact/state-dict compatibility, and bounded streaming state unless the task explicitly changes one of those contracts.

The immediate performance continuation point and rejected recent candidates are recorded in `docs/handoff.md`. `docs/state.md` remains the status/readiness owner.

## Integration smoke

`.github/workflows/smoke.yml` is a clean-runner integration diagnostic. It installs the package, imports the public API, runs the full discovered pytest suite, and executes the artifact-loaded CPU runtime benchmark.

Routine small checks should be performed locally. GitHub Actions is not the edit/test loop.

## Legacy consumers

Historical tests and experiment scripts that targeted removed constructors or model semantics were removed rather than used to restore obsolete APIs. The current package implementation, `ModelConfig -> NHSMM` construction contract, and discovered `test_*.py` suite are authoritative.
