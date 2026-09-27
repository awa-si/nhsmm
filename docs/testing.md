# Testing and verification

The maintained automated test contract is the normal pytest-discovered suite under `tests/`.

## Canonical test suite

```bash
pytest -q
```

`pyproject.toml` limits discovery to `tests/test_*.py`. New maintained regression or contract tests must follow that naming convention. Script-style test files outside discovery are not part of the repository test contract.

During development, focused pytest invocations are appropriate for the affected contract. Shared inference, distributions, context, configuration, training, artifact, or runtime changes should also be verified with the full discovered suite before completion when the full suite is available in the local workspace.

The training regression in `tests/test_training.py` covers:

- optimizer coverage and an actual causal-encoder parameter update;
- the default 40-iteration maximum budget;
- restart initialization without encoder or duration-bias warm-start leakage;
- end-to-end multi-restart optimization and restoration of a finite best candidate;
- independent best-run snapshots including `duration_logits_bias`;
- training-time K-Means emission initialization on separated observations.

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

Re-run this harness after material changes to the training objective, initialization/restart semantics, duration parameterization, context encoder, optimizer coverage, causal filter, or survival semantics.

## Controlled latent-state recovery acceptance

The latent-state recovery harness evaluates state identity only up to permutation. It uses known synthetic state labels, strong/moderate emission separation, a null-separation negative control, and multiple independent seeds. The benchmark intentionally uses `emission_init_mode="kmeans"`; the package default remains `spread`.

```bash
python scripts/validate_state_recovery.py \
  --scenario all \
  --workers 1 \
  --output /tmp/nhsmm-state-recovery.json
```

Parallel workers may be used for local research runs when CPU resources permit. Acceptance thresholds and the latest validated evidence are documented in [`package-validation.md`](package-validation.md).

## Runtime benchmark

The production-reference CPU runtime benchmark operates on a saved NHSMM artifact:

```bash
python scripts/benchmark_runtime.py MODEL_ARTIFACT --steps 128 --warmup 16 --batch-size 1
```

The benchmark reports step-latency statistics, Python heap peak, and bounded runtime-state tensor size. Latency and Python-allocation measurements use separate passes: `tracemalloc` is not active during the latency pass because allocation tracing materially perturbs one-step timing.

Absolute hosted-runner timings are measurements, not pass/fail thresholds. Performance work should use a local editable install and before/after measurements on the same host; the clean runner is only a final integration reference.

## Integration smoke

`.github/workflows/smoke.yml` is a clean-runner integration diagnostic. Routine small checks should be performed locally; GitHub Actions is not the edit/test loop.
