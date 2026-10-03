# Testing and verification

The maintained automated contract is the pytest-discovered suite under `tests/`.

## AWA workspace resources

Repository workspace defaults are defined in `workspace.ini`:

```ini
[resources]
cpu = 4
memory_mb = 8192
storage_mb = 8192
pids = 512
tmp_mb = 1024
```

The 8 GiB storage allocation is required for PyTorch development installs plus the complete CPU pytest suite. AWA currently accepts this profile; do not reduce storage back to 4 GiB for full-suite verification.

## Canonical suite

```bash
pytest -q
```

`pyproject.toml` limits discovery to `tests/test_*.py`. Focused tests are appropriate during development; shared training/runtime changes should also receive the full suite when it is available locally.

`tests/test_training.py` covers:

- optimizer coverage and causal-encoder update;
- default 40-iteration joint-training budget;
- independent restart initialization and encoder/duration-bias reset;
- deep best-run snapshots including `duration_logits_bias`;
- K-Means emission initialization on separated observations;
- end-to-end multi-restart optimization;
- scalar external-context training with non-degenerate distribution hidden width;
- transition-only refinement parameter isolation;
- refinement opt-in defaults;
- transition-specific context capacity without changing Initial/Duration/Emission bounds.

## Static checks

```bash
ruff check nhsmm tests scripts
black --check nhsmm tests scripts
```

## Controlled duration-context acceptance

```bash
python scripts/validate_duration_context.py --output /tmp/nhsmm-duration-context.json
```

Re-run after material changes to training objective, restart semantics, duration parameterization, context encoder, optimizer coverage, causal filtering, or survival semantics.

## Controlled latent-state acceptance

```bash
python scripts/validate_state_recovery.py \
  --scenario all \
  --workers 1 \
  --output /tmp/nhsmm-state-recovery.json
```

The benchmark evaluates state identity only up to permutation and explicitly uses `emission_init_mode="kmeans"`; package default remains `spread`.

## Controlled transition-context acceptance

```bash
python scripts/validate_transition_context.py \
  --scenario all \
  --workers 1 \
  --output /tmp/nhsmm-transition-context.json
```

The transition harness uses known context-dependent transition matrices, permutation-invariant state alignment, strong/moderate effects, and a null control. It sets `max_duration=1` to isolate transition recovery. Maintained validation explicitly opts into `transition_context_max_delta=1.5`, `transition_refine_steps=20`, and `transition_refine_lr=0.03`; package defaults remain conservative and refinement remains opt-in.

Re-run after changes to transition context parameterization, context-network hidden dimensions, refinement likelihood, filtering boundary semantics, state recovery, or external-context handling.

Acceptance thresholds and latest observed results are recorded in [`package-validation.md`](package-validation.md).

## Runtime benchmark

```bash
python scripts/benchmark_runtime.py MODEL_ARTIFACT --steps 128 --warmup 16 --batch-size 1
```

Latency and Python-allocation measurements use separate passes; `tracemalloc` is not active during latency measurement. Absolute hosted-runner timing is evidence, not a pass/fail threshold.

## Integration smoke

`.github/workflows/smoke.yml` is a clean-runner integration diagnostic started manually with `workflow_dispatch`. Run the local/focused checks first and use the smoke workflow after the relevant local gate is green. Routine edit/test work remains local; GitHub Actions is not the primary edit/test loop.
