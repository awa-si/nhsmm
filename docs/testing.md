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

The 8 GiB storage allocation is sufficient for the complete CPU pytest suite when PyTorch is installed from the CPU wheel index. A generic `pip install -e .` under Python 3.14 currently resolves the CUDA-enabled PyTorch dependency set and can exceed the retained-size guard.

For a CPU-only AWA verification environment:

```bash
python -m venv .venv
.venv/bin/python scripts/install_cpu_dev.py
```

The bootstrap installs `torch>=2.2` from PyTorch's CPU wheel index first, then
runs the normal editable `.[dev]` install. Because the Torch requirement is
already satisfied, pip does not resolve the CUDA dependency set.

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
- transition-specific context capacity without changing Initial/Duration/Emission bounds;
- caller train/eval mode restoration when optimization fails during input validation.

`tests/test_context_hardening.py` owns external-context shape, finiteness, variable-length, and width contracts. `tests/test_context_optin.py` owns internal-encoder opt-in and model-construction contracts. `tests/test_default_distributions.py` verifies fail-closed categorical labels/temperature controls plus frozen context-free neural-control parameters.

## HSMM research/correctness milestone gate

The independent HSMM correctness milestone is owned by this testing contract. Its acceptance harness should live under `tests/reference/` and remain deliberately independent from optimized production recursion.

The execution sequence and stop/go criteria are defined in [hsmm-correctness-work-gates.md](hsmm-correctness-work-gates.md).

Required reference coverage:

- exhaustive small-model joint path enumeration for likelihood and filtering;
- exact MAP/Viterbi reference from the same enumerated path set;
- explicit state-age and episode-boundary marginals;
- degenerate cases `K=1`, `D=1`, `T=1`, self-transitions, impossible support, and near-deterministic probabilities;
- Float64 gradient checks for initial, duration, transition, emission, and end-to-end log-likelihood;
- property-based invariants for normalization, causal prefixes, right padding, streaming equivalence, support preservation, and age reset on episode boundaries.

Reference code must not call `NHSMM.forward`, `NHSMM.decode`, `filter_step`, `filter_model_sequence`, `duration_log_hazard`, `HSMMFilterRuntime`, or private production recursion helpers. Shared data containers and raw distribution parameter extraction are allowed only when they do not reproduce production inference logic.

Numerical acceptance must distinguish mathematical equality from floating-point execution:

- Float64 reference comparisons use the tightest tolerance justified by observed error;
- Float32 batch/filter/runtime comparisons use a measured bounded tolerance recorded with the test;
- impossible support remains exact where the contract requires `-inf`/zero mass.

Statistical recovery harnesses remain a separate acceptance layer and do not substitute for this mathematical gate.

## Static checks

```bash
ruff check nhsmm tests scripts
black --check nhsmm tests scripts
```

## Controlled duration-context acceptance

```bash
python scripts/validate_duration_context.py --output /tmp/nhsmm-duration-context.json
```

The harness explicitly opts into the internal ContextEncoder because duration context is derived from observations. Independent seed runs are process-parallel by default with up to four single-threaded workers; use `--workers 1` for serial diagnostics.

Re-run after material changes to training objective, restart semantics, duration parameterization, context encoder, optimizer coverage, causal filtering, or survival semantics.

## Controlled latent-state acceptance

```bash
python scripts/validate_state_recovery.py \
  --scenario all \
  --workers 1 \
  --output /tmp/nhsmm-state-recovery.json
```

The benchmark evaluates state identity only up to permutation and explicitly uses `emission_init_mode="kmeans"`; package default remains `spread`.

## Controlled learning/prediction acceptance

    python scripts/validate_learning_prediction.py --scenario all --seeds 401,402,403,404,405 --max-iter 40 --workers 4 --output /tmp/nhsmm-learning-prediction.json

This is the canonical empirical proof that optimize() changes learned state and that the fitted model improves held-out likelihood and predicts identifiable latent states on disjoint OOS sequences.

The harness now separates two questions:

1. **learning_pass** — parameter movement, held-out likelihood gain, permutation-invariant OOS state recovery, ARI, provenance, and variable-length prediction;
2. **usefulness_pass** — boundary timing, regime run lengths, transition recovery, occupancy recovery, posterior calibration, collapse behavior, and train/OOS stability.

Usefulness metrics include exact boundary F1, a causal ±1-timestep boundary F1 for weak-signal interpretation, run-length relative error, transition-matrix MAE, occupancy L1, posterior ECE/Brier, and health diagnostics.

Latest 5-seed evidence:

| scenario | OOS accuracy | ARI | boundary F1 | boundary F1 ±1 | run-length rel. error | transition MAE | occupancy L1 | posterior ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| strong | 0.9989 | 0.9965 | 0.9890 | 0.9944 | 0.0108 | 0.0029 | 0.0022 | 0.0032 |
| moderate | 0.9633 | 0.8936 | 0.7568 | 0.9205 | 0.0213 | 0.0096 | 0.0222 | 0.0309 |
| weak | 0.7856 | 0.4642 | 0.3222 | 0.6000 | 0.0652 | 0.0788 | 0.1733 | 0.1687 |
| null | 0.3589 | 0.0000 | 0.0000 | 0.0000 | 17.8000 | 0.3333 | 1.2822 | 0.6238 |

Strong and moderate pass both learning and usefulness cleanly. Weak passes a deliberately lower-confidence usefulness contract; exact boundary timing is explicitly weaker, while ±1-timestep timing, durations, transitions, occupancy, and calibration remain informative. Null is a negative control: learning-control checks pass, but usefulness is not applicable and no predictive state identity is claimed.

The hard state-identification acceptance explicitly uses emission_init_mode="kmeans". A separate default-spread probe is recorded in validation/README.md.

The harness configuration is now externalized through the public `ValidationConfig` contract. A complete JSON contract can be supplied with `--config`; `--dump-config` writes the fully resolved contract; repeated `--set` arguments tune fields such as `model.max_iter`, `data.steps`, `health.max_state_occupancy`, `boundary_tolerance`, or `scenario.moderate.max_median_ece` without code edits. See [configuration.md](configuration.md).


## Controlled transition-context acceptance

```bash
python scripts/validate_transition_context.py \
  --scenario all \
  --workers 1 \
  --output /tmp/nhsmm-transition-context.json
```

The transition harness uses known context-dependent transition matrices, permutation-invariant state alignment, strong/moderate effects, and a null control. It sets `max_duration=1` to isolate transition recovery. Maintained validation explicitly opts into `transition_context_max_delta=1.5`, `transition_refine_steps=20`, and `transition_refine_lr=0.03`; package defaults remain conservative and refinement remains opt-in.

Re-run after changes to transition context parameterization, context-network hidden dimensions, refinement likelihood, filtering boundary semantics, state recovery, or external-context handling.

Acceptance thresholds and latest observed results are recorded in [`validation.md`](validation.md).

## Runtime benchmark

```bash
python scripts/benchmark_runtime.py MODEL_ARTIFACT --steps 128 --warmup 16 --batch-size 1
```

Latency and Python-allocation measurements use separate passes; `tracemalloc` is not active during latency measurement. Absolute hosted-runner timing is evidence, not a pass/fail threshold.

## Integration smoke

`.github/workflows/smoke.yml` is a clean-runner integration diagnostic started manually with `workflow_dispatch`. Run the local/focused checks first and use the smoke workflow after the relevant local gate is green. Routine edit/test work remains local; GitHub Actions is not the primary edit/test loop.

## Downstream Nautilus usability check

Nautilus empirical checks are downstream validation, not package-core acceptance. The repository workspace profile declares:

```ini
[workspace]
data = nautilus
```

AWA mounts the dataset read-only at `/data/nautilus`; do not copy the dataset into the repository or workspace.

The maintained current state-count evidence is recorded in [`validation/README.md`](validation/README.md). Re-run that comparison after material model/training/decoder changes or changes to the Nautilus 18-D observation mapping.
