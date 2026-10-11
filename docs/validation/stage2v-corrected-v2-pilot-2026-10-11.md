# Stage 2V.10 — Corrected generator and bounded recovery pilot

## Changes

- Corrected dataset v2 selects the next state using the final observed macro_stress of the current episode, matching the causal model boundary convention.
- Legacy v1 numerical arrays remain unchanged; generate(..., transition_context_mode="episode_start") reproduces them. The default generator now emits v2.
- The maintained usefulness CLI rejects legacy start-context manifests.
- Boundary F1 uses maximum-cardinality matching, verified against an independent dynamic programming reference across 3,072 small cases.
- Missing predicted states now carry a 100% duration-error penalty and fail an explicit duration-coverage gate.
- Generator NumPy type annotations resolve correctly.
- CLI budgets expose seeds, iterations, transition refinement, window length and window counts; each result records the resolved budget and whether it matches the full protocol.
- No NHSMM production model parameters or package defaults were changed.

## Validation and pilot

Syntax and CLI checks passed. Ten focused regression tests passed, including v1 numerical reproduction, v2 context targeting, maximum boundary matching, missing-state coverage and annotation resolution.

The 1-iteration all-arm smoke completed in 33.728 seconds with empty stderr and finite summary metrics. Its exit code 1 reflects failed statistical acceptance, not a runtime exception.

The requested bounded pilot used a hard runner timeout of 300 seconds and completed in 46.205 seconds. Seeds 901/902/903 ran in three process workers with single-threaded Torch/BLAS. Budget: 4 joint iterations, 2 context-transition refinement steps, 8 training windows, 4 evaluation windows, 240 bars per window, kmeans_cluster_variance initialization.

| Median metric | Result | Acceptance |
| --- | ---: | ---: |
| Accuracy | 0.792708 | >= 0.65 |
| ARI | 0.559971 | >= 0.40 |
| Boundary F1 +/-5 | 0.184211 | >= 0.45 |
| Filtered state-run duration relative error | 0.922500 | <= 0.50 |
| Transition MAE | 0.149596 | <= 0.20 |
| OOS likelihood per row | -8.298316 | higher is better |
| Full minus no-context HSMM likelihood | -0.145374 | >= -0.001 |

State and transition checks pass on this pilot; boundary, duration and no-context comparison checks fail. Overall acceptance remains false. This is a short pilot, not a full recovery gate.

Dataset and training budget both differ from the historical v1 full run; no controlled improvement claim is made from that comparison. Filtered argmax fragmentation remains a problem, but this score alone does not establish incorrect learned duration distributions.

## Reproduction

Generate corrected data:

    PYTHONPATH=/workspace /work/env/nhsmm/bin/python scripts/generate_trading_regime_dataset.py --output-dir /data/workspace/nhsmm-trading-regimes-v2

Run with workspace_exec_start(timeout_seconds=300):

    PYTHONPATH=/workspace OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /work/env/nhsmm/bin/python scripts/validate_trading_regime_usefulness.py --dataset /data/workspace/nhsmm-trading-regimes-v2/trading_regimes.npz --workers 3 --seeds 901,902,903 --max-iter 4 --transition-refine-steps 2 --window 240 --train-windows 8 --eval-windows 4 --emission-init-mode kmeans_cluster_variance --output /data/workspace/stage2/corrected-v2-pilot-2026-10-11.json

The timeout is enforced by the AWA runner, not by these CLI budget flags.
Dataset: seed 424242, 60000 rows, start price 50000, schema nhsmm-synthetic-trading-regimes-v2.
NPZ SHA-256: 16cab157954b32f11af274bd617d6811a3776e81eb926a3f1dae80c954d7a1f8.
Corrected numerical data is retained in the existing workspace; v1 committed reference data is preserved.
Results: results/corrected-v2-pilot-2026-10-11.json and results/corrected-v2-smoke-2026-10-11.json.
