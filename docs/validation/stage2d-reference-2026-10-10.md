# Stage 2D — Full reference run (2026-10-10)

- Repository: `awa-si/nhsmm`, branch `regime-v2`.
- Input: `data/reference/trading-regimes-v1/trading_regimes.npz` (60,000 five-minute bars; seed 424242).
- Command: `PYTHONPATH=/workspace /work/env/nhsmm/bin/python scripts/diagnose_trading_freeze_2d.py --variant A --seed 901 --dataset data/reference/trading-regimes-v1/trading_regimes.npz --output /data/workspace/stage2/stage2d-A-901.json`
- Configuration: 30 training iterations, K=4, maximum duration D=64, 24 train windows and 8 evaluation windows, 480 bars/window.
- Result: process exit 0, wall time 602.715 seconds.
- Full machine-readable result: `results/stage2d-A-901-2026-10-10.json`.

## Findings

- Matched accuracy: **0.4953125**; ARI: **0.2059640**.
- Boundary F1 within ±5 bars: **0.1555556**.
- Out-of-sample log likelihood per row: **-5.5015920**.
- Median predicted run lengths (bull/bear/range/chaotic): **10.5 / 14 / 18 / 10** bars.
- True median run lengths: **109 / 82 / 45 / 28** bars.

The predicted state sequences switch much more frequently than the ground truth. The results motivate controlled emission-scaling (Stage 2R) and transition-persistence (Stage 2S) experiments; they do **not** establish that either adjustment will improve out-of-sample performance.

## Stage transition protocol

Before beginning the next stage: (1) save the reproducible command, parameters and complete results, (2) state findings and limitations, (3) run checks, (4) commit and verify push. Repeat at every stage boundary.
