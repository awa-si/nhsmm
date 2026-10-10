# Stage 2R — Emission scaling, full 30-iteration run

- Dataset: `data/reference/trading-regimes-v1/trading_regimes.npz`.
- Variant A, seed 901, 30 iterations; trained once, then evaluated six inference-only emission-log-score scales.
- Command: `PYTHONPATH=/workspace /work/env/nhsmm/bin/python scripts/diagnose_trading_emission_scale_2r.py --seed 901 --max-iter 30 --output /data/workspace/stage2/stage2r-A-901.json`.
- Exit code 0; elapsed 592.11 s.
- Raw results: `results/stage2r-A-901-2026-10-10.json`.

| Scale | Accuracy | ARI | Boundary F1 ±5 | Switches |
|---|---:|---:|---:|---:|
| 0.25 | 0.500260 | 0.232679 | 0.181818 | 101 |
| 0.50 | 0.497135 | 0.220820 | 0.149068 | 119 |
| 0.75 | 0.494531 | 0.210101 | 0.147727 | 134 |
| 1.00 | 0.495313 | 0.205964 | 0.155556 | 138 |
| 1.50 | 0.495833 | 0.203054 | 0.150754 | 157 |
| 2.00 | 0.495313 | 0.197933 | 0.148148 | 174 |

## Interpretation and limitations

Scale 0.25 leads this single-seed sweep on accuracy, ARI, and boundary F1, and reduces switches from 138 to 101 (-26.8%). Accuracy improvement is small (+0.495 percentage points); no robustness or generalization claim is justified. The scaling is applied only during causal inference after a common fit and therefore does not produce calibrated generative likelihoods. Selection on these evaluation windows is exploratory and should not be reported as unbiased out-of-sample performance.

## Handoff

Stage 2S should test emission scales 0.25 and 1.0 against transition-persistence controls, using multiple seeds and clearly separating exploratory selection from confirmatory evaluation. Do not begin until this stage is committed and pushed.
