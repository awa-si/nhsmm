# Stage 2V.1 — Ground-truth oracle structure

## Reproduction

`PYTHONPATH=/workspace /work/env/nhsmm/bin/python scripts/diagnose_trading_oracle_structure.py --output /data/workspace/stage2/oracle-structure.json`

Dataset: `data/reference/trading-regimes-v1/trading_regimes.npz` (60,000 bars, 693 episodes). Raw results: `results/oracle-structure.json`.

## Ground-truth findings

| State | Complete episodes | Mean duration | Median duration | P(duration > 64) |
|---|---:|---:|---:|---:|
| Bull | 141 | 120.42 | 117 | 0.9362 |
| Bear | 158 | 96.27 | 89.5 | 0.7468 |
| Range | 212 | 100.80 | 120.5 | 0.5708 |
| Chaotic | 181 | 35.46 | 31 | 0.0773 |

The last episode is right-censored and excluded from the duration estimates. Across 692 observed episode boundaries, no diagonal transition occurs. The empirical transition probabilities and generator base matrix are recorded in the JSON. They need not match exactly because generator transition weights are modulated by macro stress and are sampled finitely.

## Interpretation

`max_duration=64` with a geometric tail can represent durations above 64, but cannot in general express the generator's full tail hazard shape (especially the range mixture). The baseline's uniform duration and transition initialization does not encode generator ground truth. Neither fact alone demonstrates an implementation defect. Next gates: (a) verify filter/transition/duration primitives on known hazards and exact oracle parameters, (b) compare supervised parameter recovery with unsupervised learning, (c) only then modify package semantics where tests identify a defect. No default change at this gate.
