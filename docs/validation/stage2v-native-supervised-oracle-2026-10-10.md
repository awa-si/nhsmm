# Stage 2V.3 — Native NHSMM supervised oracle (2026-10-10)

This harness fits the native NHSMM duration and transition distributions using known ground-truth episodes from `trading-regimes-v1`. It **does not** run `NHSMM.optimize()` and does not validate integrated unsupervised training. Chronological 60/40 split: 36,000 train bars, 24,000 test bars. Duration estimates exclude split-censored episodes. Transitions are evaluated at true boundaries with known source-state, episode age and macro-stress.

| Metric | Value |
| --- | ---: |
| Duration train NLL before / after | 28.3723 / 4.7801 |
| Duration test NLL | 5.2813 |
| Duration test median relative error | 0.1587 |
| Duration test P(duration > 64) MAE | 0.0342 |
| Transition train NLL before / after | 1.3870 / 0.6748 |
| Transition test NLL | 1.5071 |
| Transition test probability MAE | 0.1401 |
| Transition parameter gradients finite and nonzero | yes |

The independent supervised empirical transition benchmark had test MAE 0.0377. Native transition underperforms this baseline, which may be due to a more highly parameterized age-dependent and context-dependent transition model with limited observed episode boundaries. The mismatch is **not** evidence of a mathematical code defect on its own. The D+ duration tail has a geometric shape and cannot represent arbitrary long-duration mixtures. Native supervised distribution fit does not establish causal state recovery or the correctness of the unsupervised objective.

Reproduce with `PYTHONPATH=/workspace /work/env/nhsmm/bin/python scripts/diagnose_trading_supervised_native.py`. Complete machine-readable output is in `docs/validation/results/native-supervised-oracle-2026-10-10.json`.
