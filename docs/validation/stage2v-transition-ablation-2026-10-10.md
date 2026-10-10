# Stage 2V.3 — Ground-truth transition ablation (2026-10-10)

## Purpose and protocol

Use known episode boundaries, source states, destination states, episode ages, and external macro-stress to fit the actual NHSMM transition distribution. Compare context ON versus OFF against a pooled empirical row-frequency baseline. All models use 415 train boundaries and 277 held-out boundaries from chronological 60/40 split of `trading-regimes-v1`. Native variants retain the 64 age-indexed transition logits, including in the no-context case. They are trained under direct supervised boundary NLL for 220 Adam steps; **this does not test `NHSMM.optimize` or an integrated latent-state forecast**.

## Observations

| Method | Test transition MAE (lower better) | Test NLL |
| --- | ---: | ---: |
| Pooled empirical, age-independent | 0.03771 | 1.02751 |
| Native NHSMM age-specific, no context (seed 2311) | 0.12605 | 1.43622 |
| Native NHSMM age-specific, no context (seed 2312) | 0.12605 | 1.43622 |
| Native NHSMM age-specific, context (seed 2311) | 0.13871 | 1.50869 |
| Native NHSMM age-specific, context (seed 2312) | 0.13950 | 1.49497 |

Native no-context train NLL 1.38629 → 0.79074. Native context train NLL ~1.386 → 0.671–0.673, yet held-out NLL is worse. The diagnostic supports overfitting with age-specific parameters and further degradation by context conditioning. It does **not** isolate the effect of age dependence from other parameterization/optimization effects; a native age-tied control is still required.

## Guidance

Do not change production defaults or remove context support from the library based on this isolated dataset. Prioritize controlled age-tied transitions, regularization/parameter-sharing, and more discriminating contextual train/test diagnostics. The empirical generator's next-state probability depends on macro-stress, while the pooled estimator estimates a marginal matrix; low pooled MAE does not disprove that small context effects exist.

## Reproduce

`PYTHONPATH=/workspace /work/env/nhsmm/bin/python scripts/diagnose_trading_transition_ablation.py`

Machine-readable evidence: `docs/validation/results/transition-ablation-2026-10-10.json`.
