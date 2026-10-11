# Stage 2V.8 — Trading emission identifiability diagnosis

## Protocol

Dataset: `trading-regimes-v1` (60,000 bars, 4 regimes, 8 feature channels). Chronological 60/40 split: 36,000 training and 24,000 held-out bars. Training-only feature centering/scaling and state-labeled Gaussian fitting used for diagnostic upper-reference classifiers. Both diagonal-covariance and full-covariance supervised Gaussian classifiers are evaluated out of sample. A separate, **unsupervised** `NHSMM.optimize()` diagnostic trains on 12 × 480 windows from the first 60% and evaluates raw emission argmax on 4 × 480 windows from the final 20%, using 12 iterations and seed 901. True labels are used only to align/evaluate the fitted model, never for NHSMM optimization. These protocols have different evaluation samples and are not directly comparable as controlled training variants.

| Classifier | OOS matched accuracy | ARI |
| --- | ---: | ---: |
| Supervised diagonal Gaussian | 0.641167 | 0.297212 |
| Supervised full-covariance Gaussian | 0.661833 | 0.323982 |
| Unsupervised NHSMM emission-only argmax (separate window sample) | 0.492188 | 0.122527 |

The most discriminating single feature by between-regime mean variance / within-regime variance is `ewma_vol_12` (1.142), followed by `ewma_vol_48` (0.840). Immediate log returns (`log_return_1`: 0.0045) and short-term returns (`log_return_3`: 0.0129) have very low mean-discrimination ratios. This ratio only measures mean separation and does not capture discriminating covariance or heteroscedasticity.

The full-covariance oracle gives a modest improvement over the diagonal classifier (+2.07 percentage points accuracy). Fitted unsupervised Gaussian emissions alone do not closely recover labeled states. This points to both overlap in the observations and limitations of unsupervised regime identification; it does not prove the optimizer or likelihood code is defective. The complete 3-seed recovery suite has **not** been rerun under this diagnostic; do not represent these results as passing production acceptance.

## Reproduction

`PYTHONPATH=/workspace:/workspace/scripts /work/env/nhsmm/bin/python scripts/diagnose_trading_emission_identifiability.py`

Machine-readable outputs (including confusion matrices, emission parameters and feature ratios): `docs/validation/results/emission-identifiability-2026-10-11.json`.
