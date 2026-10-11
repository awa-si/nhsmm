# Stage 2V.9 — Emission initialization ablation (2026-10-11)

## Motivation

Stage 2V.8 identified substantial overlap in eight trading-regime features; EWMA volatility channels had the strongest between-regime mean separation, and emission variances may be central to differentiation. NHSMM already provides the generic initialization mode `kmeans_cluster_variance`, which initializes each Gaussian emission's diagonal variances from cluster assignments rather than using all-one variance in the baseline `kmeans` mode. Initialization sensitivity in Gaussian HMMs/HSMMs is documented by Maruotti & Punzo (2021), *International Statistical Review*, DOI: 10.1111/insr.12436.

## Paired unsupervised experiment

`trading-regimes-v1`, train 8 windows × 480 bars sampled from the first 60%; test 4 windows × 480 bars sampled from the final 20%. Seeds 901, 902, 903; `NHSMM.optimize()` uses 10 iterations, no context, the same observation features and parameters except `emission_init_mode`. Labels are used **only for evaluation**. No modifications to production NHSMM or defaults.

| OOS median metric | kmeans | kmeans_cluster_variance |
| --- | ---: | ---: |
| Matched state accuracy | 0.486458 | **0.548438** |
| ARI | 0.120384 | **0.161278** |
| Boundary F1 ±5 bars | **0.166667** | 0.151261 |
| Log-likelihood per bar (higher is better) | -10.115482 | **-9.405505** |

The variance-aware initializer improves OOS log-likelihood, accuracy and ARI consistently across all three seeds, while boundary F1 drops. Therefore the result is **mixed**, and does not pass the regular full-recovery acceptance criteria. The test uses a reduced training sample and lower iteration count than the canonical 24-window/30-iteration benchmark; the median results are not directly comparable to the full benchmark. Emission initialization has a measurable impact but is not demonstrated to resolve duration/boundary inference errors. A full 3-seed benchmark changing **only** the initialization mode is required before promoting this mode to any preset or default.

## Reproduce

`PYTHONPATH=/workspace /work/env/nhsmm/bin/python scripts/diagnose_trading_emission_init_ablation.py`

Machine-readable per-seed results, including initial and fitted emission variances: `docs/validation/results/emission-init-ablation-2026-10-11.json`.

## Full-recovery follow-up interface

The existing `scripts/validate_trading_regime_usefulness.py` now accepts the optional `--emission-init-mode` (`kmeans` or `kmeans_cluster_variance`), propagates it through all model baselines, and records the selected mode in the JSON report. The default remains `kmeans`, so the canonical baseline is unchanged. A complete comparison under identical acceptance gates can be run with:

`PYTHONPATH=/workspace /work/env/nhsmm/bin/python scripts/validate_trading_regime_usefulness.py --dataset data/reference/trading-regimes-v1/trading_regimes.npz --workers 3 --emission-init-mode kmeans_cluster_variance --output /data/workspace/stage2/full-recovery-variance-init.json`

That full-sized run is **not** part of this bounded ablation and must not be claimed as validated until executed.
