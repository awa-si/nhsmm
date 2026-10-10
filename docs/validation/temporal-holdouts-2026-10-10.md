# Stage 2U — Chronological holdout replication

## Setup

Train FrozenNHSMM variant A with 30 iterations on the first 60% of the trading-regimes-v1 dataset. Evaluate inference-only emission scale 0.25 versus 1.0 on two disjoint later ranges (60–80% and 80–100%), 8 evaluation windows per range, seeds 904 and 907. Source: `scripts/diagnose_trading_temporal_holdouts.py`. Raw results: `results/temporal-holdout-{904,907}.json`.

## Findings

- 60–80% holdout: scale 0.25 reduces switches 75→54 (seed 904) and 75→56 (seed 907), but decreases matched accuracy (~0.4940→~0.4900), ARI (~0.2248→~0.2105) and boundary F1 (~0.2069→~0.146).
- 80–100% holdout: scale 0.25 reduces switches 142→98 for both seeds, and improves matched accuracy (~0.4924→~0.500), ARI (~0.2048→~0.231) and boundary F1 (~0.1522→~0.1714).
- The two seeds yield near-identical metrics. Do not interpret them as independent confirmations without comparing their fitted paths.
- This is a synthetic reference dataset with disjoint evaluation time ranges but overlapping evaluation windows within each range; avoid treating each window as an independent sample.

## Decision

Do not change the default emission scale from 1.0 based on this evidence. Reduced switching alone is not sufficient: segmentation quality has mixed out-of-sample results. Keep the current repository baseline and require broader temporal / dataset diversity before tuning production defaults.
