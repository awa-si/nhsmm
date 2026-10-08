# Synthetic trading-regime dataset

`wenerate_trading_regime_dataset.py` creates a fully labeled, chronological OHLCV dataset for mechanism-level NHSMM evaluation. It is deliberately separate from real-market Gates 11-14.

## Ground-truth regimes

- `bull`: positive drift, moderate volatility, positive short-memory return persistence, long shifted-gamma episodes.
- `bear`: negative drift, higher volatility, downside jump asymmetry, shifted-Weibull episodes.
- `range`: near-zero drift, negative return autocorrelation plus episode-anchor mean reversion, bimodal episode durations.
- `chaotic`: near-zero drift, extreme volatility, frequent large jumps, short heavy-tailed episodes.

Episode durations are explicit and non-geometric. A state never transitions directly to itself; persistence is represented by episode duration, matching HSMM semantics.

## Context-conditioned transitions

A causal observable `macro_stress` follows a persistent stochastic process. At each episode boundary it tilts the known base transition law:

- higher stress increases `bear` / `chaotic` probability;
- lower stress increases `bull` / `range` probability.

The resulting ground-truth next-state probabilities are emitted as `gt_p_next_*` evaluation columns. They are labels and must not be model inputs.

## Data products

Default generation creates 60,000 five-minute bars, about 208 days, beginning at `2024-01-01T00:00:00Z`:

```bash
python scripts/generate_trading_regime_dataset.py \
  --output-dir /data/workspace/nhsmm-trading-regimes-v1
```

Outputs:

- `trading_regimes.npz` — compact numerical arrays;
- `trading_regimes.csv.gz` — inspection/interoperability form;
- `manifest.json` — seed, regime definitions, transition law, duration families, episode ledger, checksums and chronological split.

The split is frozen chronologically at 60% train / 20% validation / 20% test.

## Causality boundary

All columns not prefixed `gt_` are contemporaneous or backward-looking. Every `gt_*` column is ground truth for scoring only and is prohibited as a model feature.

## Intended NHSMM usefulness tests

This dataset can support four distinct controlled claims without using PnL to tune the representation:

 1. latent-state recovery up to permutation for the four named regimes;
 2. explicit-duration recovery and non-constant episode-end hazard;
 3. recovery of the stress-conditioned transition law;
 4. held-out density / future-outcome information versus simpler HMM or no-regime baselines.

Passing these synthetic tests demonstrates that the model can exploit known trading-like state, duration, and transition structure. It does not establish that the same structure exists in real markets.

## Frozen reference instance

The maintained reference generation uses seed `424242`, 60,000 rows, and start price `50000`.

Observed ground-truth coverage:

| Split | bull | bear | range | chaotic |
| --- | ---: | ---: | ---: | ---: |
| train | 9,224 | 9,206 | 13,416 | 4,154 |
| validation | 4,229 | 2,497 | 4,301 | 973 |
| test | 3,547 | 3,508 | 3,653 | 1,292 |

Observed full-sample signatures:

| Regime | Mean return (bp/bar) | Volatility (bp/bar) | Median planned duration | P95 planned duration |
| --- | ---: | ---: | ---: | ---: |
| bull | +3.012 | 24.18 | 128 | 212 |
| bear | -3.919 | 32.28 | 109 | 194 |
| range | -0.021 | 15.24 | 148 | 191 |
| chaotic | +0.572 | 61.24 | 43 | 83 |

Reference artifacts generated under `/data/workspace/nhsmm-trading-regimes-v1`:

- `trading_regimes.npz`: SHA-256 `370e6e9e87e52d779ea208111eeb2e03b0bf21f8b7e88859fcc8c19ac2739fb4`
- `trading_regimes.csv.gz`: SHA-256 `fd9732c7723622987098fa497da79286034bb04b62c5657c2f75b1393e35ebf2`
- `manifest.json`: episode ledger and full generator contract; 693 episodes total.
