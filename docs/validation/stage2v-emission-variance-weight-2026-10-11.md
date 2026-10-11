# Stage 2V.8: Gaussian variance encoding fix and emission-weight diagnostic

Date: 2026-10-11. Parent revision: `211abe44da3537d59f86a56917fbb77cf38226f6`.

## Verified initializer defect and correction

The Gaussian distribution uses `softplus(log_var)` as actual variance in `log_prob`, `forward` and distribution construction. The variance-aware kmeans initializer previously stored `log(target_variance)`. Consequently it initialized actual variance to `log(1 + target_variance)`, not the intended within-cluster variance. Targets 1 and 5 became 0.6931 and 1.7918.

The corrected initializer uses the stable inverse-softplus transform `v + log(-expm1(-v))` after clipping target variances to [0.05, 5]. Cluster assignments, means, singleton/empty-cluster global-variance fallback, parameter identities and the other initializer modes are unchanged. The existing parameter name and artifact representation remain intact. No global defaults change.

Functional regression coverage checks both the resulting Gaussian density against an independently written density formula and singleton/global-variance fallback. Two existing tests used `exp(log_var)` to inspect variance; they now inspect the covariance returned by the Gaussian emission. They had previously mirrored the erroneous parameter encoding rather than the implemented variance contract.

## Controlled pilot

Corrected v2 dataset SHA256 `16cab157954b32f11af274bd617d6811a3776e81eb926a3f1dae80c954d7a1f8`; chronological first 60% training and last 20% evaluation. Training-only standardization, eight training and four evaluation windows of 240 bars; four states, D=64 with tail; seeds 901,902,903; four joint iterations, two transition-refinement steps, variance-aware kmeans, lr=0.008. One old-encoding and one corrected-encoding model fit per seed. The old encoding is reproduced by a script-local subclass. Labels are used for evaluation only.

Predeclared inference-only emission weights: 1, 0.5, 0.25, 0.125. They multiply the corrected model's log-emission scores without changing fitted parameters, duration or transition distributions. All weights are reported; none is selected or promoted using these held-out labels. The weighted powers are not normalized Gaussian densities, so proper held-out likelihood is reported only for weight=1. These weights are not a new package API.

Hard runner timeout: 300 seconds. Observed elapsed: 76.884 seconds, exit 0, empty stderr.

| Median causal metric | Old encoding, 1 | Corrected, 1 | Corrected, 0.5 | Corrected, 0.25 | Corrected, 0.125 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Accuracy | 79.27% | 79.69% | 79.79% | 78.96% | 77.92% |
| ARI | 0.5600 | 0.5650 | 0.5679 | 0.5585 | 0.5368 |
| Boundary-F1 ±5 | 0.1842 | 0.1449 | 0.1961 | 0.1579 | 0.2500 |
| State-run duration error | 92.25% | 90.97% | 87.12% | 75.09% | 71.00% |
| State changes | 68 | 61 | 43 | 30 | 24 |
| Posterior entropy, nats | 0.0495 | 0.0503 | 0.0777 | 0.1215 | 0.1965 |
| Proper OOS LL/bar | -8.2983 | -8.0721 | n/a | n/a | n/a |

There are eight true state changes in these evaluation windows. The corrected initializer improves proper held-out likelihood by 0.2262 nats/bar in the median comparison and slightly improves accuracy/run durations; boundary-F1 becomes worse. Thus mathematical/parameterization correctness improves without solving regime usefulness.

Reducing emission weight reduces fragmentation substantially, while eventually lowering accuracy. Even weight 0.125 still misses the boundary (0.45) and duration (50%) thresholds. This is a mechanism probe, not evidence of an accepted live decoder or of a normalized alternative generative model.

## Training feature structure and interpretation

Training-only pooled correlations include log_return_12 versus trend_12 at 0.8676, and ewma_vol_12 versus ewma_vol_48 at 0.8375. Within-window lag-1 correlations include log_return_12 0.9345, trend_48 0.9784, ewma_vol_12 0.9539 and ewma_vol_48 0.9924. These values are common to the three seeds because windows/preprocessing are fixed.

Pooled dependence can arise from regime mixtures, overlapping feature definitions or within-regime dynamics. It does not prove conditional residual dependence or quantify effective sample size. Together with the score-weight intervention, it motivates investigating emission calibration/feature dependence on a separate controlled protocol, rather than silently changing duration priors or package defaults.

Reproduce with the retained corrected v2 data:
```bash
PYTHONPATH=/workspace /work/env/nhsmm/bin/python scripts/diagnose_trading_emission_weight.py --dataset /data/workspace/nhsmm-trading-regimes-v2/trading_regimes.npz --output /data/workspace/stage2/emission-variance-weight-v2-2026-10-11.json
```

The diagnostic context manager restores the original sequence builder. A test verifies parameter preservation, normalized state posteriors, exact unweighted equivalence, and causal prefix invariance for every predeclared weight.

## Remaining acceptance

This is a short pilot with only eight true boundaries, not full corrected-v2 recovery acceptance. Full recovery and the previously timed-out full duration-tail empirical gate remain open. No score weight or new initializer default is adopted from the pilot.

## Verification

Final full pytest suite: 747 passed, 3 CUDA-only skips; runner elapsed 102.056 seconds with a 300-second limit. The two new variance checks fail as expected under the diagnostic legacy encoder. Changed Python files pass Ruff and Black. Focused initializer/diagnostic checks: eight passed.
