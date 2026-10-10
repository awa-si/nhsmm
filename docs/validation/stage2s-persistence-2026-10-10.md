# Stage 2S — Emission scaling × transition persistence (2026-10-10)

- Branch: `regime-v2`, dataset `data/reference/trading-regimes-v1/trading_regimes.npz`.
- Seeds 904, 905, 906; 30 training iterations each; variant A; 24 evaluations (2 emission scales × 4 diagonal transition bonuses × 3 seeds).
- Command: `PYTHONPATH=/workspace OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /work/env/nhsmm/bin/python scripts/diagnose_trading_persistence_2s.py --seeds 904 905 906 --max-iter 30 --output-dir /data/workspace/stage2`.
- Completed exit 0 in 1836.842 seconds; machine-readable per-seed results in `results/stage2s-A-{904,905,906}.json`.
- A first attempt failed after 639 seconds because the transition matrix has an additional duration axis. The diagonal bonus broadcasting was corrected to apply across this axis, and the entire experiment was rerun successfully.

## Findings

Emission scale 0.25 substantially reduces predicted state switching relative to 1.0, while additional diagonal transition bonuses further reduce switches. Accuracy and boundary F1 are not consistently improved by increased bonuses. In seed 905, for scale 0.25, accuracy drops from 0.48698 at bonus 0 to 0.47552 at bonus 2. This is **not** evidence of a generally optimal persistence bonus.

**Critical independence caveat:** Seeds 905 and 906 yield exactly identical accuracy, ARI, boundary F1, and switch counts at all eight parameter combinations, although their optimal label permutations differ. They must not be treated as independent corroborating replicates until initialization and deterministic behavior are investigated. No statistical significance or robust multi-seed improvement is claimed.

The intervention adds a diagonal log-bonus to the *episode-boundary transition* matrix, then renormalizes. A self-transition at an episode boundary is not equivalent to extending an episode; duration hazard is unchanged. Emission scaling is applied only at inference time, not during fitting. Results are exploratory, not a calibrated likelihood comparison or unbiased holdout assessment.

## Next step

Investigate seed collisions/initialization stability and design independent held-out confirmation before choosing production hyperparameters. Keep core model unchanged until evidence supports a modification.
