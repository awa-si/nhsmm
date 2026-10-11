# Stage 2V.8: duration-prior ablation and bounded-tail optimizer fix

Date: 2026-10-11. Parent revision: `7e8871d9e738b24ed2cab9134f93cce319516587`.

## Confirmed optimizer defect

The D+ tail end hazard is a directly optimized probability. Inference clamps it to [EPS, 1], but Adam previously could move the raw parameter outside that interval. Outside it, the clamp derivative is zero and later objectives cannot provide a restoring gradient through this parameter.

A controlled probe starts h at 1/128 and takes one Adam step with lr=0.008 and loss=h. The raw parameter becomes -0.00018749944865703583, effective h becomes approximately 1e-12, and the next gradient is exactly zero. This is a parameter-domain/training defect; it does not establish a recursion error or explain all Trading recovery failures.

The fix projects the built-in Duration tail parameter into [EPS, 1] immediately after each joint optimizer update, before scoring or snapshots. The inference clamp remains a guard. Direct probability storage, the exact h=1 compatibility boundary, parameter names/artifact representation and package defaults remain unchanged. Custom duration components can omit the optional projection hook.

Two regression tests exercise optimizer steps across both lower and upper boundaries, then reverse the objective direction and require the parameter to recover into the interior. Both fail when the projection hook is disabled in memory and pass with the fix. A prior-injection test verifies initialization after fresh restart distributions.

## Controlled duration experiment

Corrected v2 dataset SHA256 `16cab157954b32f11af274bd617d6811a3776e81eb926a3f1dae80c954d7a1f8`; seeds 901,902,903. Chronological first 60% training and last 20% evaluation; training-only standardization; 8 training and 4 evaluation windows of 240 bars. Emissions use kmeans_cluster_variance, transition refinement uses 2 steps. Labels are restricted to scoring.

Predeclared factorial arms:
- existing uniform duration with 4 or 12 joint iterations;
- common geometric episode mean 128 with 4 or 12 iterations.

The geometric prior is script-local and label-independent. Its PMF retains masses 1..D-1 and the remaining survival mass in D+, with a matching geometric tail hazard. It is applied after each restart initialization. Increasing max_iter also changes the existing annealing schedule, so the iteration axis is a training-budget/schedule comparison rather than a pure optimization-step comparison.

Both the original and projected runs use a hard 300-second runner timeout. Elapsed: 114.915 seconds before the fix, 134.230 seconds after the fix (the latter overlapped the full tests). Both complete with exit 0 and empty stderr. Run arrays and summaries are exactly equal before and after the fix: this short Trading experiment does not activate the identified clamp failure.

| Median causal metric | Uniform, 4 | Uniform, 12 | Geometric128, 4 | Geometric128, 12 |
| --- | ---: | ---: | ---: | ---: |
| Accuracy | 79.27% | 78.75% | 79.69% | 79.38% |
| ARI | 0.5600 | 0.5591 | 0.5685 | 0.5700 |
| Boundary-F1 ±5 | 0.1842 | 0.1600 | 0.1690 | 0.1846 |
| State-run duration error | 92.25% | 91.00% | 90.34% | 90.00% |
| OOS LL per bar | -8.2983 | -8.2428 | -8.3232 | -8.2685 |
| State changes | 68 | 67 | 63 | 56 |

Neither the longer schedule nor the longer initial episode scale resolves boundary/run-duration acceptance. The geometric prior slightly reduces state changes, but OOS likelihood is worse than the matched uniform arm. It is not promoted to a package option/default based on these results. The pilot has only 8 ground-truth state changes, limiting empirical conclusions.

The longer schedule moves learned episode medians down: seed901 uniform medians become 27–30 rather than 31–32; geometric medians become 64–80 rather than their initial 89. This suggests that simply initializing longer durations does not prevent the joint objective from preferring shorter episodes in this setup. Episode duration and observed same-state run length remain distinct because of self-transitions.

Reproduce the factorial diagnostic with the existing v2 data:
```bash
PYTHONPATH=/workspace /work/env/nhsmm/bin/python scripts/diagnose_trading_duration_prior.py --dataset /data/workspace/nhsmm-trading-regimes-v2/trading_regimes.npz --output /data/workspace/stage2/duration-prior-iteration-v2-projected-2026-10-11.json
```

## Verification and remaining gates

- Focused tests: 37 passed.
- Full pytest suite: 744 passed, 3 CUDA-only skips, 99.68 seconds test time / 127.893 seconds runner elapsed. One read-only pytest cache warning; no test failure.
- Both boundary regressions fail when projection is disabled, as expected.
- Ruff and Black checked on touched Python files.
- Full canonical duration-tail acceptance command was attempted with 3 workers and a 300-second timeout. It timed out with no result payload or stdout; runner teardown completed at 301.677 seconds. No acceptance result can be inferred. This gate remains open.
- Full corrected-v2 Trading recovery remains open; these experiments are short diagnostics.

Committed artifacts include both factorial reports, the independent gradient probe, and duration-tail timeout metadata. No claim of improved Trading recovery or passed full duration-tail acceptance is made.
