# Frozen Trading gradient flow — 2026-10-15

Duration learning is connected to the production objective at the frozen artifact:
all256 PMF scores,256 gate parameters and4 tail hazards have finite, nonzero
gradients. An absent gradient path is not supported by this local check.

No optimizer updates were performed. All model state tensors remain exactly
unchanged. Only train and validation windows were extracted; no GT columns or
test windows enter this diagnostic. Oracle remains OFF. Core baseline95bc4b1;
runner6555f284b8e7119ca8d7021c0fff52e5ba0dabd2; preregistrationcb6cda6.

## Measured result

At target temperature1 on12x480 training observations:

| Group | Raw gradient L2 | Share of squared global norm | L2 after global cap5 |
| --- | ---: | ---: | ---: |
| Emission | 2026.200791 | 99.992415% | 4.999810 |
| Duration PMF | 14.081117 | 0.004829% | 0.034746 |
| Duration gate | 3.659818 | 0.000326% | 0.009031 |
| Tail hazard | 0.967921 | 0.000023% | 0.002388 |
| Transition | 8.645120 | 0.001820% | 0.021333 |
| Initial | 4.904966 | 0.000586% | 0.012103 |

Global raw norm2026.277634; predicted common clipping factor0.002467579.
This is the exact formula used by the global norm cap, calculated from measured
raw gradients, not an observed Adam update. At the final scheduled temperature
0.312590347, the emission share is99.927276% and clipping factor0.002495741;
Duration PMF L2 is45.514091. That pass evaluates terminal weights at the
historical schedule's last temperature, not the weights before update10.

| Temperature | Train LL / row | Validation LL / row |
| --- | ---: | ---: |
| 1.0 | -5.725563 | -5.541411 |
| 0.312590347 | -5.716420 | -5.535092 |

These are different temperature-dependent distributions, not a before/after
training comparison. Validation likelihood exceeding training likelihood on
these sampled windows is not a generalization guarantee.

The duration means remain31.225217,31.286151,31.857205,32.282510 bars.
D64 includes a geometric D+ bucket; it is not a hard maximum. Same-state episode
renewals remain allowed, so these means are not ground-truth regime-run means.

## Interpretation

The raw objective geometry is emission-dominated at this artifact. A small
common clipping factor alone does not prove that Adam starves duration updates:
its first/second moments can cancel common scaling, and historical moments were
not saved. Do not change clipping or introduce parameter-group learning rates
solely from these raw norm shares.

The baseline configuration used only10 maximum iterations, one restart, an
annealed objective, no scheduler and no convergence stopping. All trainable
groups are included by the production Adam parameter selection. Saved artifacts
contain weights/configuration, not optimizer state or historical update counts.
Neither convergence nor the cause of oversegmentation is established here.

## Validation and corrections

Nine focused tests, Ruff, AST and the actual32-row smoke pass. The material run
completed in66.343 seconds on Torch2.14.1+cpu. No CUDA or broad suite rerun;
the core estimator was unchanged.

Smoke first caught an incorrectly reused2-state/D8 summary helper; it was
replaced by the public gated Duration API with a4-state/D64 regression. The
first material attempt then exposed that load_artifact intentionally freezes
parameters for inference. Its empty gradient result is invalid, not evidence of
zero model gradients. The accepted runner reconstructs a normal training model,
strictly loads identical weights, verifies expected trainable flags in tests
and rejects missing groups/gradients. No statistical settings were selected
from the invalid result. The invalid report is preserved at
/work/checkpoints/trading-gradient-flow-invalid-frozen-loader-20261015.json.

Accepted machine evidence: trading-gradient-flow-20261015.json in this directory.
Runtime raw logs/results remain under /work/checkpoints/trading-gradient-flow*.
Historical five evidence-test failures from distribution-fixes-20261015.md are
not resolved by this diagnostic.

## Exact next milestone

Preregister a matched learning-budget study before fitting: fixed target
temperature, unchanged observations/scaler/features and optimizer settings,
observed actual update counts and saved optimizer/checkpoint states. Use
train/validation only, monitor parameter movement and objective progress, then
independent seed replication. Do not call a fresh Adam run a continuation of
the historical artifact without its absent optimizer moments. Preserve all
existing final-test evidence and require fresh outer data for a tuned candidate.
