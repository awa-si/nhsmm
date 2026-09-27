# General-context evidence gating — 2026-09-27

This record preserves the local research result for multidimensional causal transition-context selection. It is **independent reference evidence, not package-level acceptance**. No CI was used.

## Problem

Unregularized high-data 2D transition-context recovery preserves state identity and learns strong/moderate context effects, but the null case can retain spurious context modulation. Global L2 shrinkage and shared-duration/tied-context parameterizations were rejected because they either over-shrank real effects or failed to suppress null modulation.

## Accepted reference mechanism: detect → refine

The accepted reference design separates **detection** from **estimation**:

1. Fit the unrefined full model.
2. On five independent held-out validation folds, evaluate the **same fitted model** twice: learned transition context on vs `tctx=0`.
3. Compute ΔLL normalized per observed validation boundary and aggregate by the median across folds.
4. Select transition context iff median ΔLL/boundary `> -0.07`.
5. If selected, freeze all non-context parameters and refine only `tctx` for 20 steps at LR `0.03`.
6. If not selected, disable transition context exactly.

This avoids nuisance variance from separately optimized null/full candidates and prevents refinement from contaminating the evidence used for selection.

Confirmatory spec SHA256: `632d120c60847ffbe180e5f62e0d1c3820a5332b2b69830b7c5f55a712426389`.

## Dev — seeds 721..725

| Scenario | Selected | Median ΔLL/boundary | Median corr | Median MAE | Median amplitude |
| --- | ---: | ---: | ---: | ---: | ---: |
| strong | 5/5 | +0.0866 | 0.953 | 0.078 | 0.663 |
| moderate | 5/5 | -0.0473 | 0.840 | 0.074 | 0.463 |
| null | 1/5 | -0.1178 | 0.000 | 0.080 | 0.000 |

All pre-specified dev gates passed. The mechanism and thresholds were then frozen before fresh confirmation.

## Confirmatory — seeds 731..735

| Scenario | Selected | Seed passes | Median ΔLL/boundary | Accuracy | ARI | Median corr | Median MAE | Median amplitude | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| strong | 5/5 | 5/5 | +0.0810 | 1.000 | 1.000 | 0.960 | 0.075 | 0.718 | PASS |
| moderate | 5/5 | 5/5 | -0.0568 | 1.000 | 1.000 | 0.836 | 0.074 | 0.485 | PASS |
| null | 0/5 | 5/5 | -0.1121 | 1.000 | 1.000 | 0.000 | 0.034 | 0.000 | PASS |

The confirmatory result satisfies the frozen gates: strong/moderate context is retained and accurately estimated, while null context is disabled exactly.

## Rejected variants

- Global L2 regularization: reduced null modulation but over-shrank active effects.
- Shared-duration/tied context: preserved active effects but worsened null spurious amplitude.
- Single-split held-out selection: too much split variance.
- Cross-fit selection using independently optimized full/null models: improved selection, but active recovery missed frozen confirmatory gates.
- Paired ablation after transition refinement: refinement contaminated selection and reduced active power.

## Status

**PROVISIONAL PASS** for the reference-level question: continuous/multidimensional causal transition context can be robustly selected and recovered when detection is separated from refinement.

This does not yet certify the actual `nhsmm` package. Package-level acceptance requires implementing/materializing the corresponding harness and running it locally against the real package implementation.
