# General-context shared-linear + effect-floor confirmatory — 2026-09-27

Status: **REJECTED / confirmatory FAIL**.

This candidate was evaluated after the previously accepted package-core general-context gate remained PARTIAL on moderate 2D transition-shape correlation. The candidate changes only the transition-context parameterization to a duration-shared linear `K×K` effect and adds an evidence-selection floor; it does not modify the causal HSMM recursion.

## Frozen selection rule

- held-out evidence: `delta_ll_per_boundary > -0.07`
- pre-refine functional context amplitude: `>= 0.25`
- effect floor chosen only on development seeds `741..745`
- confirmatory seeds: `751..755`
- no threshold changes after the confirmatory run began

## Confirmatory result

| Scenario | Selected | Accuracy | ARI | Median corr | Median MAE | Result |
|---|---:|---:|---:|---:|---:|---|
| strong | 5/5 | 1.000 | 1.000 | 0.960 | 0.061 | PASS |
| moderate | 3/5 | 1.000 | 1.000 | 0.814 | 0.103 | **FAIL** |
| null | 0/5 | 1.000 | 1.000 | 0.000 | 0.017 | PASS |

Moderate fails because only **3/5** fresh seeds are selected; the frozen requirement is `>=4/5`. Strong remains 5/5 and null remains 0/5. State recovery remains effectively perfect. The candidate therefore improves transition-shape fidelity when selected but does not provide sufficient moderate-effect detection power.

## Interpretation

The open package-core deficiency remains narrow: robust detection/recovery of moderate multidimensional transition-context effects. This candidate is rejected; do not lower the evidence threshold or effect floor post hoc. The next mechanism should improve moderate evidence power while preserving the 0/5 null-selection result.
