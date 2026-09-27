# Multi-component identifiability deep dive — 2026-09-27

Base: `develop` @ `6197da2a28f06c33b52b7642f5ec1105ff1cfd22`

This record is **local independent reference evidence**, not package-level acceptance. No CI was used for this deep-dive run. The actual `nhsmm` package v2 validator still needs to run locally once the package snapshot is materialized.

## Root cause

The original joint benchmark was invalid for three reasons:

1. The cyclic synthetic transition generator was evaluated with `transition_type="semi"`, while package `semi` support is self + next-state rather than arbitrary cyclic other-state transitions.
2. Duration-dependent transition tensors `[K_src, D, K_dst]` were reshaped into `K x K` for scoring, mixing duration and source-state axes.
3. Duration scoring ignored the trained `duration_logits_bias` soft gate used by the model forward path.

## Corrected reference results

| Scenario | Ablation | Topology | n | Accuracy | ARI | Effective states | Duration gap | Transition corr |
|---|---|---|---:|---:|---:|---:|---:|---:|
| moderate | A | ergodic | 3 | 1.000 | 1.000 | 2.967 | 0.000 | 0.000 |
| moderate | B | ergodic | 3 | 1.000 | 1.000 | 2.967 | 1.804 | 0.757 |
| moderate | C | ergodic | 3 | 1.000 | 1.000 | 2.967 | 0.000 | 0.764 |
| moderate | D | ergodic | 3 | 1.000 | 1.000 | 2.967 | 1.805 | 0.861 |
| null | A | ergodic | 3 | 1.000 | 1.000 | 2.991 | 0.000 | 0.000 |
| null | B | ergodic | 3 | 1.000 | 1.000 | 2.991 | 0.142 | 0.000 |
| null | C | ergodic | 3 | 1.000 | 1.000 | 2.991 | 0.000 | 0.000 |
| null | D | ergodic | 3 | 1.000 | 1.000 | 2.991 | 0.155 | 0.000 |
| strong | A | ergodic | 3 | 1.000 | 1.000 | 2.968 | 0.000 | 0.000 |
| strong | B | ergodic | 3 | 1.000 | 1.000 | 2.968 | 2.694 | 0.966 |
| strong | C | ergodic | 3 | 1.000 | 1.000 | 2.968 | 0.000 | 0.979 |
| strong | D | ergodic | 3 | 1.000 | 1.000 | 2.968 | 2.699 | 0.987 |
| strong | D | semi | 3 | 0.603 | 0.234 | 2.121 | 0.629 | 0.571 |

A = state-only, B = state + duration context, C = state + transition context, D = full joint duration + transition context.

## Interpretation

- All corrected `ergodic` A/B/C/D cases retain median state Accuracy/ARI = `1.0` across strong, moderate, and null scenarios.
- Strong D retains duration and transition signal simultaneously (`duration gap ≈ 2.699`, `transition corr ≈ 0.987`).
- Moderate D also retains both mechanisms (`duration gap ≈ 1.805`, `transition corr ≈ 0.861`).
- Null D preserves state recovery while learned context effects remain small (`duration gap ≈ 0.155`, transition correlation baseline `0`).
- Reapplying the old `semi` support to strong D drops median Accuracy to `0.603`, ARI to `0.234`, and effective states to `2.121`, reproducing the collapse pattern.

## Status

No NHSMM model-code change is justified by the original CI failure. The corrected package validator must still be run against the actual package locally before promoting this evidence to package acceptance.
