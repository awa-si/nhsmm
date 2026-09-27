# Multi-component component-matched package confirmation — 2026-09-27

Source package core: `awa-si/nhsmm` `develop` pinned at `efd4826ddc48586c47debf075d1c6332596cb0de`. No CI.

All component-isolated A/B/C generators are matched to the model ablation: omitted mechanisms are nulled in both the model and synthetic ground truth. Full-joint D is unchanged. Frozen seeds: `501..515`.

| Scenario | Ablation | Accuracy | ARI | Duration gap | Transition MAE | Transition corr | Null transition Δ | Noncollapsed | Result |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| strong | A | 1.000 | 1.000 | 0.000 | 0.189 | 0.000 | 0.000 | 15/15 | PASS |
| strong | B | 1.000 | 1.000 | 4.426 | 0.163 | 0.000 | 0.015 | 15/15 | PASS |
| strong | C | 1.000 | 1.000 | 0.000 | 0.184 | 0.996 | 0.270 | 15/15 | PASS |
| strong | D | 1.000 | 1.000 | 4.383 | 0.169 | 0.997 | 0.306 | 15/15 | PASS |
| moderate | A | 1.000 | 1.000 | 0.000 | 0.189 | 0.000 | 0.000 | 15/15 | PASS |
| moderate | B | 1.000 | 1.000 | 2.709 | 0.168 | 0.000 | 0.010 | 15/15 | PASS |
| moderate | C | 1.000 | 1.000 | 0.000 | 0.162 | 0.963 | 0.136 | 15/15 | PASS |
| moderate | D | 1.000 | 1.000 | 2.273 | 0.132 | 0.938 | 0.153 | 15/15 | PASS |
| null | A | 1.000 | 1.000 | 0.000 | 0.189 | 0.000 | 0.000 | 15/15 | PASS |
| null | B | 1.000 | 1.000 | -0.125 | 0.176 | 0.000 | 0.005 | 15/15 | PASS |
| null | C | 1.000 | 1.000 | 0.000 | 0.155 | 0.000 | 0.044 | 13/15 | PASS |
| null | D | 1.000 | 1.000 | -0.116 | 0.130 | 0.000 | 0.040 | 13/15 | PASS |

**Overall: PACKAGE-CORE PASS — 12/12 frozen scenario × ablation cells pass.**

Key interpretation: the earlier isolated-C failure was a benchmark misspecification caused by retaining duration-context ground truth while disabling duration context in the model. After component matching, strong C passes with median transition MAE `0.184` and correlation `0.996`; moderate C passes with MAE `0.162` and correlation `0.963`. Null C/D remain below the frozen spurious transition-delta gate (`0.044` / `0.040`) and retain the required `13/15` noncollapsed runs.

No package objective, topology, or frozen metric threshold was changed to obtain this result.
