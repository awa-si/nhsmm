# General-context shared-duration ablation — 2026-09-27

Local independent reference development ablation; no CI.

Tested transition-context parameters shared across the duration axis instead of independent `[H,K,D,K]` context coefficients. Seeds `641..643`, high-data configuration.

| Scenario | Accuracy | ARI | Transition corr | Transition MAE | Context amplitude |
|---|---:|---:|---:|---:|---:|
| 2D strong | 1.000 | 1.000 | 0.953 | 0.066 | 0.892 |
| 2D moderate | 1.000 | 1.000 | 0.853 | 0.097 | 0.672 |
| 2D null | 1.000 | 1.000 | n/a | 0.086 | 0.377 |

**Result: REJECTED.** Active effects remain recoverable, but 2D-null spurious amplitude worsens to median `0.377`. Purely sharing/reducing transition-context capacity across duration does not solve the null-control problem.

Next mechanism should target **evidence-dependent context selection/gating**, not uniform shrinkage (L2) and not simple duration-axis tying.
