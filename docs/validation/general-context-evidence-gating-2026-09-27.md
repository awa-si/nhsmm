# General-context evidence gating — 2026-09-27

This record preserves the current local research state for multidimensional causal transition context selection. It is independent reference evidence, not package-level acceptance. No CI was used.

## Background

Unregularized high-data 2D transition-context recovery preserves state identity and learns active strong/moderate context effects, but the null case can retain spurious context modulation. Global L2 shrinkage and shared-duration/tied-context parameterizations were both rejected in confirmatory experiments.

## Held-out evidence selection

The current mechanism fits context and null-context candidates on the same training partitions and selects the context branch from held-out likelihood evidence aggregated over cross-fit folds.

### Strict gate (`tau = 0`) — dev seeds 671..675

| Scenario | Selected | Median ΔLL / boundary | Median corr | Median MAE | Median amplitude |
| --- | ---: | ---: | ---: | ---: | ---: |
| strong | 5/5 | +0.0961 | 0.942 | 0.095 | 0.677 |
| moderate | 0/5 | -0.0572 | ~0 | 0.153 | 0 |
| null | 0/5 | -0.0995 | 0 | 0.045 | 0 |

Interpretation: excellent null control, inadequate power for moderate effects.

### Relaxed gate (`tau = -0.08`) — fresh confirmatory seeds 681..685

| Scenario | Selected | Median ΔLL / boundary | Median corr | Median MAE | Median amplitude |
| --- | ---: | ---: | ---: | ---: | ---: |
| strong | 5/5 | +0.0883 | 0.951 | 0.102 | 0.673 |
| moderate | 5/5 | -0.0573 | 0.780 | 0.089 | 0.425 |
| null | 1/5 | -0.1106 | 0 | 0.047 | 0 |

The single null false positive is seed 683. Its fold ΔLL values are `[-0.0357, -0.0895, -0.2142, -0.0464, -0.0464]`; none are positive, but the median exceeds `-0.08`. This demonstrates that the relaxed median-only rule is insufficient.

## Other evidence-gate variants already tested

- Refined held-out variant on seeds 691..695: strong 5/5, moderate 1/5, null 0/5 — rejected for low moderate power.
- Paired variant on seeds 701..705: strong 3/5, moderate 1/5, null 0/5 — rejected for low active power.

## Current conclusion

Evidence-dependent selection remains preferable to global parameter shrinkage because it can set the null branch exactly to zero without biasing accepted active models. The unresolved problem is selection power/calibration, not state recovery or active-model fit quality.

## Next frozen design task

Use the existing dev cross-fit results to define one additional fold-stability condition alongside the relaxed `tau=-0.08` criterion. The condition must be chosen before looking at new confirmatory seeds. Acceptance requires:

- strong selected 5/5;
- moderate selected at least 4/5;
- null selected 0/5;
- accepted active-model median transition correlation >= 0.70;
- accepted active-model median MAE <= 0.12.

No further threshold rescue is allowed after the next confirmatory run.
