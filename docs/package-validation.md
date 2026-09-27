# Package validation: duration and latent-state recovery

## Scope

This document records package-level empirical evidence for NHSMM training under controlled synthetic ground truth. It is independent of Nautilus, market data, trading targets, and downstream supervised models.

The claims here are intentionally narrow: they validate package mechanisms under known generators, not usefulness in an external domain.

## Context-conditioned duration recovery

The maintained duration benchmark uses two latent states and an observable context that changes the episode-duration distribution. Training uses a causal `DefaultEncoder`, `K=2`, `D=30`, `max_iter=40`, one initialization per seed, and separately generated evaluation sequences.

The primary diagnostic is

`mean P(end within 5 | short-duration context) - mean P(end within 5 | long-duration context)`.

A positive value is the expected direction. State occupancy rejects trivial collapse. The null-context scenario requires both absolute median and absolute mean gaps to remain at or below `0.015`.

Latest regression on seeds 201-215 after the restart/K-Means training changes, with the default `emission_init_mode="spread"`:

| Scenario | Positive gap | Gap >= 0.02 | Median gap | Mean gap | Non-collapsed |
| --- | ---: | ---: | ---: | ---: | ---: |
| strong, duration means 6 vs 18 | 14/15 | 14/15 | 0.09954 | 0.09166 | 15/15 |
| moderate, duration means 9 vs 15 | 13/15 | 9/15 | 0.02434 | 0.03218 | 15/15 |
| null, duration means 12 vs 12 | 9/15 | 7/15 | 0.01370 | 0.01184 | 15/15 |

Acceptance result: **PASS** for strong context, moderate context, and the null-context no-spurious-separation gate.

## Permutation-invariant latent-state recovery

The state benchmark uses `K=3`, three observed coordinates, explicit-duration state episodes, and state-dependent Gaussian means. State labels are evaluated only up to permutation. Training uses `emission_init_mode="kmeans"`, `max_iter=40`, and one initialization per data seed; evaluation uses separately generated sequences.

Predefined gates:

- strong separation: median matched accuracy >= `0.90`, median ARI >= `0.80`, at least 14/15 non-collapsed;
- moderate separation: median matched accuracy >= `0.70`, median ARI >= `0.40`, at least 14/15 non-collapsed;
- null separation: median matched accuracy <= `0.45` and absolute median ARI <= `0.10`.

Latest acceptance on seeds 301-315:

| Scenario | Separation | Median matched accuracy | Median ARI | Median effective states | Non-collapsed |
| --- | ---: | ---: | ---: | ---: | ---: |
| strong | 4.0 | 0.99667 | 0.99018 | 2.99070 | 15/15 |
| moderate | 2.0 | 0.93000 | 0.80182 | 2.99107 | 15/15 |
| null | 0.0 | 0.36333 | 0.00000 | 1.16304 | descriptive |

Acceptance result: **PASS** for all three scenarios. The null result is expected to collapse or partition arbitrarily because no state-identifying emission signal exists; the relevant guard is that matched accuracy and ARI stay at chance/no-signal levels.

### Initialization finding

The previous data-independent `spread` initialization could enter persistent two-state local minima on otherwise strongly identifiable three-state synthetic data. More iterations and marginal state-usage KL regularization did not reliably rescue those seeds. A training-time K-Means++/Lloyd initializer recovered all three states robustly in the controlled benchmark.

K-Means is therefore exposed as an **explicit opt-in initializer**, not the global default. Making it the global default changed the established duration null-control behavior, while retaining `spread` preserved the duration acceptance gate.

## Restart semantics

`n_init` is now treated as restart semantics rather than iterative warm-starting. Each run receives freshly initialized distributions and the encoder is restored to its pre-optimization baseline. Best-run parameters are deep snapshots and include `duration_logits_bias`, so later restarts cannot mutate the saved candidate by reference.

## Interpretation

Current evidence supports two package-level claims under controlled generators:

1. with the default `spread` initializer, the causal NHSMM recovers context-conditioned duration/hazard ordering without spurious null-context separation;
2. with explicit `emission_init_mode="kmeans"`, the model robustly recovers identifiable latent states up to permutation while remaining at chance under null separation.

It does **not** establish meaningful state semantics, transition-context recovery, or downstream predictive value in an external domain.

## Reproduction

```bash
python scripts/validate_duration_context.py --output /tmp/nhsmm-duration-context.json
python scripts/validate_state_recovery.py --scenario all --output /tmp/nhsmm-state-recovery.json
```
