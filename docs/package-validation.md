# Package validation: duration, state, and transition-context recovery

## Scope

This document records controlled package-level empirical evidence. It is independent of Nautilus, market data, trading targets, and downstream supervised models. The claims are mechanism-specific and do not establish external-domain utility.

## Duration-context recovery

The duration benchmark uses two latent states and an observed context that changes episode-duration ground truth. Training uses causal NHSMM, `K=2`, `D=30`, `max_iter=40`, and independent evaluation sequences. The main diagnostic is the difference in mean probability of ending within five steps between short- and long-duration contexts. The null guard requires both absolute median and mean gaps to stay at or below `0.015`.

Seeds `201..215`, default `emission_init_mode="spread"`:

| Scenario | Positive gap | Gap >= 0.02 | Median gap | Mean gap | Non-collapsed |
| --- | ---: | ---: | ---: | ---: | ---: |
| strong, means 6 vs 18 | 14/15 | 14/15 | 0.09954 | 0.09166 | 15/15 |
| moderate, means 9 vs 15 | 13/15 | 9/15 | 0.02434 | 0.03218 | 15/15 |
| null, means 12 vs 12 | 9/15 | 7/15 | 0.01370 | 0.01184 | 15/15 |

Acceptance: **PASS**.

## Permutation-invariant latent-state recovery

The state benchmark uses `K=3`, three observed coordinates, explicit-duration episodes, and state-dependent Gaussian means. Labels are scored only up to permutation. The benchmark explicitly opts into `emission_init_mode="kmeans"`; the package default remains `spread`.

Predefined gates: strong median accuracy >= `0.90`, ARI >= `0.80`; moderate accuracy >= `0.70`, ARI >= `0.40`; null accuracy <= `0.45` and |median ARI| <= `0.10`; strong/moderate require at least 14/15 non-collapsed runs.

Seeds `301..315`:

| Scenario | Median accuracy | Median ARI | Median effective states | Non-collapsed |
| --- | ---: | ---: | ---: | ---: |
| strong | 0.99667 | 0.99018 | 2.99070 | 15/15 |
| moderate | 0.93000 | 0.80182 | 2.99107 | 15/15 |
| null | 0.36333 | 0.00000 | 1.16304 | descriptive |

Acceptance: **PASS**.

The earlier data-independent `spread` initialization could enter persistent two-state local minima on strongly identifiable synthetic data. More iterations and marginal state-usage regularization did not reliably rescue those seeds. Training-time K-Means++/Lloyd initialization did, so K-Means remains an explicit opt-in rather than the global default.

## Transition-context recovery

The transition benchmark isolates transition learning with `K=3`, strongly identifiable Gaussian emissions, `max_duration=1`, and a binary causal context `c_t` that selects the known transition law for boundary `t -> t+1`. State identity is aligned up to permutation before transition matrices are scored.

Two package deficiencies were isolated before final acceptance:

- one-dimensional external context inherited a one-unit distribution hidden layer, so `LayerNorm(1)` erased all context variation; distribution context networks now retain a minimum hidden width of 16;
- joint training learned transition-effect direction but underfit strong context amplitude. Increasing only clamp size, training budget, learning rate, initial `delta_scale`, or changing activation did not robustly solve it.

Component ablation showed that post-joint refinement needs only `transition.context_net + transition.delta_scale`; base transition logits need not move. The maintained benchmark therefore opts into:

- `transition_context_max_delta=1.5`;
- `transition_refine_steps=20`;
- `transition_refine_lr=0.03`;
- exact marginal sequence likelihood, temperature `1.0`, no duration-bias penalty;
- all non-refined parameters frozen.

Predefined gates: strong MAE <= `0.15`, delta correlation >= `0.70`, direction >= `0.80`; moderate MAE <= `0.18`, correlation >= `0.45`, direction >= `0.70`; null median absolute learned context delta <= `0.08`; strong/moderate require at least 14/15 non-collapsed runs.

Seeds `401..415`:

| Scenario | Median MAE | Median KL | Delta correlation | Direction | Abs context delta | Non-collapsed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| strong | 0.01782 | 0.00370 | 0.99999 | 1.000 | 0.47720 | 15/15 |
| moderate | 0.01833 | 0.00260 | 0.99194 | 1.000 | 0.16448 | 15/15 |
| null | 0.00524 | 0.00032 | 0.00000 | n/a | 3.04e-7 | 15/15 |

Acceptance: **PASS**. The null control remains far below the `0.08` spurious-effect guard.

## Interpretation

Current controlled evidence supports three narrow package claims: duration-context ordering is recoverable without spurious null separation; identifiable latent states are recoverable up to permutation with explicit K-Means initialization; and known context-conditioned transition laws are recoverable with explicit transition capacity/refinement while preserving a near-zero null effect.

These results do **not** establish domain semantics or downstream predictive value.

## Reproduction

```bash
python scripts/validate_duration_context.py --output /tmp/nhsmm-duration-context.json
python scripts/validate_state_recovery.py --scenario all --output /tmp/nhsmm-state-recovery.json
python scripts/validate_transition_context.py --scenario all --output /tmp/nhsmm-transition-context.json
```
