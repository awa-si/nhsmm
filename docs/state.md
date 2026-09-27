# NHSMM State

Current package readiness and next package-level research boundary. Detailed semantics: [`model.md`](model.md). Controlled evidence: [`package-validation.md`](package-validation.md). Verification policy: [`testing.md`](testing.md).

## Current status

**Phase:** duration-context, latent-state, and transition-context recovery baselines verified. **Multi-component identifiability remains provisionally accepted.** General-context robustness is now the active research boundary.

The corrected multi-component reference remains a provisional PASS only: package-level confirmation against the actual `nhsmm` implementation is still outstanding because the current connector does not provide repository materialization into the local runtime. No CI is used for this research loop.

General-context robustness has a narrower unresolved issue: scalar and active 2D causal transition contexts recover well, but a 2D null context can produce spurious transition modulation. Global L2 shrinkage and shared-duration context parameterization were both tested and rejected because they either over-shrank active effects or failed to remove null modulation.

The current best direction is **evidence-dependent context selection**. Cross-fit held-out likelihood cleanly separates strong from null context, but moderate effects sit near the decision boundary. With `tau=0`, dev selected strong 5/5, moderate 0/5, null 0/5. A relaxed `tau=-0.08` promoted moderate recovery but on fresh confirmatory seeds selected strong 5/5, moderate 5/5, and null 1/5. This is not accepted yet. The next step is a second fold-stability/evidence condition that removes the residual null false positive without sacrificing moderate power.

Package validation remains independent of Nautilus, market data, trading labels, and downstream supervised models.

## Verified training baseline

- [x] `NHSMM.optimize()` covers all trainable parameters, including the causal encoder.
- [x] Default joint-training maximum is `40` iterations.
- [x] `n_init` uses independent restarts; best-run snapshots are deep and include `duration_logits_bias`.
- [x] Default emission initialization remains `spread`; explicit `kmeans` is available for state-identifiability-sensitive training.
- [x] Scalar external context retains a distribution hidden width >=16, avoiding `LayerNorm(1)` collapse.
- [x] Transition context capacity is independently configurable.
- [x] Optional transition refinement updates only `transition.context_net + transition.delta_scale` against exact sequence likelihood; it is disabled by default.
- [x] No auxiliary supervised transition loss or alternative duration parameterization is required by the current controlled gates.

## Accepted / provisional evidence

### Duration context — seeds 201..215

| Scenario | Median gap | Mean gap | Non-collapsed | Result |
| --- | ---: | ---: | ---: | --- |
| strong | 0.09954 | 0.09166 | 15/15 | PASS |
| moderate | 0.02434 | 0.03218 | 15/15 | PASS |
| null | 0.01370 | 0.01184 | 15/15 | PASS |

### Latent state — seeds 301..315

| Scenario | Median accuracy | Median ARI | Non-collapsed | Result |
| --- | ---: | ---: | ---: | --- |
| strong | 0.99667 | 0.99018 | 15/15 | PASS |
| moderate | 0.93000 | 0.80182 | 15/15 | PASS |
| null | 0.36333 | 0.00000 | descriptive | PASS |

### Transition context — seeds 401..415

Validation config: `max_duration=1`, explicit binary external context, `transition_context_max_delta=1.5`, `transition_refine_steps=20`, `transition_refine_lr=0.03`.

| Scenario | Median MAE | Median KL | Delta correlation | Direction | Abs context delta | Non-collapsed | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| strong | 0.01782 | 0.00370 | 0.99999 | 1.000 | 0.47720 | 15/15 | PASS |
| moderate | 0.01833 | 0.00260 | 0.99194 | 1.000 | 0.16448 | 15/15 | PASS |
| null | 0.00524 | 0.00032 | 0.00000 | n/a | 3.04e-7 | 15/15 | PASS |

### Multi-component identifiability — provisional acceptance

The corrected independent local causal-HSMM reference used `K=3`, `D=24`, seeds `501..503`, explicit A/B/C/D ablations, and an ergodic topology. Strong and moderate full-joint cases retained matched accuracy/ARI `1.000`; null A-D retained state identity without material spurious context. Reintroducing the old `semi` support reproduced the collapse. See [`validation/multicomponent-reference-deep-dive-2026-09-27.md`](validation/multicomponent-reference-deep-dive-2026-09-27.md).

**Status:** PROVISIONAL PASS. Package-level acceptance still requires a corrected local v2 run against the actual package implementation.

### General-context robustness — current boundary

See [`validation/general-context-robustness-2026-09-27.md`](validation/general-context-robustness-2026-09-27.md), [`validation/general-context-shared-duration-ablation-2026-09-27.md`](validation/general-context-shared-duration-ablation-2026-09-27.md), and [`validation/general-context-evidence-gating-2026-09-27.md`](validation/general-context-evidence-gating-2026-09-27.md).

Current status:

- scalar strong/moderate/null: controlled recovery acceptable;
- 2D strong/moderate: controlled recovery acceptable;
- 2D null: unresolved without selection/gating;
- L2 rescue: rejected;
- shared-duration/tied-context rescue: rejected;
- cross-fit evidence gate `tau=0`: strong 5/5, moderate 0/5, null 0/5 on dev;
- relaxed cross-fit gate `tau=-0.08`: strong 5/5, moderate 5/5, null 1/5 on fresh confirmatory seeds;
- therefore evidence gating is promising but **not accepted** yet.

## Verification limits

The full discovered pytest suite was not rerun because the local workspace is sparse. No GitHub Actions run is used for this research loop. The general-context work is independent local reference evidence until corresponding package-level harnesses are materialized and run locally.

## Package boundary

These controlled results establish mechanism recovery under known synthetic ground truth. They do not establish domain semantics or downstream predictive value for any consumer.

## Next research slices

1. **Evidence-gate refinement:** add a pre-specified fold-stability condition to the relaxed cross-fit gate, chosen on dev data and frozen before fresh confirmation. Hard requirement: retain moderate power while returning null selection to zero.
2. **Package confirmation:** when local package materialization becomes available, run corrected v2 multi-component and general-context validators against the actual `nhsmm` implementation.
3. Change objectives or parameterizations only if the controlled package-level gates expose a reproducible deficiency.
