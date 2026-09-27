# NHSMM State

Current package readiness and next package-level research boundary. Detailed semantics: [`model.md`](model.md). Controlled evidence: [`package-validation.md`](package-validation.md). Verification policy: [`testing.md`](testing.md).

## Current status

**Phase:** duration-context, latent-state, and transition-context recovery baselines verified. **Multi-component identifiability full-joint D is now confirmed against the package core locally.** General-context robustness remains **provisionally accepted at reference level** using the frozen detect→refine mechanism.

Package-core confirmation used a SHA-pinned sparse local workspace reconstructed from `develop` sources. The exercised paths were canonical training, causal likelihood recursion, duration/transition/emission distributions, and causal filtering. Import/encoder glue was inert because every validation path supplies explicit external context. No CI is used for this research loop.

The full-joint D gate passed across all frozen seeds `501..515` in strong, moderate, and null scenarios. Strong and moderate retained perfect state recovery and passed duration/transition recovery gates; null retained no material spurious duration/transition context. The transition-only C ablation still shows an absolute transition-MAE failure under the old generator because the model disables duration context while the data still contains duration-context effects; the package legally absorbs some omitted duration structure through self-boundary transitions. That C result is classified as an ablation-design misspecification, not a full-joint identifiability failure. A/B/C diagnostics must therefore be rerun with component-matched generators while D remains unchanged.

General-context robustness exposed one specific defect in the ungated reference: multidimensional null context could retain spurious transition modulation despite perfect state recovery. Global L2 shrinkage and shared-duration/tied context were rejected. The accepted reference design separates detection from estimation: paired cross-fitted held-out evidence is computed on the same unrefined fitted model with transition context on vs off; context is selected at median ΔLL/boundary `> -0.07`; only selected context is then refined for 20 `tctx`-only steps. On fresh confirmatory seeds `731..735` this selected strong 5/5, moderate 5/5, null 0/5 and all per-seed recovery/null gates passed. This is a **PROVISIONAL PASS**, not package-level acceptance.

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

### Multi-component identifiability — package-core confirmation

The corrected independent local causal-HSMM reference first established structural recoverability. Package-core confirmation then ran the frozen full-joint D gate with `K=3`, `D=24`, seeds `501..515`, ergodic topology, kmeans emission initialization, and transition refinement where enabled.

| Scenario | Accuracy | ARI | Duration gap | Transition corr | Transition MAE | Null transition Δ | Noncollapsed | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| strong | 1.000 | 1.000 | 4.383 | 0.997 | 0.169 | 0.306 | 15/15 | PASS |
| moderate | 1.000 | 1.000 | 2.273 | 0.938 | 0.132 | 0.153 | 15/15 | PASS |
| null | 1.000 | 1.000 | -0.116 | 0.000 | 0.130 | 0.040 | 13/15 | PASS |

See [`validation/multicomponent-package-core-confirmation-2026-09-27.md`](validation/multicomponent-package-core-confirmation-2026-09-27.md).

**Status:** PACKAGE-CORE PASS for full-joint multi-component identifiability. Component-isolated A/B/C diagnostics remain to be rerun with component-matched generators; this does not reopen D.

### General-context robustness — current boundary

See [`validation/general-context-robustness-2026-09-27.md`](validation/general-context-robustness-2026-09-27.md), [`validation/general-context-shared-duration-ablation-2026-09-27.md`](validation/general-context-shared-duration-ablation-2026-09-27.md), and [`validation/general-context-evidence-gating-2026-09-27.md`](validation/general-context-evidence-gating-2026-09-27.md).

Current status:

- scalar strong/moderate/null: controlled recovery acceptable;
- ungated 2D null: spurious context modulation reproduced;
- L2 rescue: rejected;
- shared-duration/tied-context rescue: rejected;
- independent full-vs-null evidence gates: rejected for selection/recovery instability;
- **detect→refine paired cross-fit gate: PROVISIONAL PASS**;
- confirmatory seeds `731..735`: strong selected 5/5, moderate selected 5/5, null selected 0/5;
- strong median corr `0.960`, MAE `0.075`;
- moderate median corr `0.836`, MAE `0.074`;
- null context amplitude `0.000`, median MAE `0.034`.

**Status:** PROVISIONAL PASS at independent-reference level. Package-level acceptance still requires a local run against the actual `nhsmm` implementation.

## Verification limits

The full discovered pytest suite was not rerun because the local workspace is sparse. No GitHub Actions run is used for this research loop. Package-core confirmation is scoped to the exact external-context dependency closure used by the multi-component validator; encoder/import glue was inert.

## Package boundary

These controlled results establish mechanism recovery under known synthetic ground truth. They do not establish domain semantics or downstream predictive value for any consumer.

## Next research slices

1. **Component-matched A/B/C confirmation:** rerun isolated diagnostics with omitted generator mechanisms nulled; D remains frozen and already accepted at package-core level.
2. **General-context package confirmation:** translate the frozen detect→refine reference gate to the actual package implementation and run it locally.
3. **API/design translation:** only after general-context package confirmation, decide whether evidence selection belongs inside package training, a validation utility, or consumer orchestration; do not promote the reference mechanism blindly into core model semantics.
4. Change objectives or parameterizations only if the controlled package-level gates expose a reproducible deficiency.
