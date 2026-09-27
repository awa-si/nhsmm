# NHSMM State

Current package readiness and next package-level research boundary. Detailed semantics: [`model.md`](model.md). Controlled evidence: [`package-validation.md`](package-validation.md). Verification policy: [`testing.md`](testing.md).

## Current status

**Phase:** duration-context, latent-state, and transition-context recovery baselines verified. **Multi-component identifiability remains provisionally accepted.** General-context robustness remains **provisionally accepted at reference level** using the frozen detect→refine mechanism. **Package confirmation is active and has now started against the actual package core locally.**

The previously missing package-materialization boundary has been partially removed by reconstructing the exact dependency closure needed for the validator from SHA-pinned `develop` sources into a sparse local workspace. The real package training/filtering path imports successfully with `K=3`, `D=24`, `causal=True`. No CI is used for this research loop.

A first real-package A/B/C/D smoke on one strong seed is numerically stable: state recovery is `1.000/1.000` accuracy/ARI in all four ablations, the learned duration and transition effects have the correct direction, and full-joint D is inside the strong transition-MAE gate. The transition-only C case on this single smoke seed has MAE `0.225`, slightly above the production median gate; this is not a gate failure because the production decision is median-based across the frozen multi-seed set. The displayed noncollapsed failure in the 1-seed smoke is likewise not interpretable against the production `>=13/15` gate. **No package-level PASS has been declared yet.** The next step is the unchanged 3-seed package run, followed by the full frozen gate if stable.

General-context robustness exposed one specific defect in the ungated reference: multidimensional null context could retain spurious transition modulation despite perfect state recovery. Global L2 shrinkage and shared-duration/tied context were rejected. The accepted reference design separates detection from estimation: paired cross-fitted held-out evidence is computed on the same unrefined fitted model with transition context on vs off; context is selected at median ΔLL/boundary `> -0.07`; only selected context is then refined for 20 `tctx`-only steps. On fresh confirmatory seeds 731..735 this selected strong 5/5, moderate 5/5, null 0/5 and all per-seed recovery/null gates passed. This is a **PROVISIONAL PASS**, not package-level acceptance.

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

**Status:** PROVISIONAL PASS. Package-level confirmation is now in progress against the actual package core.

### Package confirmation — in progress

Current pinned package head for the local sparse confirmation workspace: `efd4826ddc48586c47debf075d1c6332596cb0de`.

Real package components materialized for the validator include the package configuration, context/router, convergence, distributions, causal filtering, and the canonical `models/base.py` + `models/training.py` path. The first one-seed strong A/B/C/D smoke completed with:

- A state-only: matched accuracy/ARI `1.000/1.000`;
- B state + duration: matched accuracy/ARI `1.000/1.000`, duration effect direction correct;
- C state + transition: matched accuracy/ARI `1.000/1.000`, transition effect direction correct, single-seed MAE `0.225`;
- D full joint: matched accuracy/ARI `1.000/1.000`, transition MAE inside the strong gate.

**Interpretation:** smoke PASS for numerical/package-path viability only. This does not promote the provisional research result. The unchanged 3-seed run is next; only then may the full 15-seed gate be executed and evaluated.

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

The full discovered pytest suite was not rerun because the local workspace is sparse. No GitHub Actions run is used for this research loop. The current package confirmation is intentionally scoped to the exact dependency closure required by the multi-component validator.

## Package boundary

These controlled results establish mechanism recovery under known synthetic ground truth. They do not establish domain semantics or downstream predictive value for any consumer.

## Next research slices

1. **Package confirmation:** run the unchanged 3-seed multi-component package gate locally against the materialized package core; if stable, execute the full frozen 15-seed gate and either promote to package-level PASS or reopen the deficiency.
2. **General-context package confirmation:** after multi-component package confirmation, translate the frozen detect→refine reference gate to the actual package implementation and run it locally.
3. **API/design translation:** only after package confirmation, decide whether evidence selection belongs inside package training, a validation utility, or consumer orchestration; do not promote the reference mechanism blindly into core model semantics.
4. Change objectives or parameterizations only if the controlled package-level gates expose a reproducible deficiency.
