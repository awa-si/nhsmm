# NHSMM Model Contract

This document owns the human-facing model and training/runtime contract. Mathematical and engineering invariants live in [`agent-domain.md`](agent-domain.md).

## Scope

NHSMM is a context-aware explicit-duration latent-state sequence model implemented in PyTorch. The canonical model separates initial-state, duration, transition, and emission distributions plus optional neural context encoding.

```python
from nhsmm import ModelConfig, NHSMM

config = ModelConfig(n_states=3, n_features=5, max_duration=35)
model = NHSMM(config=config)
model.initialize_distributions()
```

Historical constructors are not canonical contracts.

## Training contract

`NHSMM.optimize()` covers every trainable model parameter, including the context encoder. The default maximum joint-training budget is `40` iterations; convergence stopping and scheduling remain configurable.

`n_init` denotes independent optimization restarts. Each restart receives freshly initialized probabilistic distributions while the encoder is restored to the same pre-optimization baseline. Best-run state is stored as an independent deep snapshot, including `duration_logits_bias`.

The default emission initializer remains `spread`. `emission_init_mode="kmeans"` is an explicit opt-in training initializer for state-identifiability-sensitive workloads.

Distribution context networks use a hidden width of at least 16 units even for one-dimensional external context. This prevents `LayerNorm(1)` from erasing scalar context variation. Distribution base parameters are initialized independently of observed context; context modulation is learned through the likelihood objective.

Transition context modulation has a separate capacity bound, `transition_context_max_delta` (default `0.5`). An optional calibration stage is enabled with `transition_refine_steps > 0`. It freezes base transition logits and all non-transition parameters, then optimizes only `transition.context_net` and `transition.delta_scale` against the same exact marginal sequence likelihood at temperature `1.0`, without the duration-bias penalty or any supervised transition target. Refinement is disabled by default.

Controlled validation selected `transition_context_max_delta=1.5`, `transition_refine_steps=20`, and `transition_refine_lr=0.03` for the maintained transition-context acceptance benchmark. These are benchmark opt-ins, not global defaults.

## Duration semantics

Duration index `i` denotes total duration `i + 1`; episode-age index `i` denotes age `i + 1`.

For `causal=True`, information available at `F_t` determines the duration/end hazard for boundary `t -> t+1`. The causal filter posterior is represented over `(latent_state, episode_age)`. Continuation advances age without a transition; a transition occurs only after an episode boundary. A self-transition starts a new episode at age one.

Retrospective/non-causal segment inference is a separate path and must not be interpreted as an online filtered state estimate.

## Causal runtime

`HSMMFilterRuntime` provides one-observation-at-a-time causal inference. Runtime outputs include current latent-state posterior, state-age posterior, active-episode end probability, configurable-horizon survival/end probabilities, and episode-boundary/next-state transition quantities.

`expected remaining duration` is intentionally not a primary runtime output because dynamic causal hazard semantics require an additional frozen-future-hazard assumption for that quantity.

## Inference and artifacts

Production inference preparation requires initialized finite parameters, switches to eval mode, and freezes parameters by default. Strict state loading is required.

Artifact v1 persists resolved `ModelConfig`, canonical encoder metadata, schema/mode metadata, and the complete model/distribution `state_dict`. Loading is fail-closed for unsupported or incompatible artifacts.

## Evaluation boundary

Latent state IDs are semantically neutral. Package synthetic validation establishes recovery only against controlled known ground truth and only up to permutation where applicable. External consumers must perform causally valid out-of-sample evaluation for their own domain.

See [`package-validation.md`](package-validation.md) for empirical evidence, [`testing.md`](testing.md) for verification policy, and [`state.md`](state.md) for current readiness.
