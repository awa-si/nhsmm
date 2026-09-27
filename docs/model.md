# NHSMM Model Contract

This document owns the human-facing model and runtime contract for the current repository state. Mathematical and engineering invariants used by repository agents live in [`agent-domain.md`](agent-domain.md).

## Scope

NHSMM is a context-aware explicit-duration latent-state sequence model implemented in PyTorch. The canonical model separates:

- initial-state distribution;
- duration distribution;
- transition distribution;
- emission distribution;
- optional neural context encoding.

The public construction path is configuration-driven:

```python
from nhsmm import ModelConfig, NHSMM

config = ModelConfig(n_states=3, n_features=5, max_duration=35)
model = NHSMM(config=config)
model.initialize_distributions()
```

Historical `HSMM`, `NeuralHSMM`, `GaussianHSMM`, and older constructor APIs are not canonical contracts.

## Training contract

`NHSMM.optimize()` optimizes every trainable model parameter returned by `model.parameters()`. This includes the context encoder as well as initial, duration, transition, emission, and duration-bias parameters. A context encoder that participates in the likelihood graph must not remain fixed at random initialization during normal optimization.

The default maximum iteration budget is `40`. Convergence stopping and scheduling remain configurable, so this is a maximum budget rather than a requirement to execute exactly 40 updates. The default was raised after controlled package-level duration-context recovery showed that short budgets could learn the correct duration direction weakly but fail multi-seed hazard robustness, while the unchanged 40-iteration model passed the package acceptance gate.

The empirical package evidence and its limits are recorded in [`package-validation.md`](package-validation.md).

## Duration semantics

Duration index `i` denotes total duration `i + 1`. Episode-age index `i` likewise denotes age `i + 1`.

For `causal=True`, the model uses dynamic causal boundary-time hazard semantics: information available at `F_t` determines the duration/end hazard used for boundary `t -> t+1`. The duration law is not frozen at episode start.

The causal filter posterior is represented over `(latent_state, episode_age)`. Continuation advances age without a transition; a transition occurs only after an episode boundary. A self-transition starts a new episode at age one.

Retrospective/non-causal segment inference is a separate path and must not be interpreted as an online filtered state estimate.

## Causal runtime

`HSMMFilterRuntime` provides one-observation-at-a-time causal inference. The canonical `DefaultEncoder` carries bounded incremental CNN/LSTM state so accepted observations are encoded once rather than by repeatedly re-encoding the full prefix.

Runtime outputs include:

- current latent-state posterior;
- current state-age posterior;
- one-step active-episode end probability;
- configurable-horizon survival/end-within-horizon probabilities;
- episode-boundary and next-episode latent-state transition quantities.

`expected remaining duration` is intentionally not a primary runtime output because under dynamic causal hazard semantics it requires an additional frozen-future-hazard assumption.

## Inference preparation

Production inference uses explicit preparation/loading helpers. Inference preparation requires initialized distributions, rejects non-finite parameters, switches to eval mode, and freezes parameters by default.

Strict state loading is required. Callers may require `causal=True` when constructing an inference model.

## Artifact contract

Artifact v1 persists the resolved `ModelConfig`, canonical `DefaultEncoder` metadata, redundant schema dimensions/mode metadata, and full model/distribution `state_dict`.

Loading is fail-closed for unsupported versions, incompatible schema/config/encoder metadata, malformed state mappings, and strict state-load mismatches. Artifact v1 supports the reconstructible canonical `DefaultEncoder`; arbitrary custom encoders are not serialized as a production contract.

## Evaluation boundary

NHSMM latent states remain semantically neutral unless an external evaluation explicitly defines a mapping. Package-level synthetic validation establishes recovery of known context-conditioned duration structure; it does not establish downstream utility in any particular domain.

External evaluation must use causally available observations and appropriate out-of-sample data. In-sample likelihood, state-plot plausibility, or semantic state matching alone are not acceptance criteria.

## Verification

Maintained tests use standard pytest discovery under `tests/test_*.py`. See [`testing.md`](testing.md) for the verification contract, [`package-validation.md`](package-validation.md) for controlled package-level empirical evidence, and [`state.md`](state.md) for current repository readiness.
