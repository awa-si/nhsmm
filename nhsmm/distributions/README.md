# NHSMM Distributions

`default.py` contains the canonical trainable distribution components used by the current NHSMM implementation:

- `Initial`: initial latent-state distribution.
- `Duration`: explicit episode-duration distribution.
- `Transition`: latent-state transition distribution at episode boundaries.
- `Emission`: observation likelihood by latent state.
- `Categorical` and `IndependentStudentT`: local distribution helpers used by the components above.

The file is responsible for distribution parameterization, context modulation, normalization, masking, temperature scaling, and output tensor shape. It does not define the full HSMM recursion; forward/Viterbi/filter/runtime code consumes these outputs.

## Canonical tensor contracts

Let:

- `B` = batch size
- `T` = sequence length
- `K` = number of latent states
- `D` = `max_duration`
- `F` = observation feature dimension
- `H` = context dimension

### Initial

`Initial.log_matrix(...)` returns normalized latent-state log probabilities with canonical shape:

```text
[B, T, K]
```

For a context-free distribution, singleton batch/time dimensions are introduced as needed.

The last axis is the latent-state axis. It is normalized with `log_softmax`, so adding a common offset to the underlying trainable logits does not change the initial-state distribution.

### Duration

`Duration.log_matrix(...)` returns normalized log probabilities:

```text
[B, T, K, D]
```

Normalization is over the last `D` axis.

Duration-axis convention:

```text
duration_index = d - 1
```

Therefore:

```text
index 0 -> total episode duration 1
index 1 -> total episode duration 2
...
index D-1 -> total episode duration D
```

This file establishes the representation convention. Forward, Viterbi, filtering, survival, and runtime code must preserve the same interpretation.

`Duration` supports hard masks with shape:

```text
[D]
[K, D]
```

Masked durations have exact zero support (`-inf` log probability before normalization).

`soft_dmax` belongs to `Duration`: it multiplicatively gates duration support before normalization and may be global `[D]` or state-dependent `[K,D]`.

### Transition

Without duration dependence:

```text
Transition.log_matrix(...) -> [B, T, K_src, K_dst]
```

With `max_duration=D`:

```text
Transition.log_matrix(...) -> [B, T, K_src, D, K_dst]
```

Normalization is always over `K_dst`, the final axis.

For the duration-dependent case, duration index `i` uses the same convention as `Duration`:

```text
i = d - 1
```

Transition probabilities are conditional on an episode boundary. Duration support is not applied through `Transition.soft_dmax`; a duration-only additive offset would cancel under destination-state softmax normalization. Duration gating therefore remains a `Duration` responsibility.

Transition topology:

- `ergodic`: all destination states allowed.
- `semi`: self-transition and next-state transition allowed.
- `left-to-right`: self and forward states only, consistently in both duration-independent and duration-dependent forms.

Structural and external transition masks are hard masks. Forbidden transitions have exact zero probability after softmax.

### Emission

For input observations:

```text
x: [B, T, F]
```

`Emission.log_prob(...)` returns:

```text
[B, T, K]
```

Supported emission families:

- diagonal Gaussian via `MultivariateNormal`
- independent Student-t via `IndependentStudentT`

Context modulation changes state-conditioned location parameters while preserving the learned absolute state locations. The context delta is centered across states so context does not introduce a shared location offset. Emission locations are not temperature-scaled; temperature applies to categorical logits.

## Context modulation

Components may be configured with `context_dim=None`, in which case they operate without a context network.

When context is present, accepted shapes are:

```text
[H]
[B, H]
[B, T, H]
[S, B, T, H]
```

The component projects and maps context into an additive parameter delta. The delta is bounded by `max_delta` after scaling by the trainable `delta_scale`.

Initialization establishes context-free base parameters. When `initialize(context=...)` is used, the supplied context is applied exactly once when constructing the returned distribution; it is not baked into the stored base parameters. Reinitialization updates existing registered parameters in place so optimizer and external parameter references remain valid.

A `timestep` selects one causal context position where supported.

## Causal duration-context contract

For `ModelConfig.causal=True`, the canonical interpretation is **dynamic causal hazard**, not an episode-start duration draw that is frozen for the lifetime of the episode.

At filtered timestep `t`, `Duration.log_matrix(...)` under the current information set `F_t` provides the duration law used to derive the end/continue hazard for the currently active episode age. The boundary decision `t -> t+1` must therefore use duration and transition quantities conditioned only on `F_t`.

Consequences:

- the duration law may change at later timesteps as causal context changes;
- consumers must not use `F_{t+1}` to decide whether the episode already ended between `t` and `t+1`;
- the runtime/filter contract is boundary-time causal: current context determines the next boundary hazard;
- a multi-horizon survival forecast may freeze the current `F_t` duration law across its forecast horizon, but that is a forecasting approximation and does not redefine the online update semantics;
- a causal forward/Viterbi implementation must be mathematically equivalent to this age/hazard interpretation rather than scoring a completed segment from context observed only at its endpoint.

For non-causal retrospective inference, a separate explicitly documented segment-scoring convention may be used; it must not be presented as causal filtering semantics.

## Temperature

Temperature scaling applies to categorical state/duration/transition logits before structural hard masks are imposed. Emission location parameters are not temperature-scaled. Temperature is an explicit finite scalar control greater than zero; the stored categorical temperature parameters are frozen and are not part of generic model optimization. This ordering preserves hard support for impossible durations/transitions.

## Numerical and support rules

- Non-finite trainable/context-modulated parameters are rejected before normalization.
- Structural masks must leave at least one valid outcome per categorical row; impossible outcomes use the package negative-infinity sentinel.
- Categorical probabilities are normalized over the final categorical axis; non-final category axes are rejected explicitly.
- Context-free construction is supported.
- `Categorical.rsample()` uses Gumbel-softmax and supports differentiable soft samples and straight-through hard samples.

## What `default.py` does not guarantee by itself

The distribution layer defines tensor meaning and support, but it does not prove that all consumers index duration identically. The following must be validated separately across the model stack:

- `forward()` segment duration indexing
- Viterbi duration indexing
- online filter age/duration indexing
- runtime boundary indexing
- survival/hazard horizon semantics

The canonical cross-module target is:

```text
duration index 0 == total duration 1
age index 0      == current episode age 1
```

Any consumer that deviates from this convention is a model-semantics bug rather than a distribution-layer alternative.
