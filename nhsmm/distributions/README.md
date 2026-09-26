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

`Initial.log_matrix(...)` returns latent-state logits with canonical shape:

```text
[B, T, K]
```

For a context-free distribution, singleton batch/time dimensions are introduced as needed.

The last axis is the latent-state axis.

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

Context modulation changes state-conditioned location parameters. Temperature is applied through the common modulation path and therefore affects context/base modulation consistently when supplied.

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

A `timestep` selects one causal context position where supported.

## Temperature

Temperature scaling is applied to finite trainable/context-modulated parameters before structural hard masks are imposed. This ordering preserves exact `-inf` support for impossible durations/transitions.

## Numerical and support rules

- Non-finite trainable/context-modulated parameters are rejected before normalization.
- Structural impossibilities use `-inf`, not merely very negative finite logits.
- Categorical probabilities are normalized over their declared final categorical axis.
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
