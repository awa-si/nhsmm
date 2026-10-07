# Bounded duration-tail extension contract

This document defines the proposed opt-in bounded-tail extension for NHSMM duration semantics. It does **not** change the currently accepted finite-support HSMM contract in `hsmm-mathematical-contract.md` until production implementation and equivalence gates are complete.

## Goal

The current finite duration support forces every episode to end by `max_duration=D`. The bounded-tail extension keeps runtime state bounded while allowing episodes longer than `D`.

The causal latent age representation remains width `D`:

```text
age bucket 1      -> exact age 1
...
age bucket D-1    -> exact age D-1
age bucket D      -> age >= D  (the D+ tail bucket)
```

`D=1` is valid: the single age bucket means `age >= 1`.

## Duration mass representation

For each state and causal information set, the duration component supplies a normalized mass vector of width `D`:

```text
q(1), ..., q(D-1), q_tail
```

where:

```text
q_tail = P(total_duration >= D)
```

The first `D-1` entries remain exact completed-duration masses. The final entry is **not** `P(total_duration=D)` unless the tail hazard is one.

A separate conditional tail end probability is required:

```text
h_tail in (0, 1]
```

Without `h_tail`, a bounded `D+` bucket is under-specified: the model would know how much probability reaches the tail but not when a tail episode ends.

## Causal hazard

For exact age `a < D`, define survival mass:

```text
S(a) = sum_{r=a..D-1} q(r) + q_tail
```

Then:

```text
h_end(a)  = q(a) / S(a)
h_cont(a) = 1 - h_end(a)
```

At the `D+` bucket:

```text
h_end(D+)  = h_tail
h_cont(D+) = 1 - h_tail
```

Continuation from `D+` remains in `D+`; it does not create an unbounded age tensor.

For `h_tail=1`, the tail terminates immediately on its first boundary and the model reduces to the current hard finite-support behavior when `q_tail` is interpreted as the current exact `q(D)` mass.

## Implied total-duration distribution

For `d < D`:

```text
P(total_duration=d) = q(d)
```

For `d >= D`:

```text
P(total_duration=d)
  = q_tail * (1-h_tail)^(d-D) * h_tail
```

Therefore the tail is geometric conditional on reaching `D`. This is a deliberate bounded-state approximation, not a claim that real regime tails are geometrically distributed.

The implied distribution is normalized whenever the mass vector is normalized and `0 < h_tail <= 1`.

## Boundary transition semantics

For exact ages `1..D-1`, duration-dependent transition index `a-1` keeps its current meaning.

For any boundary while the episode is in `D+`, transition index `D-1` is used. Thus the last transition slice means:

```text
A(state_i, D+, state_j)
```

rather than a boundary occurring at exact age `D`.

A self-transition still starts a new episode and resets the next age bucket to `1`.

## Dynamic causal context

In the causal model, both the mass vector and `h_tail` are evaluated from the information set available at the current boundary. If context changes while an episode remains in `D+`, the next boundary may use a different tail hazard.

This preserves the existing dynamic-hazard interpretation: no future observation may influence whether the current boundary ends the episode.

## Survival and one-step forecast semantics

When tail mode is enabled, survival forecasting freezes the current duration mass and current tail hazard, consistent with the existing no-future-context forecast contract. For a path that reaches `D+`, each additional future boundary contributes another factor of `1-h_tail`. Therefore survival horizons may extend arbitrarily beyond `D` while runtime state remains bounded.

The one-step transition forecast uses the same tail-aware end/continue hazard as filtering. Mass that continues in `D+` remains in the same latent state; mass that ends in `D+` uses transition slice `D-1` and starts the next episode at age `1`.

`h_tail=1` must reproduce the previous finite-support survival and transition forecasts exactly.

## Non-causal interpretation

A non-causal completed segment of duration `d >= D` receives the implied tail duration mass:

```text
q_tail * (1-h_tail)^(d-D) * h_tail
```

A non-causal DP therefore cannot assume segment duration is bounded by `D`; for an observed sequence of length `T`, candidate completed segment durations are bounded by `T` instead. Production non-causal support is a later implementation step and must be independently checked against exhaustive enumeration before enablement.

## Non-causal production implementation

The opt-in non-causal forward and Viterbi paths now admit actual completed segment lengths up to the observed sequence length `T` while retaining only `D` duration buckets in dynamic-programming state. Actual durations `d >= D` use bucket `D+` for subsequent duration-dependent transitions.

For a completed tail segment, both likelihood and MAP scoring use the contract mass

```text
q_tail * (1-h_tail)^(d-D) * h_tail
```

Viterbi retains the actual winning segment length in its backpointer even when multiple lengths collapse into the same `D+` DP bucket. This is required to reconstruct the state path correctly. The implementation is checked against independent exhaustive enumeration over all sequence compositions.

## Compatibility invariants

Before production enablement, the extension must satisfy:

1. `h_tail=1` reproduces current finite-support causal behavior exactly.
2. `D=1, h_tail=1` reproduces the current every-boundary-end HMM-equivalent case.
3. Tail continuation keeps the age state at `D+` rather than increasing tensor width.
4. `h_end + h_cont = 1` at every reachable exact or tail age.
5. The implied infinite total-duration PMF sums to one.
6. Current finite-support behavior remains the default until an explicit config opt-in is implemented.

## Implementation sequence

1. Freeze and independently test this mathematical representation.
2. Add an opt-in configuration and duration/tail parameterization without changing default behavior.
3. Extend causal forward/filter/runtime and prove `h_tail=1` equivalence to the current implementation.
4. Extend survival/forecast semantics.
5. Extend non-causal likelihood/Viterbi with exhaustive-reference equivalence. **Complete.**
6. Re-run canonical package acceptance before considering the extension production-ready.
