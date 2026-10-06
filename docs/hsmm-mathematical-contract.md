# HSMM mathematical contract

This document freezes the model semantics used by the HSMM correctness research milestone. It describes the probability model implemented by the package; it is not an implementation guide for the independent reference oracle.

## Notation

For one sequence of length `T`:

- `z_t in {1,...,K}` is the latent state at observation timestep `t`;
- `a_t in {1,...,D}` is the one-based age of the currently active episode at `t`;
- `x_t` is the observation at `t`;
- `pi(k)` is the initial-state probability;
- `q_t(k,d)` is the normalized duration PMF over total duration `d in {1,...,D}` returned for state `k` at model timestep `t`;
- `A_t(i,a,j)` is the normalized probability of starting a new episode in state `j` when an episode in state `i` ends at age `a` on boundary `t -> t+1`;
- `e_t(k) = p(x_t | z_t=k, context_t)` is the emission probability.

Array duration index `r` is zero-based and represents one-based duration/age `r+1`.

The default `DistributionSet` constructs duration-dependent transitions, so the canonical transition shape is `[K,D,K]` per timestep. A duration-independent transition implementation is an allowed component variant, not the default package model.

## Duration PMF and discrete hazard

For a normalized duration PMF `q_t(k,d)`, define survival mass at age `a`:

```text
S_t(k,a) = sum_{r=a..D} q_t(k,r)
```

The probability that the current episode ends on the next boundary, conditional on having reached age `a`, is:

```text
h_end_t(k,a) = q_t(k,a) / S_t(k,a)
```

and continuation is:

```text
h_cont_t(k,a) = S_t(k,a+1) / S_t(k,a)
```

with `h_cont_t(k,D)=0`. For every reachable `(k,a)`:

```text
h_end_t(k,a) + h_cont_t(k,a) = 1.
```

Impossible duration support has exact zero mass. If `q_t(k,a)=0`, the corresponding end branch has zero mass; if no duration greater than `a` is possible, continuation has zero mass.

## Causal state-age model

The causal model is a dynamic-hazard process over `(z_t,a_t)`.

Initialization:

```text
P(z_0=k, a_0=1, x_0) = pi(k) * e_0(k)
P(a_0>1) = 0.
```

For boundary `t -> t+1`, from `(i,a)` there are exactly two branch types.

Continuation of the same episode:

```text
P(z_{t+1}=i, a_{t+1}=a+1 | z_t=i,a_t=a,F_t)
    = h_cont_t(i,a)
```

for `a < D`.

Episode boundary followed by a new episode:

```text
P(z_{t+1}=j, a_{t+1}=1 | z_t=i,a_t=a,F_t)
    = h_end_t(i,a) * A_t(i,a,j).
```

A self-transition `j=i` is still a new episode and therefore resets age to one.

After propagation across the boundary, `x_{t+1}` is scored through `e_{t+1}(z_{t+1})`. Thus `x_{t+1}` cannot influence whether the preceding episode ended on `t -> t+1`.

For a causal latent state-age path `(z_0,a_0),...,(z_{T-1},a_{T-1})`, its joint observation/path probability is the initial factor times every boundary branch probability times every emission probability.

### Dynamic-context consequence

When `q_t` changes with context over time, the causal model does **not** correspond to drawing one immutable total duration at episode start. Each future boundary uses the duration PMF/hazard available at that boundary's causal information set.

Therefore the causal dynamic-hazard model is equivalent to a classical fixed-duration segment model only when the relevant duration PMF is constant across the episode (in particular, the context-free case).

## Non-causal segment model

The non-causal forward path is a segment dynamic program. A segment ending at timestep `t`, with state `k` and total duration `d`, starts at:

```text
s = t - d + 1.
```

Its segment factor is:

```text
q_t(k,d) * product_{r=s..t} e_r(k).
```

For the first segment (`s=0`), the predecessor factor is `pi(k)` and no transition is applied.

For a later segment, if the preceding segment ended at `s-1` in state `i` with duration `d_prev`, its boundary factor is:

```text
A_{s-1}(i,d_prev,k).
```

Thus a complete non-causal segmentation with segment states `k_m`, durations `d_m`, starts `s_m`, and ends `t_m` has probability:

```text
pi(k_1)
* q_{t_1}(k_1,d_1)
* product_{r=s_1..t_1} e_r(k_1)
* product_{m=2..M} [
    A_{s_m-1}(k_{m-1},d_{m-1},k_m)
    * q_{t_m}(k_m,d_m)
    * product_{r=s_m..t_m} e_r(k_m)
  ].
```

The production non-causal forward recursion sums exactly over predecessor state and predecessor duration before applying the current segment factor.

### Relationship to the causal model

With context-free duration probabilities, the hazard identity telescopes:

```text
product_{a=1..d-1} h_cont(k,a) * h_end(k,d) = q(k,d).
```

Therefore causal state-age paths and segment paths assign the same duration mass when the duration PMF is fixed across the episode and boundary transition/emission factors are aligned.

With time-varying duration context, the production causal and non-causal paths are intentionally **not assumed equivalent**: causal inference composes boundary-local hazards `q_t`, whereas non-causal segment inference scores one `q_t(k,d)` at the segment end. Correctness gates must validate each contract separately unless a stronger context equivalence is explicitly established.

## Viterbi contract

Viterbi decoding must return a globally maximum-probability valid latent path under the selected causal or non-causal factorization.

For duration-dependent transitions, a non-causal MAP recurrence must maximize jointly over predecessor state and predecessor duration for each candidate destination state:

```text
max_{i,d_prev} [ predecessor_score(i,d_prev) + log A(i,d_prev,j) ].
```

It is not generally valid to choose the locally best predecessor duration for state `i` before applying the destination-dependent transition score.

### Known Gate-0 implementation blocker

At Gate 0 audit time, the production non-causal `_viterbi` path collapses predecessor duration into `V[prev_t,state]` / `best_dur[prev_t,state]` before applying `A(prev_state,prev_duration,next_state)`. This can discard a slightly worse predecessor duration whose transition to the requested destination is much better.

The non-causal forward likelihood does not have this defect because it sums over predecessor state and predecessor duration jointly. Gate 1 must not start until the Viterbi recurrence is corrected and a regression case protects this contract.

## Hand-check fixtures

These fixtures define unambiguous semantics before the independent oracle exists.

### Fixture A — `K=1`, `D=1`

There is one state and every episode has duration one. Every boundary ends the current episode and starts a new episode in the same state. For observations `x_0..x_{T-1}`:

```text
P(x_0:T-1) = product_t e_t(1).
```

All filtered state mass is on state 1 and all age mass is on age 1.

### Fixture B — deterministic duration two, `K=1`

Let `q(d=2)=1`. Starting at age one:

```text
age sequence = 1,2,1,2,...
```

The age-1 boundary must continue with probability one; the age-2 boundary must end with probability one. The self-transition after that end starts a new episode at age one.

### Fixture C — self-transition resets age

For any `K`, if an episode ends in state `i` and `A(i,a,i)=1`, the next latent state is still `i` but the next episode age is exactly one. State identity alone cannot be used to infer whether an episode boundary occurred.

### Fixture D — impossible duration support

If `q(k,1)=0`, an episode in state `k` that has reached age one cannot end at the first boundary. If `q(k,d)=0` for all `d>a`, an episode that has reached age `a` cannot continue. These branches must carry exact zero probability / `-inf` log mass.

## Gate-0 status

The mathematical semantics above are frozen for the correctness milestone with one implementation blocker: non-causal Viterbi with duration-dependent transitions violates the stated global-MAP recurrence. The blocker is a code defect, not an ambiguity in the contract.

Gate 0 is therefore **contract complete / implementation blocked**. Correct the non-causal Viterbi recurrence and add its regression before beginning the independent exhaustive oracle in Gate 1.
