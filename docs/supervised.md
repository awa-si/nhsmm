# Fully observed causal state paths

`NHSMM.supervised_log_likelihood(X, states, context=None, reduce=False)` scores
the observed joint density `p(X, states | context)`. This is a supervised
generative objective. It differs from both marginal `p(X)` and conditional
classification `p(states | X)`. Labels never enter the emission inputs or encoder.

Each state path is an unpadded `torch.long` tensor with values `0..K-1`, one per
input sequence. A single sequence may use one tensor. Ragged inputs use lists.
Every sequence begins at episode age one. Its final episode is right-censored:
there is no terminal end-hazard factor. The objective uses the same causal timing
as filtering: boundary factors for `t-1 -> t` use context at `t-1`; current
observations enter through the emission at `t`.

For an ordinary HMM choose `max_duration=1`, `duration_tail=False`,
`causal=True`, no context encoder and `context_dim=None`. Every edge is a new
episode, including a self-transition; the resulting state model is a standard
Markov chain with diagonal Gaussian or Student-t emissions.

For state-labelled HSMM paths, choose `allow_self_transitions=False`. State
changes then identify episode boundaries uniquely. This option currently
requires ergodic transitions and K>=2. It changes the canonical structural mask,
so forward, filtering, Viterbi and reconstructed artifacts all obey the same
exclusion. Default `True` preserves existing model behavior and old artifacts.
When self resets are allowed, state labels alone cannot identify HSMM episodes;
this supervised API rejects that ambiguous configuration. Explicit boundary
labels and partially labelled sequences are not implemented.

Given state `s`, zero-based age `a`, duration end hazard `h_t(s,a)` and transition
matrix `A_t(s,a,j)`, the observed path contributes:

- `log pi(s_0)` and all emission log densities;
- `log(1-h_{t-1}(s,a))` on a continuation;
- `log h_{t-1}(s,a) + log A_{t-1}(s,a,j)` on an episode change.

An impossible path returns `-inf`. Invalid labels and noncausal models raise.
Finite-support episodes cannot exceed D. D+ models use the existing tail hazard.
Zero-probability labelled events cannot be optimized with this API; fitting fails
before an update instead of masking the event or inventing its derivative.

`fit_supervised(X, states, context=None, steps=200, lr=0.03)` runs a bounded
full-batch Adam fit from current parameters. If needed, distributions initialize
once. It performs exactly `steps` updates, checks finite losses/gradients, projects
tail parameters and returns a separate observed-joint objective summary. It does
not call latent-state optimization, restart, permute labels, select on validation
or modify the existing latent optimizer summary. Training mode is restored.
The caller owns initialization, seeds, split/data provenance and external-context
availability. Dropout-free settings are recommended for deterministic verification.

The canonical independent probability-product, gradient, marginalization and
artifact tests are in Nautilus `tests/ml_stack/nhsmm/test_supervised_paths.py`.
The synthetic native ML-stack ladder is owned by Nautilus
`adapter/qlib/nhsmm_research/supervised_ladder.py`. It establishes controlled
supervised learning behavior; it does not establish market regimes, eliminate
latent-state collapse in unsupervised training or qualify a trading model.

Emission initialization without an explicit mode now honors the configured
random/spread mode. Data-dependent kmeans modes retain a spread placeholder
until the observation-aware latent training initializer runs; supervised fits
from current parameters do not perform implicit kmeans initialization.

## Frozen-base context training

fit_supervised(..., context_only=True) trains only distribution context networks, projections and delta scales from current parameters. Base parameters stay fixed; trainability flags and model mode are restored. Explicit external context is required. Missing context is not replaced by zeros. See [the timing contract](causal-external-context.md).

Mixed availability is supplied as CausalContext.available. Unavailable rows
retain their observations and base density/boundary factors; their context heads
are not evaluated. Context-only fitting requires at least one available row.
The caller owns the availability information boundary. See the timing contract.


## Causal filtering objective

Anchor: `doc:nhsmm.supervised-filter` in `nhsmm/supervised.py`.
`supervised_filter_log_score(X, states, context=None, reduce=False)` sums
`log P(z_t | X_0:t, causal context)` per sequence. This is a composite per-row
classification score, not log probability of the complete label sequence given
all observations. Each prefix uses model-produced filtering history; no true
previous label, future row, detached history or terminal exit is introduced.
Scores remain in log space, so rare events are not silently clipped. Impossible
labeled events score -inf and fitting rejects them before an update.

`fit_supervised(..., objective="causal_filter_log_score")` minimizes the negative
sum divided by the number of valid observations using the existing bounded Adam
loop. The existing `observed_joint_causal_prefix` objective remains the default.
The filtering objective accepts causal HMMs, finite-duration HSMMs and D+
models. Noncausal configurations fail explicitly before optimization. For each
prefix, the normalized state-age frontier is summed over all ages in log space.
The D+ bucket includes all ages at least D; its continuation mass is retained.
Self resets are latent events and are marginalized by the forward kernel, so
state-only filtering supervision permits them. The observed-joint objective
still requires no self resets for HSMMs because its labelled path must uniquely
identify episodes. Ragged sequences reset independently;
padding and empty prefixes contribute no score. Causal external E/U routing and
boolean availability use the existing core forward primitives. Labels only gather
prefix scores. The artifact/state_dict schema and inference law are unchanged.

Decision: adapt the existing normalized state-age forward kernel and PyTorch
log-space/autograd operations instead of importing a second model framework or
creating a parallel inference/optimizer implementation. The internal normalized
forward option avoids adding and subtracting a large cumulative evidence offset.
Complexity follows the existing causal forward scan; the graph retains full
history for exact gradients. No truncated backpropagation is used.

References reviewed: Chen, Chang and Mehta (L4DC/PMLR 331, 2026),
https://proceedings.mlr.press/v331/chen26c.html, establishes differentiable causal
HMM filtering with SGD but optimizes next-observation prediction, not this
supervised prefix-state score. No parameters, architecture or market-performance
claim are imported from that paper. PyTorch 2.14 logsumexp documentation:
https://docs.pytorch.org/docs/2.14/generated/torch.logsumexp.html.
Independent prefix enumeration/gradient tests live in the canonical Nautilus
model suite. Classification improvements require separate empirical evidence.


## Small-HMM gradient cost

Anchor: `doc:nhsmm.hmm-prefix-scan` in `nhsmm/models/forward.py`.
The initial full-history serial gradient probe (CPU, K=3, F=2, T=1440,
three updates, one thread) cost 3.614 seconds/update, exceeding the proposed
300-second candidate budget at 300 updates. The filtering-score path therefore
uses an inclusive log-semiring matrix-prefix scan for K<=3, D=1, no tail and
finite singleton duration scores. Each block is rescaled by one common log
mass; final state normalization cancels this factor, including its derivatives.
The larger-state or unsupported-score paths retain the serial reference.
Default forward/inference and the generative objective are unchanged.
The specialization adds O(T K^3 log T) arithmetic for small dense matrices to
replace T dependent Python/autograd stages with log(T) tensor stages. It is
not claimed to be work-optimal or beneficial for large K. Exact zero support
uses the existing safe logsumexp; impossible prefixes fail rather than clip.

Adapted mathematical approach: Hassan, Sarkka and Garcia-Fernandez (2021),
Temporal Parallelization of Inference in Hidden Markov Models, IEEE TSP 69,
4875-4887, doi:10.1109/TSP.2021.3103338, https://arxiv.org/abs/2102.05743.
This implementation uses established stable tensor primitives rather than the
PyTorch 2.14 associative_scan prototype, whose documentation warns of missing
autograd support: https://docs.pytorch.org/docs/2.14/higher_order_ops/associative_scan.html.
No external implementation is copied. Independent prefix/gradient references
and direct same-input serial-versus-scan checks are required before market use.

Integrated verification on 2026-10-20: 1,330 tests passed, three CUDA skips (Nautilus canonical model suite plus affected supervised/runner integration bundles). The same 1,440-row three-update CPU probe fell from 10.8417 to 1.4960 seconds; initial/final objective values agree in that probe. This is a bounded same-host result, not a universal performance guarantee.

## HSMM filtering supervision

The score is
`sum_t log sum_a P(z_t=label_t, age_t=a | X_0:t, causal context)`.
Each prefix starts at age one and has an open final episode. No terminal
exit factor or completed-duration PMF is applied to that episode. The shared
dynamic-hazard law in [the mathematical contract](hsmm-mathematical-contract.md)
uses outgoing U_t on boundary t -> t+1; time-varying context does not freeze a
duration draw at episode start. With no context and a fixed PMF, prefix survival
agrees with the classical right-censored segment law.

Implementation decision: reuse the existing normalized state-age recursion
without changing propagation, parameters, artifact schema or optimizer.
The small-HMM scan remains restricted to its existing D=1/no-tail case.
HSMM gradients retain the complete serial graph; market-scale performance
requires measurement before allocating a fit budget.

Primary references reviewed: Yu (2010), Hidden semi-Markov models,
doi:10.1016/j.artint.2009.11.011,
https://www.cs.ubc.ca/~murphyk/Teaching/CS540-Spring10/projects/Yu-hsmm09.pdf
(state-plus-age representation and boundary censoring); Chiappa,
Explicit-Duration Markov Switching Models, https://arxiv.org/abs/1909.05800
(explicit-duration representations). These references motivate the state-age
formulation; they do not establish correctness of this implementation.
Canonical independent tests enumerate raw-parameter probability products,
including continuation and reset branches, compare every prefix and gradients,
and check finite differences, missing context, ragged resets and tail support.

Filtering labels select separate prefix marginals, not a single constrained
state path: a label sequence with zero observed-joint support can still have
finite per-row filtering scores. State-only filtering supervision also does
not guarantee identifiability of duration distributions and self-reset rates.

Verification checkpoint 2026-10-20: the canonical Nautilus model suite and
affected host regressions pass (1,404 passed, three CUDA-unavailable skips).
The expanded tests independently enumerate reset/continuation histories from
raw parameters and compare every prefix frontier, filtering score and gradients.
See Nautilus docs/nhsmm-hsmm-filter-math-gate.md and its machine evidence for
source hashes, bounded long-sequence numerics, timing and scope limits.
