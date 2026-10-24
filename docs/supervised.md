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
HSMM first derivatives use the exact adjoint described below; interior higher
derivatives retain full-history serial autograd. At the exact h_tail=1 boundary, filter-score derivatives use the missing-path
correction below. First and second derivatives are supported; constructing a
third-derivative graph is explicitly rejected. Market-scale performance requires measurement
before allocating a fit budget.

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

## HSMM gradient cost

Anchor: `doc:nhsmm.causal-prefix-adjoint` in
`nhsmm/models/filter_grad.py`. On CPU Float32/Float64, differentiable normalized
HSMM/D+ prefixes use an exact vector-Jacobian product through the complete
history. Ordinary inference and the bounded HMM scan keep their existing paths.
Other devices/dtypes retain full-history autograd.

The forward pass calls the same `_causal_prefix_serial` owner as ordinary
forward inference; it stores normalized frontiers and factor inputs rather
than every recurrence operation's autograd graph. First-derivative edge weights
are batched across time with the existing NumPy dependency on CPU. Tensor-to-array
views only read saved values; the exact custom VJP returns tensor gradients to
the complete Torch parameter/context graph. Interior higher derivatives use
Torch recomputation. Fully populated batches omit per-row padding/index copies while
retaining the same causal step owner and evidence-addition order. Let p be a flattened
state-age posterior and v its incoming log-posterior adjoint. Normalization gives
`g = v - p * sum(v)`. A prediction edge with mass w entering destination j
receives `g_j * w / predicted_mass_j`. Sum each edge derivative into its
source adjoint and duration/transition factor; sum destination-age derivatives
for the emission. Iterate backward through every row. Empty support contributes
zero; D+ continuation and reset edges are both included. Ragged padding has
zero derivative. No label, approximation, clipping or truncated history enters
this derivative rule.

First-derivative work is O(T K^2 D), matching the state-age transition law.
Saved frontiers cost O(B T K D); the duration-dependent transition inputs still
cost O(B T K^2 D). This reduces recurrence graph overhead, not model state space.
When higher derivatives are requested, recompute the same serial owner under
autograd, preserving Hessian-vector products rather than differentiating saved
forward intermediates as constants. No new dependency, optimizer, artifact field
or public configuration is added. Native torch.func transforms are not claimed.

A related hard-support fix makes a single supported `logaddexp` branch an
explicit identity: PyTorch's generic backward can otherwise produce a NaN
second derivative for one finite and one -inf input. Both-impossible reductions
retain -inf value and zero derivative. Finite-branch values and first derivatives
are unchanged.

Decision basis: retain the existing inference and PyTorch tensor primitives;
adapt PyTorch's supported custom autograd/recomputation mechanism instead of
adding another HSMM backend or a dense expanded-state matrix scan. The latter
would materially enlarge arithmetic/storage for the measured K=3,D=8 workload.
Primary implementation references:
https://docs.pytorch.org/docs/2.14/notes/extending.html and
https://docs.pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html.
Independent raw-parameter probability/gradient tests remain mandatory, together
with direct full-history VJP comparisons, finite differences, gradcheck,
gradgradcheck, hard support, ragged resets and same-workload runtime/memory checks.

NumPy log-space primitive reference: https://numpy.org/doc/stable/reference/generated/numpy.logaddexp.html.

Performance checkpoint 2026-10-20: 1,421 canonical/affected integration tests
passed, three CUDA skips (209.87 s). The paired K=3,F=3,D=8,T=1,440 CPU
probe takes 0.956 s/update (finite) and 1.117 s/update (tail) in its additional
ten-update phase. The paired three-update fits are 2.06x/1.66x faster; process
peak RSS falls by 23.0%/29.4%. Cold startup and longer-phase timings remain
separate. A 300-update tail fit is not admitted under a 300-second cap by this
bounded evidence. No market fit/budget reset or inference/artifact schema
change occurred. See Nautilus docs/nhsmm-hsmm-adjoint-gate.md for exact source,
probability/gradient tolerances, regression evidence and measurement limits.


## Exact finite-support boundary in D+ filter supervision

At h_tail=1, omitted zero-continuation paths still have nonzero derivatives.
Ordinary first-order training keeps the exact existing log-domain missing-flux
VJP. Higher requests reconstruct a differentiable degree-two missing-path
expansion from the once-sampled forward inputs. One extra frontier removes one
zero continuation edge; an ordered pair removes two. With delta=1-h, label and
total evidence ratios contribute delta*r1 + delta*delta*r2
minus half the squared first-order term to each prefix log probability.
This gives exact boundary Hessians and mixed parameter/context derivatives.
No perturbed hazard, resampled network head, probability clipping or dense
state-age transition matrix is used.

The higher-derivative path costs O(T*M^2*K^2*D), M<=K active boundary states,
with extra frontier storage O(M^2*K*D) plus its autograd graph. It is only built
when differentiable first gradients are requested. Third-derivative graphs
are rejected explicitly. The separate marginal-likelihood boundary VJP remains
first-order. Independent raw-parameter probability-path Hessian-vector products
cover D=1/2/3, self resets, mixed boundary/interior tails and E/U contexts.

## Large common emission offsets

Prefix filtering scores center emissions before combining them with latent
priors and boundaries, including the small-HMM scan and exact h_tail=1 VJP.
Causal marginal likelihood separates and restores the shared row constants
around its forward reduction. Independent raw-weight path enumeration verifies
parameter, emission-score and explicit E/U context gradients for both scores
in Float32/64. See [the numerical contract](model.md#causal-emission-offset-invariance).
These checks preserve the declared objectives; they do not change supervision
into a marginal state-path MAP objective or alter the existing first-order
boundary derivative limit.


## Boundary-aware regularized static fitting

prepare_supervised_sequences(X, states, starts_at_boundary=flags, mode="hsmm")
requires one explicit boolean per input sequence (a scalar for one sequence).
Unknown first episodes are trimmed through their first observed state change.
All-unknown constant paths are omitted; no remaining rows is an error. The helper
is for training only: evaluation labels must never trim an inference window.
It returns observations, labels and an offset/retained-row provenance report.
HMM rows do not need age trimming.

After fitting emissions/initial weights with fit_supervised on that cohort,
model.fit_supervised_boundaries(states, starts_at_boundary=flags) refits only
static canonical duration/transition parameters. The final episode contributes
observed continuations but no invented exit. Duration hazards are
(end_count+alpha)/(end_count+continue_count+alpha+beta), with finite h_D=1.
D+ pools all tail-bucket exposures. The implied PMF uses survival products.

Transition rows add a positive symmetric pseudocount and optional concentration
times a pooled train-only destination distribution, respecting structural
support. This is an empirical-Bayes shrinkage/count estimator; it is not a new
Adam objective and does not promise lower training NLL. Four ModelConfig
supervised_duration_end_prior / supervised_duration_continue_prior /
supervised_transition_prior / supervised_transition_pool_strength fields own
the priors. Existing fit_supervised behavior is unchanged.

The model must be initialized, static causal HMM/HSMM and ergodic. State-labelled
HSMM count fitting requires no self resets. Custom heads are rejected.
Temperature and duration gates are inverted when encoding target probabilities;
actual model scores are verified on a private candidate before live mutation.
Emission/initial weights, mode and trainability are preserved. Fit manifests
should retain returned counts, priors and cohort provenance.

For unknown-age inference use a preserved filter state/history, or explicitly
choose stationary_filter_prior for a time-homogeneous stationary entry law.
See stationary-entry.md. Neither training labels nor validation labels enter
the causal inference initialization.
