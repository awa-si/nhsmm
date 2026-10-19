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
