# Hypothesis: neural context for a stable HSMM

Status: research hypothesis; not evidence of predictive or trading value.
Date: 2026-10-19. Model owner: `awa-si/nhsmm@v3`.

This document records the model rationale and an OHLCV motivating example.
The implemented probability and timing contract remains [docs/model.md](docs/model.md).
Market data preparation, TA, experiment budgets, validation and execution remain
in [Nautilus](https://github.com/awa-si/nautilus/blob/main/gates.md); they are not
new responsibilities of this model package. No runtime/API change is made here.

## Core hypothesis

A stable explicit-duration HSMM models the latent state and episode age.
When additional causal information is available, an optional neural mapping
encodes it as context and conditions selected HSMM distributions. The HSMM
remains the probabilistic core; the neural part supplies learned conditioning.
Neural conditioning may improve state dynamics, duration or observation forecasts,
but that incremental benefit must be measured rather than assumed.

The development order is ordinary HMM -> explicit-duration HSMM -> optional
neural conditioning, with correctness and stability checked at each step.
Staged training starts from a stable HSMM before learning context effects.
Whether to freeze the base parameters or jointly fine-tune them is an explicit
experimental choice. Adding a neural component does not itself establish value.

## OHLCV motivating example

For decisions every five minutes, candidate context includes 1m, 15m and 1h
OHLCV histories and causally derived technical features. This is a proposed
input-role assignment, not a universal rule that other timeframes are context.
Features from the signal timeframe can also serve as context.

| Block | Candidate information | Qualification |
| --- | --- | --- |
| 5m | Base return, range and volume observations | Example model clock only; decision frequency need not equal inference frequency |
| 1m | Within-5m path, realized variation and volume distribution | Contains detail lost by 5m OHLCV aggregation |
| 15m / 1h | Slower trend, volatility and relative price position | With aligned bars and the same source, derivable from sufficient 5m history |
| TA | Compact transformations of the available histories | Adds representation and inductive bias, not new raw information |

For example, starting at 100, the 1m closes [101, 102, 98, 99, 100] and
[102, 101, 99, 98, 100], with no extra wicks and volume 10 per minute,
both yield 5m OHLCV [100, 102, 98, 100, 50]. Their sums of squared 1m log
returns are approximately 0.00200058 and 0.00140046. Thus the finer path
cannot be reconstructed from the aggregate. This establishes information loss,
not forecastability. Higher-timeframe features may help a finite model use long
history efficiently even when they are deterministic summaries of that history.

The HSMM clock must be declared separately. If one step is five minutes,
episode age and duration are measured in five-minute steps. A one-minute HSMM
with five-minute decisions has different duration semantics. Missing intervals
must not be silently compressed into consecutive observations.

## Information and probability contract

Let H_t contain only observations and features actually available at decision
time t. A candidate encoder produces C_t = g_theta(H_t). For forecasting, C_t
may condition the next state/age propagation and the next observation density.
For example, p(z_next, age_next, x_next | z_t, age_t, C_t) describes a
one-step predictive factor; it does not define a new public API.

- Track interval start/end, publication/receipt and feature availability times.
  Admit a feature only when its inputs and computation are available.
- Start with completed bars. At 10:05, the final 10:00-10:15 candle is unknown.
  A developing candle is a separate as-of feature requiring identical historical
  reconstruction; do not backfill its eventual high, low, close or volume.
- Compute TA causally. Fit scaling, imputation and learned encoding on training
  data only; respect warm-up, gaps and instrument/session boundaries.
- A transform of x_t is not automatically an admissible exogenous covariate
  for a generative emission likelihood of that same x_t. Avoid emission
  self-conditioning; availability alone does not establish normalization.
- Current internal causal encoding uses observations through t to construct C_t
  for subsequent emission/propagation use. External context follows the distinct
  timing contract in docs/model.md: emissions use the supplied current row and
  boundary propagation uses the previous row. The host must align rows for the
  intended information boundary; do not add an accidental second lag.
- Do not treat nested OHLCV aggregates as independent repeated measurements in
  a product of emission densities. Preserve their deterministic relationships.
- Online estimates use causal filtering. Retrospective smoothing/Viterbi states
  must not be presented as states available to a live decision.

## Optional context and fallback

Context-free inference is a supported model mode. Intermittently unavailable
context requires an explicit policy; replacing missing context by zero is not
generally neutral because zero can be a valid input and neural biases can act.

The current streaming external/non-external context mode is fixed until reset.
Automatic mid-stream switching or a learned availability mask is not established
by this hypothesis. A future availability-aware design must be specified and
tested separately against the public runtime contract.

Disabling context on a jointly fine-tuned model does not necessarily recover the
original trained static HSMM: base parameters may have changed. An exact fallback
to that reference requires an explicit parameter/stream-state preservation or
reconstruction policy. A reset also starts a new inference stream; it is not a
transparent way to preserve the previous filtering posterior.

## Related research and limits

These are related building blocks, not independent validation of this exact
1m/5m/15m/1h neural-HSMM combination.

1. [Bengio and Frasconi, An Input Output HMM Architecture, NIPS 1994](https://papers.nips.cc/paper/1994/hash/8065d07da4a77621450aa84fee5656d9-Abstract.html).
   Establishes the neural input-output HMM architecture; not a trading result.
2. [Bengio, Lauzon and Ducharme, Experiments on the Application of IOHMMs to Model Financial Returns Series, CIRANO 2002s-48](https://cirano.qc.ca/files/publications/2002s-48.pdf).
   Experiments use linear/neural conditional emissions, with unconditional
   transitions, on financial returns. Inputs include current return/variation
   and summaries over four and twelve periods. Nonlinear IOHMMs improve density
   prediction in some comparisons; the historical mean remains the strongest
   first-moment reference. Neural superiority is not uniform across datasets.
3. [Christensen, Turner and Godsill, Hidden Markov Models Applied To Intraday Momentum Trading With Side Information, 2020](https://arxiv.org/abs/2006.08307).
   Uses 1m E-mini S&P 500 data and conditions HMM transitions on volatility-ratio
   and intraday-seasonality signals through an IOHMM. Conditioning uses splines,
   not our neural HSMM. The reported combined IOHMM improves on the HMM, but
   standalone side-information signals have higher Sharpe ratios than the
   combination. This motivates a context-only baseline.
4. [Adam and Oelschlaeger, Hidden Markov Models for Multi-Scale Time Series: An Application to Stock Market Data, 2020](https://research-repository.st-andrews.ac.uk/handle/10023/21734).
   Joint hierarchical modeling of monthly trade volumes and daily returns over
   16 years of Goldman Sachs data. Different temporal resolutions are modeled
   explicitly; this is not a test of intraday TA-conditioned HSMMs.
5. [Bulla and Bulla, Stylized Facts of Financial Time Series and Hidden Semi-Markov Models, 2006](https://doi.org/10.1016/j.csda.2006.07.021).
   Supports explicit durations as a way to reproduce persistence in squared
   returns better than the investigated HMM. It does not establish context
   benefits or executable trading profit.
6. [Oelschlaeger, Adam and Michels, fHMM, Journal of Statistical Software, 2024](https://www.jstatsoft.org/article/view/v109i09).
   Provides financial hierarchical-HMM modeling across temporal resolutions,
   with replication code; a methodological comparison reference.

## Falsifiable questions

| Claim | Evidence required | Failure or limitation |
| --- | --- | --- |
| 1m detail adds value | Improvement beyond matched 5m history on future data | Distinct information without predictive gain |
| 15m/1h summaries help | Improvement beyond a long-history 5m reference | Gain explained only by a longer lookback or extra capacity |
| Neural mapping is useful | Gain over simple linear/spline conditioning with the same inputs | Simple conditioning matches or beats it |
| State-duration structure helps | Gain over a context-only predictive model | Context alone explains the gain |
| Optional context is robust | Defined availability handling and stable causal state/age estimates | Collapse, invalid likelihood, or broken fallback behavior |
| Prediction helps trading | Improvement after matched costs and execution assumptions | Better density/state diagnostics without economic benefit |

## Validation outline owned by the host

Before running, preregister the instrument/source, data identity, model and
decision clocks, feature schema/lookbacks, availability rules, targets, folds,
budget, seeds, primary metric and acceptance threshold. This outline allocates
no new fit budget and does not authorize using an already inspected test as
an untouched holdout.

Compare static HMM/HSMM, a context-only predictor, simple context-conditioned
HSMM and neural context-conditioned HSMM. Ablate 1m, 15m/1h and their combination
on identical rows, effective historical coverage and decision boundaries.
Match capacity/optimization where possible and report remaining differences.
Compare frozen-base conditioning with joint fine-tuning only as separately
declared training protocols. Preserve every failed candidate and selection trial.

Use chronological walk-forward evaluation, train-only processing and the
required purge/embargo for forward targets. Refit every learned transformation
inside its training fold. Repeated seeds on one dataset are not independent
market samples; use temporal blocks and account for multiple comparisons.

Evaluate predictive log score and distribution calibration, finite normalized
likelihood, occupancy/effective-state count and duration stability. Align state
identities across refits before comparing state columns. Real-market regime
accuracy needs independently defined labels; latent market states are not known
ground truth merely because they have been named. Synthetic observed-state
training and forward-return supervision are distinct tasks with distinct labels.
Trading utility requires a separate evaluation after fees, spread, slippage,
turnover and execution constraints. No automatic production promotion follows.

## Current evidence boundary

The retained supervised synthetic studies verify bounded behavior on known
generator states and sinusoidal exogenous context. They do not test this
multi-timeframe OHLCV hypothesis. Their paired static-control follow-up found
healthy models but no consistent overall neural advantage on those fixtures.
That finding neither confirms nor rejects the proposed market-context mapping.

Host evidence: [synthetic confirmation](https://github.com/awa-si/nautilus/blob/503064e48e08e32af5a724bfe2e8bd12b135a919/docs/nhsmm-supervised-confirmation.md)
and [paired context comparison](https://github.com/awa-si/nautilus/blob/503064e48e08e32af5a724bfe2e8bd12b135a919/docs/nhsmm-context-comparison.md).
