# Hypothesis: neural context for a stable HSMM

Status: research hypothesis; not evidence of predictive or trading value.
Date: 2026-10-19. Research/contract review: 2026-10-19. Model owner: `awa-si/nhsmm@v3`.

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

## What is established and what remains a hypothesis

- Established model family: conditional/neural HMMs, explicit-duration models
  and covariate-dependent HSMM durations have published predecessors. Combining
  these components is not, by itself, a novelty or performance claim.
- Information argument: finer bars can retain information lost by aggregation;
  deterministic TA cannot add information beyond its complete source history.
  Easier access for a finite fitted model is a separate, testable advantage.
- Proposed application: the particular 1m/5m/15m/1h partition, selected TA blocks,
  component routing and staged training policy remain unvalidated on market data.
- Empirical claim: predictive density, state interpretation, directional
  forecasting and net trading performance are different endpoints. Improvement
  in one must not be reported as improvement in all of them.

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
not forecastability. This example uses equal interval boundaries and one source;
independently supplied feeds may differ in alignment, corrections or coverage.
Such differences require provenance checks, not a claim of new market information.
Higher-timeframe features may help a finite model use long
history efficiently even when they are deterministic summaries of that history.

The HSMM clock must be declared separately. If one step is five minutes,
episode age and duration are measured in five-minute steps. A one-minute HSMM
with five-minute decisions has different duration semantics. Missing intervals
must not be silently compressed into consecutive observations.

Higher-timeframe values carried forward at 5m decisions are stale observations,
not fresh independent measurements on every decision. Track their age, source
bar end, warm-up and missingness. Calendar phase/time-of-day is an important
control: an apparent timeframe effect may instead reflect intraday seasonality.
OHLCV-derived 1m detail is not an order-book or bid-ask measurement.

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

### Explicit component timing and legacy shared external rows

`CausalContext(emission=E, boundary=U)` now supplies distinct information sets.
E_t must be available before observing x_t; U_t may include x_t and controls only
the outgoing boundary t -> t+1. The initial prior uses E_0. Both dense/static and
ragged inputs are supported; batch likelihood, observed-path supervision and
streaming use the same routing. No automatic lag or timestamp inference is
applied. The host must still verify the feature construction against event time.


The legacy plain-Tensor route retains the following timing:

| Component at step t | Internal observation-derived context | External row U_t |
| --- | --- | --- |
| Initial prior at t=0 | Zero context | U_0 |
| Emission for x_t | C_(t-1), with C_(-1)=0 | U_t |
| Outgoing boundary t -> t+1 | C_t | U_t |

For a plain external Tensor row U_t, the compatibility contract uses U_t for the emission of
x_t and for outgoing duration/transition factors on boundary t -> t+1. The
next emission uses U_(t+1). Thus a row derived after observing x_t cannot simply
be reused as an exogenous input to its own emission likelihood. Lagging an
external tensor to protect emissions also changes the information available to
the outgoing boundary. A single row does not generally provide both pre-x_t
emission context and a different post-x_t boundary context by one index shift.

Before a market experiment, specify a component-by-component event-time table.
Verify the chosen routing against batch and streaming behavior. Using a more
conservative lag is a valid declared design choice; changing routing to expose
newer information now uses the explicit `CausalContext` route described below. The
internal causal encoder already has distinct emission-lag semantics; it is not
interchangeable with arbitrary externally derived context.

For history-derived context, evaluate the sequential predictive score
sum_t log p(x_(t+1) | H_t), with every feature computed from its allowed prefix.
Do not assume that a likelihood conditional on an entire externally fixed
context trajectory has the same interpretation when later context rows encode
earlier observations. No same-bar reconstruction score is evidence of a future
forecast. This is a modeling requirement derived from the information boundary.

### Duration conditioning: dynamic hazard versus entry-time duration

The [mathematical contract](docs/hsmm-mathematical-contract.md) defines the
current causal core as a dynamic-hazard process over state and episode age.
For finite duration support, away from a separately configured D+ tail:

```text
h_t(i,a) = q_t(i,a) / sum_(r=a..D) q_t(i,r)
P(continue episode i, age a+1 | i,a,H_t) = 1 - h_t(i,a)
P(new episode j, age 1 | i,a,H_t) = h_t(i,a) * A_t(i,a,j)
```

Each reachable row must be normalized and have positive survival support.
Context changes must not remove survival support from an already reachable age.
When q_t changes during an episode, multiplying successive boundary factors is
not generally equivalent to drawing one duration from q at episode entry.
An entry-conditioned duration model and the current dynamic-hazard model are
different hypotheses; neither the name HSMM nor a paper using that name proves
their equivalence. Koslik's entry-time-indexed construction [7] is a relevant
comparison, not an implementation identity.

The geometric dwell-time restriction belongs to a homogeneous first-order HMM.
With time-varying, covariate-dependent transition probabilities, a first-order
HMM can already have non-geometric overall dwell times [7, introduction].
Therefore explicit-age value must be tested against a context-matched
time-varying-transition HMM, not only a static HMM.

Sequence endpoints are right-censored on the causal path: a data-window end is
not an observed regime exit. A fresh age-one start also differs from starting
inside an unknown-age market episode. Declare warm-up/initial-age treatment and
use identical boundary information for all candidates. See the mathematical
contract; do not compare causal prefix and completed-segmentation likelihoods
as if they were the same score.

Multi-step duration/transition forecasts require a stated treatment of future
context. Frozen-current-context scenarios, specified context paths and forecasts
that integrate over unknown future context are different quantities. Future
realized TA must not be supplied as if it were known at the forecast origin.

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

`fit_supervised(..., context_only=True)` now freezes base distribution
parameters while training only their context networks, projections and delta
scales. It restores the original trainability flags and model mode. This option
does not implement availability masks or reconstruct a static posterior.
Both `CausalContext` fields are required; nonfinite or misaligned values are
rejected. Plain zero context remains an ordinary network input.

An availability-aware residual gate on finite parameter coordinates, followed
by the existing probability links, is one candidate design: base plus a masked
context correction. It is a proposal, not a current capability. Missing inputs
must be handled before network evaluation; multiplying a NaN by zero is not
safe. Exact base-parameter fallback additionally requires preserving those
parameters. Even then, a posterior conditioned on earlier context need not
equal the posterior of a static model that never used context.

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

7. [Koslik, Hidden Semi-Markov Models with Inhomogeneous State Dwell-Time Distributions, 2024 preprint](https://arxiv.org/html/2405.13553v1).
   Directly studies covariate-dependent dwell times and conditional transitions.
   Section 2.3 indexes the duration law by episode-entry time; the associated
   hazards retain that entry index as age advances. This differs from this
   package's current-boundary dynamic q_t. Its introduction also explains why
   inhomogeneous HMMs need not imply geometric overall dwell times. Application:
   animal movement, not intraday trading. This review uses the 2024 preprint.
8. [Rojas-Salazar et al., A Bayesian Hidden Semi-Markov Model with Covariate-Dependent State Duration Parameters for High-Frequency Environmental Data, 2021 preprint](https://arxiv.org/abs/2109.09949).
   Models duration parameters using covariates observed before state transitions.
   Also addresses residual conditional dependence in high-frequency observations.
   Supports duration-context modeling as a researched idea; not financial evidence
   and not proof that our factorization matches theirs.
9. [Wiseman, Shieber and Rush, Learning Neural Templates for Text Generation, EMNLP 2018](https://aclanthology.org/D18-1356/).
   A neural system with an HSMM decoder learning latent templates jointly with
   generation. Establishes a neural-HSMM precedent outside finance; its segmental
   text-generation setting does not validate causal trading or dynamic hazards.
10. [Corsi, A Simple Approximate Long-Memory Model of Realized Volatility, 2009](https://doi.org/10.1093/jjfinec/nbp001).
    HAR-RV combines volatility components over daily, weekly and monthly horizons.
    Supports a parsimonious multi-horizon volatility baseline, not latent-state
    or intraday neural-HSMM superiority. An intraday adaptation is a new experiment.

Verification scope: the six original references were checked during the initial
literature review; the additional sources above address missing model comparisons.
Koslik section 2.3 was checked in full text. Rojas-Salazar and Wiseman are used
for the scope stated by their author/publisher abstracts; Corsi's multi-horizon
formulation was checked against the publisher-indexed abstract; its full text
was not reviewed here. No new empirical replication was performed.
None of these sources is an independent reproduction of this repository.

The original literature review checked implementation statements at core revision
`c97329da2aa8133a38550f7d0383f75a4414adba` against docs/model.md,
docs/hsmm-mathematical-contract.md, docs/supervised.md, runtime.py and the
boundary/objective implementations. The subsequent implementation batch adds
the explicit routing and frozen-base option described above. It also repairs an
independently detected streaming defect: the optimized ergodic transition route
must apply the structural mask, including disabled self-transitions, before
normalization. Independent probability products and batch/streaming comparisons
cover that case. See [the API contract](docs/causal-external-context.md).

## Falsifiable questions

| Claim | Evidence required | Failure or limitation |
| --- | --- | --- |
| 1m detail adds value | Improvement beyond matched 5m history on future data | Distinct information without predictive gain |
| 15m/1h summaries help | Improvement beyond a long-history 5m reference | Gain explained only by a longer lookback or extra capacity |
| Neural mapping is useful | Gain over simple linear/spline conditioning with the same inputs | Simple conditioning matches or beats it |
| Explicit-age structure helps | Gain over a context-matched time-varying-transition HMM | Covariates explain persistence without explicit age |
| Latent state structure helps | Gain over a context-only predictive model | Context alone explains the gain |
| Optional context is robust | Defined availability handling and stable causal state/age estimates | Collapse, invalid likelihood, or broken fallback behavior |
| Prediction helps trading | Improvement after matched costs and execution assumptions | Better density/state diagnostics without economic benefit |

## Validation outline owned by the host

Before running, preregister the instrument/source, data identity, model and
decision clocks, feature schema/lookbacks, availability rules, targets, folds,
budget, seeds, primary metric and acceptance threshold. This outline allocates
no new fit budget and does not authorize using an already inspected test as
an untouched holdout.

Compare static HMM/HSMM, a context-only predictor, simple context-conditioned
HSMM, a context-matched time-varying-transition HMM and neural context-conditioned
HSMM. Ablate 1m, 15m/1h and their combination
on identical rows, effective historical coverage and decision boundaries.
Match capacity/optimization where possible and report remaining differences.
Compare frozen-base conditioning with joint fine-tuning only as separately
declared training protocols. Preserve every failed candidate and selection trial.

Isolate the neural routing before combining effects: emission-only,
duration-only and transition-only conditioning, then a preregistered combination.
The initial-state prior acts at sequence starts; repeatedly resetting it at each
decision would change the model. Keep the observation vector and measure fixed
for predictive-density comparisons. A log score on a different set of TA outputs
is not directly comparable with a return-density score.

Use chronological walk-forward evaluation, train-only processing and the
required purge/embargo for forward targets. Refit every learned transformation
inside its training fold. Repeated seeds on one dataset are not independent
market samples; use temporal blocks and account for multiple comparisons.
Past data may legitimately warm up features and filtering at an evaluation
boundary if the same history would be available live; evaluation outcomes must
not refit weights, preprocessing or thresholds. Report the initialization choice.

Include a seasonality-only control and matched longer-history 5m baselines.
For volatility targets, include a parsimonious HAR-style multi-horizon baseline
with its intraday adaptation declared; do not compare different target measures.
Input masks, lagging and temporal displacement are diagnostic interventions,
not automatic valid significance tests: naive shuffling can destroy serial
dependence and timeframe alignment. Preserve the clock and use an appropriate
blocked null design if a significance test is intended.

Evaluate predictive log score and distribution calibration, finite normalized
likelihood, occupancy/effective-state count and duration stability. Align state
identities across refits before comparing state columns. Real-market regime
accuracy needs independently defined labels; latent market states are not known
ground truth merely because they have been named. Synthetic observed-state
training and forward-return supervision are distinct tasks with distinct labels.
TA-derived regime labels test reproduction of that label rule, not independent
discovery of market regimes; retrospective pivots require their confirmation
times. State-only supervised paths identify episode boundaries only under the
core's no-self-reset restriction; see [supervised semantics](docs/supervised.md).
Partial labels and explicit boundary labels are not current supervised features.

Check residual serial dependence and overlapping rolling features: otherwise
extra states can absorb emission misspecification rather than economic regimes.
A rare state is not automatically a defect, and forcing equal occupancy is not
validation. Diagnose degeneracy relative to the declared purpose and compare
Gaussian/Student-t or autoregressive alternatives only in a budgeted study.
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

## Review outcome and next evidence gate

The architecture remains plausible, with stronger direct duration-context and
neural-HSMM precedents than the initial six-reference list captured. The exact
trading feature mapping, timing route, missing-context behavior and predictive
benefit remain open. Before a real-market fit, the host must resolve the external
component timing table, select the dynamic-hazard interpretation, freeze the
matched baselines/metrics and declare a genuinely uninspected evaluation period.

The context-interface batch changes runtime routing and adds bounded synthetic
correctness tests; it does not run a new market study. Automatic per-row fallback,
real OHLCV/TA pipeline verification and an empirical market advantage remain open.
