# External context timing

`CausalContext(emission=E, boundary=U)` separates external information sets.
E[t] must be known before x[t]. U[t] may include x[t] and controls the outgoing
t -> t+1 transition and duration hazard. Initial probabilities use E[0].
The caller owns timestamps and causal feature construction. No automatic lag
is applied. Plain Tensor contexts retain their existing shared timing semantics.

Sequence APIs accept the existing dense/static or ragged context shapes in both fields. Streaming accepts single-step Tensor shapes. Missing
channels, nonfinite available values and incompatible lengths are rejected.
Explicit availability masks select base scores for missing rows, as described below.
A stream cannot change between internal and external context without reset;
reset discards filtering history. No-context inference bypasses context heads
when the model has no internal encoder; it does not reconstruct a separately
trained baseline posterior.

Use `fit_supervised(..., context_only=True)` to train only distribution context
networks, projections and delta scales from the current fitted base parameters.
Base parameters remain fixed; requires_grad flags and model mode are restored.
This is an explicit option, not a changed training default. Latent fitting retains
its existing initialization policy. A staged fitted base must use supervised
fitting or the documented preserve-initialization latent configuration.

Duration conditioning remains a dynamic hazard evaluated at each boundary, not
an episode-entry duration draw. This interface makes no claim of trading alpha.


## Runtime correction and compatibility

The optimized ergodic transition path now applies structural constraints before
normalizing. This restores disabled self-transition semantics and agreement with
batch inference. Existing artifacts can produce corrected streaming posteriors;
their parameter paths and schema remain unchanged. Historical research archives
retain their original source identity.

## Verification before base-score extension

The canonical suite passed 1209 tests, with 3 skips for unavailable CUDA.
Thirty study-harness regression tests also passed. The eleven new timing tests
include independent path products, streaming parity and frozen-base invariants.
The synthetic close-time fixture does not validate production OHLCV/TA code.
Exact source hashes and execution outputs are retained in
https://github.com/awa-si/nautilus/blob/c45d94f4a8b6c87870ee2dc8f775c9609128c027/docs/nhsmm-causal-context.json.
No new market study or automatic per-row missing-context fallback was run.

## Explicit base scores during external streams

`BASE_CONTEXT` is an explicit whole-step bypass of the context networks and
internal encoder. It uses the current model's base distributions. It is not a
zero context vector and does not recover an independently trained baseline if
base parameters were changed by joint training.

Start an external stream with `context=BASE_CONTEXT` or explicit context, then
alternate `BASE_CONTEXT`, Tensor context and `CausalContext` without resetting.
The posterior and episode age continue across missing context. The previous
step's stored boundary scores still propagate into the current observation;
base boundary scores selected at t affect the following t -> t+1 edge.
`None` keeps its internal/no-external meaning and cannot switch stream protocol.
Batch likelihood/filtering accept `BASE_CONTEXT` to bypass any internal encoder.
Context-only fitting and transition refinement reject this bypass.

BASE_CONTEXT applies to the entire batch at a step. For per-item availability,
use CausalContext.available as described below. Per-component masks are unsupported. An external adapter owns missingness,
freshness and observation-clock policy; invalid context is not automatically
reclassified as missing.


## Base-score extension verification

The extended source passes 1215 canonical core tests, with 3 CUDA skips.
The host passes 162 integration/TA tests and 11 subtests; one isolated no-GBM
installation-profile test skips. A separate market-adapter smoke succeeds with
GBM, Qlib, MLflow and RD-Agent imports blocked. Ruff and Black pass.
Exact source hashes and execution outputs:
https://github.com/awa-si/nautilus/blob/cf3a8ef59fbbdf2484f5679307568b2c812f38bf/docs/nhsmm-context-fallback.json.

The opt-in market adapter and production TA/Axis replay are documented in the
host report. Synthetic fixture verification does not establish a historical
market advantage. The following extension adds mixed-availability batch scoring
and training; the counts above describe the earlier base-score revision.

## Mixed availability within a batch

Use `CausalContext(emission=E, boundary=U, available=M)`. M is a strictly
boolean scalar, shared [T], exact [B,T], or a list of [Ti] masks for ragged
sequences. None preserves the all-available legacy behavior. For a batched
streaming step use [B,1]; [B] is not a per-stream shorthand.

At unavailable rows, all context corrections are bypassed before probability
links and constraints. Initial probabilities use M[0]; emissions and outgoing
hazards/transitions use M[t]. The prior step's boundary still propagates into
the current observation. Observations are retained; the posterior and episode
age continue. Missing context rows may contain NaN/Inf: they are sanitized and
excluded before head evaluation. Available rows must be finite. A zero vector
without a false mask remains ordinary context and can activate neural biases.

Likelihood, observed-path supervision, decoding, filtering, streaming and latent
optimization chunks carry the same mask. Context-only fitting and transition
refinement reject datasets with no available rows before changing parameters.
Masked context requires row-gating support in every distribution; unsupported
custom distributions are rejected. The unmasked custom-distribution call contract
is unchanged. Both E and U must retain their ordinary valid shapes.

The mask gates both information sets together; separate emission/boundary
availability is unsupported. It is a caller-supplied covariate, not a learned
missingness process. Pre-observation emission availability must itself obey the
information boundary. Exact fallback means current base scores, not the posterior
of an independently run static model. No parameter paths or artifact schema change.

Verification and native RD-Agent/Qlib/MLflow integration evidence are retained in
https://github.com/awa-si/nautilus/blob/5adf39f034e02c72f2c5fd42b133669233865370/docs/nhsmm-masked-context.md.
