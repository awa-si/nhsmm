# External context timing

`CausalContext(emission=E, boundary=U)` separates external information sets.
E[t] must be known before x[t]. U[t] may include x[t] and controls the outgoing
t -> t+1 transition and duration hazard. Initial probabilities use E[0].
The caller owns timestamps and causal feature construction. No automatic lag
is applied. Plain Tensor contexts retain their existing shared timing semantics.

Sequence APIs accept the existing dense/static or ragged context shapes in both fields. Streaming accepts single-step Tensor shapes. Missing
channels, nonfinite values and incompatible lengths are rejected. This API does
not substitute zero for missing data and does not implement a per-row fallback.
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

## Verified source

The canonical suite passed 1209 tests, with 3 skips for unavailable CUDA.
Thirty study-harness regression tests also passed. The eleven new timing tests
include independent path products, streaming parity and frozen-base invariants.
The synthetic close-time fixture does not validate production OHLCV/TA code.
Exact source hashes and execution outputs are retained in
https://github.com/awa-si/nautilus/blob/c45d94f4a8b6c87870ee2dc8f775c9609128c027/docs/nhsmm-causal-context.json.
No new market study or automatic per-row missing-context fallback was run.
