---
title: HSMM CPU qualification
description: Verified model-library runtime scope and operational limits.
type: specification
status: active
language: en
owner: AWA
created: 2026-10-22
updated: 2026-10-27
---

# HSMM CPU qualification

This is a technical model-library qualification, separate from the host's
research-only trading baseline. It does not approve a live trading strategy,
establish profitable signals, or promote retained research artifacts into a
live artifact location.

## Qualified runtime and model scope

The recorded local run uses Linux x86-64, Python 3.14.8, PyTorch 2.14.1+cpu and
NumPy 2.5.4. Float32 and Float64 are exercised. The later supervised calibration also passes
on Python 3.14.9 with the same Torch/NumPy versions. This does not qualify every
version admitted by the broad package dependency bounds, accelerators,
Float16/BFloat16, torch.compile, or arbitrary custom distributions/encoders.

The canonical CPU gates cover causal HMM, finite-duration HSMM and the D+ tail,
right-censored final episodes, supervised observed-joint and prefix-filter
objectives, optional external component context, and canonical internal context.
Use the public normalized filter/runtime APIs for posterior probabilities;
raw absolute forward scores have the finite-resolution limits in model.md.
At exactly h_tail=1, the public supervised filtering score supports exact
second and mixed derivatives; third-derivative graphs are rejected.
The separate marginal-likelihood boundary correction remains first-order
and rejects higher-order requests. Use the public score interfaces rather
than deriving boundary gradients from raw hard-zero log frontiers.
See [the derivative owner](duration-tail-contract.md#forecast-and-filtering-derivatives-at-h_tail1)
and [supervised filtering](supervised.md) for the precise scope.

Gaussian and Student-t packaged save/load/runtime paths are exercised.
The extreme-offset *new* analytic emission-parameter gate is Gaussian-only;
it adds to the existing distribution and raw-parameter references, not a claim
of universal extreme-value accuracy for every emission family.

## Operational contract

Load trusted versioned artifacts with load_artifact(require_causal=True).
Keep default freeze=True for serving and do not mutate model parameters during
an episode. One independent runtime belongs to each instrument/stream.
Validate schema, feature order, scaling and fit provenance at the host boundary.
Use reset for an independent episode; do not reset on every observation.
Provide strictly increasing timestamps when timestamp mode is enabled.

Keep context causal: emission information is available before the observation,
boundary context applies to the outgoing boundary, and availability masks
control corrections rather than dropping observations. Labels are score
selectors during supervised fitting, never teacher-forced filter state.
Observed-joint HSMM labels require the documented episode-identifiability
contract; use no self resets where state labels identify episodes.

Default encoderless and canonical incremental-encoder paths have bounded
streaming state. Custom encoders may use a retained-prefix fallback and are
outside this bounded-state qualification. See model.md for runtime validation,
support errors, cache behavior and artifact publication.

save_artifact now publishes by atomic same-directory replacement. See
[publication guarantees and limits](model.md#artifact-publication).
There is no writer compare-and-swap, power-loss durability promise, or
automatic cross-process coordination.

## Evidence and release boundary

The canonical host owns the exact source hashes, JUnit results, unchanged
historical input/artifact hashes, eight retained inference comparisons,
wheel identity/metadata smoke and runtime measurements:
[CPU qualification evidence](https://github.com/awa-si/nautilus/blob/main/docs/nhsmm-cpu-qualification.md).

A passed retained replay is compatibility evidence on the already consumed
October 20 development cohort. It is not a fresh holdout result. Reserved
market cohorts remain unopened; no market refit or optimizer invocation is
admitted by that replay.

Public smoke/release perform package checks without a private checkout or
custom Actions secret. The complete canonical suite runs in its private test
owner against an explicitly pinned public core SHA. Require its successful
matching-source evidence before release; public smoke alone is not that gate.
See [CI ownership](testing.md#ci-ownership-without-a-cross-repository-secret).
Hosted outcomes and the broader dependency-version matrix remain separate
from local evidence. No tag or PyPI publication follows from this qualification.

## Supervised static qualification update (2026-10-24)

Model source 8c0998c is qualified for the declared CPU synthetic HMM/HSMM cases.
The exact-source full canonical owner gate passes 1852 tests with three CUDA
skips. The later host correction has 134 targeted passes, and nine native
calibration profiles complete with all 99 artifact hashes verified.
The first saved profile is replayed without training; the remaining eight
consume the declared budget. Geometric controls retain HMM.
See the [complete result and limits](https://github.com/awa-si/nautilus/blob/1bbcd0b2ffd5d5d8766d702c8a0185ae9b351594/docs/usable-hsmm.md).
These are bounded synthetic results, not evidence of market regime accuracy,
profitability, accelerator support or live strategy readiness. Source code,
ordinary defaults and artifact schema are unchanged by this documentation update.

## Observation-shift qualification limit

A frozen-model diagnostic on already consumed synthetic validation rows
completes technically but fails the declared robustness quality flag in all
nine profiles. Stronger noise and a fixed mean shift degrade accuracy while
posterior confidence stays high. No optimizer updates or core-code changes
occurred; the original well-separated qualification is not a general noise,
shift or confidence-calibration guarantee.
Read the [diagnostic and scope limits](https://github.com/awa-si/nautilus/blob/81066bd4baaa23f8c7aa76f029d6e9b68d85ea26/docs/hsmm-robustness.md).
The separately fixed [variance-only study](https://github.com/awa-si/nautilus/blob/main/docs/hsmm-emission-calibration.md)
failed its clean HMM floor. The later [causal noise-routing study](https://github.com/awa-si/nautilus/blob/main/docs/hsmm-noise-routing.md)
passed on fresh synthetic episodes, but is a two-expert forecast policy rather
than one changed core likelihood; its [market admission](https://github.com/awa-si/nautilus/blob/main/docs/hsmm-market-routing-admission.md)
was rejected by the original artifacts' feature, precision and prior contracts.
Constant mean shift remains unresolved. These results do not widen this
model-library qualification or change defaults. A new fit needs its own
prospectively fixed protocol and admissible evidence. No market production or
neural-context promotion follows.
