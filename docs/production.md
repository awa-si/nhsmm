# HSMM CPU qualification

This is a technical model-library qualification, separate from the host's
research-only trading baseline. It does not approve a live trading strategy,
establish profitable signals, or promote retained research artifacts into a
live artifact location.

## Qualified runtime and model scope

The recorded local run uses Linux x86-64, Python 3.14.8, PyTorch 2.14.1+cpu and
NumPy 2.5.4. Float32 and Float64 are exercised. This does not qualify every
version admitted by the broad package dependency bounds, accelerators,
Float16/BFloat16, torch.compile, or arbitrary custom distributions/encoders.

The canonical CPU gates cover causal HMM, finite-duration HSMM and the D+ tail,
right-censored final episodes, supervised observed-joint and prefix-filter
objectives, optional external component context, and canonical internal context.
Use the public normalized filter/runtime APIs for posterior probabilities;
raw absolute forward scores have the finite-resolution limits in model.md.
At exactly h_tail=1, the score correction supports first-order derivatives;
higher-order requests are rejected. Interior/forecast derivative contracts
remain those in the mathematical owner.

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

The smoke and release workflows pin the canonical external test commit.
The smoke workflow also runs on v3 pushes and pull requests. Hosted run
execution, secret provisioning and the broader dependency-version matrix
must be reported separately from local evidence. No package tag or PyPI
publication follows merely from this qualification.
