# NHSMM State

Current package readiness and next package-level boundary. Detailed model semantics are in [model.md](model.md), configuration and tuning in [configuration.md](configuration.md), validation semantics and accepted package evidence in [validation.md](validation.md), verification policy in [testing.md](testing.md), and release procedure in [release.md](release.md).

## Current status

**Phase:** package-core hardening and validation are complete enough for the next integration gate. The current model core has been audited for distribution semantics, HSMM recursion, variable-length handling, runtime state, context routing, and Neural context modulation.

Current maintained defaults and contracts:

- ModelConfig.use_context_encoder=False; the internal ContextEncoder is opt-in.
- External context remains independently supported through context_dim.
- The canonical production causal path is the NHSMM causal filter/runtime contract documented in [model.md](model.md).
- n_states=3 remains the current Nautilus baseline from the maintained 3-vs-4 usability study.
- Historical machine-readable validation evidence is retained under [validation/README.md](validation/README.md).
- The full controlled package acceptance suite was re-run batchwise on 2026-10-06 after the ContextEncoder opt-in migration; duration, state recovery, learning/prediction, and transition-context gates all passed.

## Accepted package-core evidence

Controlled package evidence currently supports these narrow claims:

- **Duration context:** PASS for strong/moderate recovery with a bounded null control.
- **Latent-state recovery:** PASS up to permutation when the controlled acceptance harness explicitly uses K-Means initialization.
- **Learning and held-out prediction:** PASS for the maintained strong/moderate/weak contracts; null remains a negative control.
- **Transition context:** PASS for strong/moderate recovery with near-zero null effect under the maintained refinement policy.
- **Multi-component identifiability:** package-core PASS across the frozen controlled gate.
- **General-context robustness:** package-core PASS under split-fit direction replication.

These claims are package/mechanism evidence only. They do not establish trading utility, market semantics, or downstream predictive value. See [validation.md](validation.md) for the maintained metrics, thresholds, and reproduction commands.

## Production hardening closed on 2026-10-05/06

### Distribution and runtime

Closed and regression-tested:

- distribution initialization no longer double-applies context;
- emission/K-Means initialization preserves registered parameter identity;
- runtime score caches invalidate after parameter replacement;
- decode normalizes list input before batch-shape access;
- log_likelihood rejects positive infinity rather than mapping it to a finite maximum;
- unknown distribution activation names fail closed.

### Mathematical and PyTorch correctness

Independent checks verified:

- causal state-age filtering against probability-space brute force;
- non-causal forward likelihood and Viterbi against complete small-model segment/state enumeration;
- batch vs single-sequence likelihood and decoding consistency;
- Gaussian and Student-t emission likelihoods against PyTorch references.

Closed defects:

- variable-length list likelihood/training padding leakage;
- padded-row NaN autograd in forward recursions;
- duplicate public external-context routing;
- max_duration=1 empty duration-regularizer NaN;
- scheduled distribution temperatures excluded from the standard optimizer.

### Model boundary and lifecycle

Closed and regression-tested:

- public inference requires initialized distributions with a defined runtime error;
- observation rank, feature width, and finiteness validate at the model boundary;
- empty-sequence likelihood behavior is consistent;
- structurally incompatible optimize(..., cfg=...) configs fail closed;
- optimization preserves the caller's train/eval mode across restart-created distributions and validation failures;
- model construction owns a resolved config copy, so explicit encoder injection cannot mutate a reusable caller config and leak Neural opt-in into later models;
- variable-length external context validates against `context_dim` independently of observation feature width.

### ContextEncoder opt-in

The internal learned observation-derived context path is optional:

- default: no ContextEncoder, no distribution context networks unless external context capacity is explicitly configured;
- use_context_encoder=True enables the canonical internal encoder;
- passing an explicit encoder is also an opt-in;
- runtime, filtering, diagnostics, artifacts, and tests support both paths;
- artifact v1 accepts encoder_config=None and remains compatible with older encoder-bearing artifacts.

### Neural semantics

Closed and regression-tested:

- categorical log_prob rejects fractional, non-finite, and out-of-support labels instead of silently truncating them;
- emission context deltas remain state-centered while respecting max_delta;
- the unused internal Neural.alpha parameter was removed;
- temperature remains categorical-logit semantics and does not alter continuous emission location parameters;
- categorical temperature controls fail closed for non-finite/non-positive values and their stored parameters are frozen;
- context-free `delta_scale` controls are frozen because they are behaviorally inactive without context networks.

## Verification

Latest standalone core gate after the package audit fixes:

    ruff check nhsmm tests scripts: PASS
    black --check nhsmm tests scripts: PASS
    git diff --check: PASS
    focused regression suite: 131 passed
    python -m pytest -q: 362 passed, 3 skipped

The three skips are CUDA-only tests on the CPU workspace. A public-API artifact/inference/runtime round-trip also passed: artifact save/load preserved batch likelihood exactly, artifact-loaded and `load_inference_model(...)` runtimes produced identical posteriors across 17 causal streaming steps, state/age posteriors remained normalized, reset semantics passed, inference parameters were frozen, and variable-length list likelihood remained finite.

Full controlled revalidation on 2026-10-06 also passed:

- duration context: strong/moderate/null PASS;
- latent-state recovery: strong/moderate/null PASS;
- learning/prediction: strong/moderate/weak learning+usefulness PASS; null learning-control PASS with usefulness not applicable;
- transition context: strong/moderate/null PASS.

The validation harnesses were run in scenario-sized batches with persistent JSON artifacts under the AWA workspace work store. See [validation.md](validation.md) for the current measured summaries.

## Nautilus usability boundary

The maintained 2026-10-05 3-vs-4 state-count study favored three states:

- 3-state runs were healthier and had slightly better mean OOS log likelihood per step;
- four states added soft posterior capacity without a hard-state usage gain;
- operational baseline therefore remains n_states=3.

That study predates the current real-data integration gate and is not a substitute for a fresh current-feature replay.

## Research/correctness milestone — independently verified HSMM core

The next package-level research milestone is to raise the HSMM core from regression-tested correctness to independent mathematical verification. This milestone is package-internal and precedes any renewed downstream Nautilus acceptance claim.

Execution is staged by [hsmm-correctness-work-gates.md](hsmm-correctness-work-gates.md).
Gate 0 mathematical semantics are frozen in [hsmm-mathematical-contract.md](hsmm-mathematical-contract.md); Gate 1 is currently blocked by the audited non-causal duration-dependent Viterbi predecessor-collapse defect.

The milestone is complete only when all of the following are observed:

- an intentionally simple exhaustive reference implementation enumerates all valid segment/state paths for small `K`, `D`, and `T` without importing production recursion, filtering, hazard, or runtime helpers;
- production `log_likelihood`, causal filtering, state/age posteriors, episode-boundary probabilities, and Viterbi decoding agree with that independent reference across the maintained small-model matrix, including `K=1`, `D=1`, `T=1`, self-transitions, impossible durations, and near-deterministic probabilities;
- causal batch forward, stateless model filtering, and stateful runtime agree within an explicitly measured Float32 numerical contract, while a Float64 reference gate uses materially tighter tolerances;
- end-to-end likelihood gradients for initial, duration, transition, and emission parameters pass `torch.autograd.gradcheck` or an equivalent finite-difference reference on small Float64 models;
- property-based tests cover normalization, causal prefix invariance, padding invariance, streaming equivalence, support preservation, and episode-age reset semantics;
- the full canonical pytest/static gate remains green after the new correctness harness is integrated.

Passing this milestone supports the precise claim: **the maintained HSMM recurrence and outputs are independently verified against exhaustive small-model reference calculations**. It does not imply formal proof for arbitrary floating-point executions, statistical identifiability on every dataset, or downstream trading utility.

No unresolved package-core mathematical defect is currently known. The open work is stronger independent verification, not a known bug fix.

Downstream Nautilus integration remains intentionally deferred and outside this milestone.
