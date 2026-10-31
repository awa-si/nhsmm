---
title: Model verification
description: Canonical test ownership, verification commands, and model-core evidence.
type: guide
status: active
language: en
owner: NHSMM
created: 2026-09-26
updated: 2026-10-31
---

# Model verification

Canonical model, mathematical reference, distribution, training, runtime, artifact
and validation-snapshot tests have migrated to the [Nautilus ML stack](https://github.com/awa-si/nautilus/tree/main/tests/ml_stack/nhsmm).
There is no duplicate tests/ tree in this model package. Model implementation
remains here; this core never imports the host.

Install this source checkout editable in the research environment, then run from
Nautilus:

```bash
python -m pytest -q tests/ml_stack/nhsmm
python -m pytest -q tests/ml_stack/nhsmm/reference
```

The independent probability-space HSMM reference moved unchanged. Assertions,
numerical tolerances, gradient/likelihood/filtering/Viterbi/finite-D+ checks, and
all surviving core behavior tests are retained. Run the full model suite for
changes to shared inference, distributions, context, configuration or artifacts.
Do not hide failures with deselections or changed tolerances; report CUDA skips.

Oracle/benchmark helpers now live in Nautilus adapter.qlib.nhsmm_research.
Frozen fixtures/input reports live in tests/ml_stack/fixtures/nhsmm. Tests no
longer require research files in this core checkout. Run CLIs from Nautilus
with python -m and the new module name. Oracle remains OFF and opt-in.

## CI ownership without a cross-repository secret

Local core static checks remain ruff check nhsmm scripts and
black --check nhsmm scripts. The public smoke workflow runs only this repository:
install the qualified CPU dependency profile, pip integrity and
scripts/smoke_cpu.py. Release reuses that same workflow, then builds and verifies
the wheel and sdist. The installed wheel runs the same smoke outside the checkout.
Neither workflow reads a private repository or needs a custom Actions secret.

doc:nhsmm.public-package-smoke covers finite/D+ Float32/64 causal batch/stream
parity, trusted artifact save/load, exact learned weights and fresh reset.
This is a public package integration check, not the full mathematical test suite.

The full canonical suite stays in private awa-si/nautilus under
.github/workflows/nhsmm-core.yml. That workflow is manual-only
(`workflow_dispatch`) and requires `nhsmm_core_sha`: the exact full 40-character
commit to test from this repository's `v3` branch. It checks out that commit and
asserts the resolved SHA before running tests; its evidence records both the
NHSMM core and Nautilus test-owner SHAs plus source hashes and JUnit results.

For the next NHSMM code change, manually dispatch this workflow against that
change's exact commit SHA. Do not use the previous pin, an automatic push run,
or the public package smoke as a substitute. Require the successful
source-matched full gate before release. The public release workflow cannot
automatically read or enforce this private run. No private tests are copied
into this package, and no token value is required.

## Source-bound repair and context evidence

Current behavior is owned by [the model/runtime contract](model.md) and the
[HSMM mathematical contract](hsmm-mathematical-contract.md). The following
reports preserve historical results for their named source revisions; their
pass counts do not certify the current branch head.

- **Mathematical repair (2026-10-19):** the source-bound audit corrected five
  issues: normalized non-causal likelihood, exact first-order tail-boundary
  gradients, segment support zeros, reachable-duration support, and artifact
  precision. The fixed snapshot added 59 regressions, matched 540 independently
  enumerated forward/MAP cases (maximum evidence residual
  `2.220446049250313e-16`), and passed 1,156 tests with three CUDA skips;
  Ruff and Black passed. Public method signatures and state-dict paths were
  unchanged, and a causal v2 reference artifact retained its exact fixed-prefix
  score. No real training ran.
  A three-repeat CPU A/B (`K=3,F=2,D=32,B=1,T=256`) measured median forward
  `0.1754 -> 0.1981 s` and backward `0.1919 -> 0.1635 s`; this mixed result
  supports no general speedup claim. Full hashes and run identity are in the
  [Nautilus repair record](https://github.com/awa-si/nautilus/blob/main/docs/nhsmm-correctness-repair.json).
- **External-context timing and streaming fallback:** exact-source reports retain
  timing checks (1,209 core tests, three CUDA skips, 30 host harness regressions),
  base-score checks (1,215 core tests, three CUDA skips; 162 host integration/TA
  tests plus 11 subtests), and masked-context checks. See the
  [timing report](https://github.com/awa-si/nautilus/blob/c45d94f4a8b6c87870ee2dc8f775c9609128c027/docs/nhsmm-causal-context.json),
  [base-score report](https://github.com/awa-si/nautilus/blob/cf3a8ef59fbbdf2484f5679307568b2c812f38bf/docs/nhsmm-context-fallback.json), and
  [availability report](https://github.com/awa-si/nautilus/blob/5adf39f034e02c72f2c5fd42b133669233865370/docs/nhsmm-masked-context.md).
  These are historical synthetic/integration results, not evidence of market
  advantage.


The canonical test_forecast_precision.py module checks rare finite/D+ end and state-change risks against 80-digit Decimal references in Float32/Float64, hard-support forecast gradients against independent probability formulas, geometric first/second derivatives, result identities, and separate duration/transition target learning.


## Central tunable verification

The canonical test_central_tunables.py checks invalid/valid overrides, actual probability laws, context/encoder propagation, Adam/scheduler/patience settings, supervised precedence and current/legacy artifact predictions. Host test_nhsmm_config_tuning.py checks the bounded search policy, grid, split separation, request guards and native Qlib/MLflow recording.

Alternate optimization-config regressions verify the actual annealing temperatures,
main/refinement Adam and clipping settings, and rejection of ignored construction
or supervised-only overrides before any parameter update.

2026-10-21 optimizer-config verification: 125 focused tests passed. The full
model plus tuning/auditor run passed 1474 tests with three CUDA skips; one
metadata setup error was resolved by exporting the original pyproject.toml.
That unchanged test passed separately (1475 distinct passes total). No model
code changed between those runs; Ruff and Black passed for the complete core.

Temperature-boundary verification (2026-10-21): test_temperature_numerics.py
adds 24 cases, including 100-digit Decimal schedule references, bit-exact
ordinary arithmetic, actual extreme annealed/constant training temperatures,
and rejection of an underflowing positive ratio/target combination before
model mutation. The complete model plus tuning/auditor suite passed 1499 tests
with three CUDA skips. Initial focused testing exposed an invalid zero-ratio
test setup; it was corrected to a valid underflowing configuration, without
changing configuration bounds. Corrected focused run: 39 passed.

Training-finiteness verification (2026-10-21): 18 new injected-failure cases
check objective/score finiteness, NaN/Inf gradients, total-norm overflow from
finite elements, mode/trainability restoration and absence of successful
completion telemetry. The focused training bundle passed 57 tests; the full
model plus tuning/auditor suite passed 1517 with three CUDA skips. The prior
source reproduced a NaN-gradient Adam update corrupting emission locations.
The initial reproduction test used a nonexistent parameter name; corrected
mu injection established the actual failure. Ruff and Black passed.

The canonical test_supervised_setup.py injects Adam-construction and mode-switch
failures in training/eval and base/context-only modes. It verifies unchanged
weights and restored trainability, including parameters frozen by the caller,
and retains coverage of the no-trainable-parameters cleanup path.

2026-10-21 setup-restoration gate: 28 focused checks passed; full model plus
affected native ML-stack bundle passed 1566 with three CUDA skips. The first
regression reproduces incorrect frozen base flags after Adam setup failure
on core 371e19c. No defaults, supervised scores or training budgets changed.

## Exact finite-support boundary derivatives

Canonical test_hsmm_boundary_derivatives.py adds 45 cases for the exact D+
h_tail=1 boundary: raw-parameter path-enumeration gradients in Float32/64,
D=1/2/3, self resets, mixed interior/boundary states, forecast first and mixed
second derivatives, next-state derivative conservation, ragged and empty
prefixes, one-sided numerical differences, projected supervised Adam direction,
and single evaluation of stochastic heads. The original 2026-10-21 gate
restricted the score-boundary VJP to first order. That restriction still
applies to marginal likelihood. At source 8c0998c, supervised filtering
separately supports exact second derivatives and rejects third-derivative
graphs, as covered by the usable-supervised-HSMM gate below.
Forecast values and artifact schemas remain unchanged. This is mathematical
verification, not empirical trading qualification.

2026-10-21 boundary gate: 45 new cases pass; the final same tests on core
84d38da have 44 failures and one pass. Full canonical model plus affected
native Qlib/MLflow, research publication, configuration/input audits and
retained-replay auditor bundle: 1647 passed, three CUDA-unavailable skips.
Ruff and Black pass for the complete core. No defaults, artifact schema,
market fit budget or reserved data changed. Exact source and JUnit evidence
are retained in Nautilus docs/nhsmm-hsmm-boundary-verification.json.

## Long-prefix HSMM and streaming state verification

Canonical test_hsmm_streaming_integrity.py adds 26 cases. Four 1440-row
Gaussian HSMM sequences (finite/D+ and Float32/64) compare every batch and
streaming state-age posterior with a separately constructed NumPy long-double
expanded-state probability chain. Its inputs are raw weights; it never calls
production distribution scoring, hazard or filtering helpers. Final likelihood
per row, one-step state/end probabilities and frozen-duration survival forecasts
are checked independently. Reset starts 128 fresh suffix observations from age
one. Two custom exact-support models retain deterministic 100-row age cycles.

The same module checks variance-floor cache invalidation, inference-tensor
parameter updates without version counters (including incremental LSTM gates),
unchanged persistent state during inference, invalid timestamp rejection before
state advance and valid datetime continuation. Model reset clears runtime caches.
Posterior comparison tolerances are 2e-5 for Float32 and 2e-12 for Float64;
log-evidence per-row tolerances are 1e-5 and 1e-10. These are bounded numerical
checks, not a proof over arbitrary sequence lengths or custom distributions.

2026-10-21 streaming gate: 26 new cases pass; the same final tests on core
c25da77 have 17 failures and nine passes. Focused runtime/encoder bundle:
47 passed. Full canonical model plus affected native Qlib/MLflow/RD-Agent,
research publication and input/configuration/retained-replay audit bundle:
1673 passed, three CUDA skips. Complete core Ruff and Black pass.
No model default or artifact-schema change, new market fit or reserved test
score. Evidence: Nautilus docs/nhsmm-streaming-verification.json.


## Viterbi common-offset gate

The host test_viterbi_numerics.py verifies MAP decisions under large shared
Gaussian density offsets, dynamic finite/D+ raw-weight path enumeration,
ragged empty/single-row inputs and 1440-row weak-preference causal HMMs.
Existing exhaustive decoder tests retain exact ties and hard support.
See [the numerical contract](model.md#viterbi-common-offset-stability)
for offset cancellation, noncausal endpoint limits and rounding boundaries.


## Causal filter emission-offset gate

Canonical host test_filter_offset_invariance.py checks identical Gaussian
emissions with large finite observations, ragged empty/single-row sequences,
batch/runtime posteriors, latent-law evidence gradients, and raw-parameter
path enumeration for likelihood/filter scores and all parameter/E/U gradients.
Finite/D+ durations, exact h_tail=1 and differing hard transition support sizes
are included. The host report records baseline failures, source/JUnit hashes
and full native ML-stack validation.


## Artifact publication and real Gaussian derivative gates

Canonical host test_artifact_publication.py injects serialization, file-sync
and replacement failures for absent and existing targets, verifies that the
target is unchanged until serialization completes, and checks Float32/64
Gaussian/Student-t round trips and existing POSIX permissions.

test_gaussian_offset_derivatives.py uses real Gaussian scoring and independently
enumerated prefix-conditioned occupancies to check mean, raw-variance and
observation gradients at large positive and negative observations. It covers
HMM, finite HSMM and D+ (D=1/3), Float32/64, and marginal/filter objectives.
Derivative errors are compared after division by their natural observation
scale (x or x squared); it does not claim arbitrary absolute precision.


Scoped CPU qualification and operational limits: [production.md](production.md).

The former 2026-10-22 cross-repository test-owner pin was superseded on
2026-10-24 by the owner-local canonical workflow above. The public smoke workflow
runs manually through `workflow_dispatch`; `workflow_call` lets the manually
dispatched release workflow reuse it. It has no push or pull-request trigger.

## Independent model-mode qualification

Canonical test_model_modes.py verifies static construction without neural
networks, raw-weight path-enumeration scores/posteriors, synthetic supervised
HMM/HSMM fitting, v3 artifact round trips, continued fitting, streaming and
external-context fallback. Public scripts/smoke_cpu.py also exercises HMM,
finite/D+ HSMM and internally conditioned NHSMM in Float32/64. Profile
selection adds no new inference algorithm or artifact field. See model-modes.md.

## HMM filtering performance regressions

Canonical test_hmm_optimized_filter.py checks bit-exact D=1 no-gradient steps
against the unchanged general kernel at K=2/3/6 and Float32/64, continued
selection of the autograd path, invalid reachable duration rejection,
three-state neural HMM prefixes against independent raw path enumeration,
ragged empty/singleton inputs and future-prefix causality. Existing full model
and serving gates still apply. Timing comparisons use identical artifact
weights/inputs and alternate old/new code; score tolerances remain unchanged.

## Extended HSMM and HMM preservation gate

Canonical test_hsmm_extended_validation.py checks 17/64-age finite and D+
batch/stream frontiers against independent raw-weight expanded Markov chains,
ragged dynamic age-dependent transition/duration adjoints against a separate
probability recurrence, interior Hessian-vector products and zero-sensitivity
duration suffix derivatives with exact hard support and separated scores.
Hazard batching excludes padded and unused final boundaries. Full model and
existing HMM optimization/usability gates remain required.

## Usable supervised HSMM gate

Canonical test_usable_hsmm.py verifies boundary provenance and censoring counts,
actual temperature/gate-aware smoothed laws, unchanged emissions/initial weights,
invalid-fit immutability, stationary state-age priors against independently
expanded Markov chains, batch/stream/reset parity, exact h_tail=1 Hessian-vector
products and explicit third-derivative rejection. Existing HMM gates remain.
Host synthetic calibration is owned by Nautilus usable_hsmm.py through native
Qlib TrainerR/MLflow, with a sealed split/source/candidate budget.

The completed 2026-10-24 native calibration is recorded in
[Nautilus](https://github.com/awa-si/nautilus/blob/1bbcd0b2ffd5d5d8766d702c8a0185ae9b351594/docs/usable-hsmm.md):
nine declared profiles pass; geometric controls retain HMM.
The unchanged core's canonical source pair is NHSMM 8c0998c / Nautilus 37028d7
(1852 passes, three CUDA skips). The later host-only scorer/replay correction
has 134 targeted rechecks; it does not replace or rerun the full canonical gate.
