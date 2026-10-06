# HSMM correctness research work gates

This file defines the execution sequence for the independently verified HSMM-core milestone. The acceptance contract remains in [testing.md](testing.md); current package status remains in [state.md](state.md).

## Gate 0 — Freeze the mathematical contract

**Goal:** make the target semantics explicit before building a reference oracle.

Required decisions and fixtures:

- canonical latent variable is `(state, episode_age)` for the causal path;
- duration support is `1..D`;
- transition occurs only when an episode ends;
- self-transition starts a new episode and resets age to one;
- emission at timestep `t` scores the state active at `t`;
- causal boundary `t-1 -> t` uses only information available through `t-1` before scoring `x_t`;
- non-causal segment likelihood uses one duration draw per segment and one transition per segment boundary;
- impossible support is represented by exact zero mass / `-inf` log mass.

**Acceptance:** a small set of hand-written examples for `K=1`, `D=1`, one self-transition case, and one impossible-duration case has an unambiguous expected path probability written directly from the factorization.

**Stop condition:** do not implement the exhaustive oracle if any factorization or indexing rule is ambiguous.

**Current status:** complete. The frozen semantics are in [hsmm-mathematical-contract.md](hsmm-mathematical-contract.md). The audited non-causal duration-dependent Viterbi predecessor-collapse defect is fixed and regression-tested; Gate 1 is unblocked.

## Gate 1 — Independent exhaustive path oracle

**Goal:** implement a deliberately slow mathematical oracle under `tests/reference/`.

Planned files:

```text
tests/reference/__init__.py
tests/reference/hsmm_exact.py
tests/reference/test_exact_likelihood.py
tests/reference/test_exact_filtering.py
tests/reference/test_exact_viterbi.py
```

`hsmm_exact.py` must enumerate valid latent episode/state paths directly from raw initial, duration, transition, and emission log probabilities. It must not call production inference helpers listed in `testing.md`.

The oracle must expose enough information to compute independently:

- full path log probability;
- total log likelihood via exhaustive log-sum-exp;
- causal prefix evidence;
- `P(state_t, age_t | x_0:t)`;
- state marginal;
- age marginal;
- probability that the active episode ends before the next observation;
- exact MAP state path and its score.

**Acceptance:** oracle results match manually calculated tiny fixtures from Gate 0.

**Current status:** complete. `tests/reference/hsmm_exact.py` independently enumerates causal state-age paths and non-causal segment paths, exposes likelihood, prefix evidence, joint/state/age posterior derivation, episode-end probability, and MAP outputs, and is guarded by a test forbidding imports from `nhsmm`. Hand-check fixtures cover `K=1/D=1`, deterministic duration two, self-transition age reset, impossible support, a manual two-timestep segment likelihood, and the duration-dependent MAP counterexample.

**Stop condition:** oracle and manual fixture disagree.

## Gate 2 — Production likelihood equivalence

**Goal:** prove production likelihood recurrences compute the same sum over latent paths as the independent oracle on tractable models.

Matrix should cover at least:

- `K in {1,2,3}`;
- `D in {1,2,3}`;
- `T in {1,2,3,4,5,6}` where exhaustive complexity remains bounded;
- uniform, random, near-deterministic, and structurally impossible support cases;
- duration-independent and duration-dependent transition semantics where supported.

Checks:

- `NHSMM.log_likelihood` vs exhaustive total likelihood;
- final `forward` evidence vs exhaustive total likelihood;
- variable-length batches vs per-sequence exhaustive values.

Run the oracle primarily in Float64. Float32 is a separate numerical gate.

**Acceptance:** every maintained matrix case passes the Float64 tolerance recorded in the test; no expected-failure exemptions.

**Current status:** complete. Production causal and non-causal forward likelihoods plus public `log_likelihood` reduction are checked against the independent exhaustive reference across uniform, random, near-deterministic, and impossible-support cases spanning `K={1,2,3}`, `D={1,2,3}`, `T=1..6` representative tractable combinations, with variable-length batch equivalence. Gate 2 exposed and fixed two core numerical-contract defects: non-causal DP temporaries now preserve the input dtype, and impossible valid sequence mass remains exact `-inf` instead of being coerced to the Float32 minimum sentinel.

**Stop condition:** any mismatch larger than the justified numerical tolerance is treated as a model-core defect, not a test relaxation opportunity.

## Gate 3 — Causal filtering and posterior equivalence

**Goal:** verify every causal prefix posterior, not only final likelihood.

For every tractable prefix compare exhaustive conditioning with:

- normalized causal `NHSMM.forward` output;
- `filter_model_sequence`;
- public `filter_step` with independently supplied normalized probabilities;
- `HSMMFilterRuntime.step`.

Checks include:

- joint `(state, age)` posterior;
- state posterior;
- age posterior;
- prefix evidence;
- episode-end probability;
- exact impossible-support cells.

**Acceptance:** all outputs agree with the independent oracle within the dtype-specific numerical contract.

**Current status:** complete. Causal batch forward, `filter_model_sequence`, public `filter_step`, and `HSMMFilterRuntime.step` are independently checked against exhaustive prefix posteriors. The maintained Gate-3 matrix covers uniform, random, near-deterministic, and impossible-support scores across `K={1,2,3}`, `D={1,2,3}`, and tractable sequence lengths. Joint `(state,age)` posterior, state marginal, age marginal, prefix evidence, episode-end probability, boundary-transition joint mass, next-episode state mass, next-state prior, and state-change probability all agree with the oracle.

**Stop condition:** production paths may not be considered mutually validating when they disagree with the independent oracle.

## Gate 4 — Viterbi/MAP equivalence

**Goal:** prove decode selects the globally highest-probability valid latent path.

Compare production decoding with the exhaustive maximum path for the same small-model matrix.

Required cases:

- ties with deterministic documented tie behavior where relevant;
- self-transition episode reset;
- `D=1`;
- forced duration;
- forced transition;
- competing paths whose local greedy choice differs from the global MAP path.

**Acceptance:** state path and MAP score agree with the exhaustive oracle for all maintained cases.

## Gate 5 — Degenerate and structural invariants

**Goal:** lock the semantics that are easy to violate during optimization/refactoring.

Required invariant families:

- `K=1`: no state uncertainty, only duration/age uncertainty;
- `D=1`: every timestep starts a new episode;
- `T=1`: only initial + first emission contribute;
- self-transition resets age to one;
- impossible duration/transition support never receives probability mass;
- right padding cannot alter valid prefixes;
- batching cannot alter a sequence result;
- causal outputs are invariant to future observation changes;
- normalized posterior sums to one on every valid timestep.

**Acceptance:** invariant tests pass in both representative Float64 and Float32 configurations where meaningful.

## Gate 6 — Gradient correctness

**Goal:** verify training follows the intended likelihood, not merely a finite differentiable expression.

Use tiny Float64 models and `torch.autograd.gradcheck` or an independently coded central finite-difference objective.

Parameter groups:

- initial logits;
- duration logits;
- transition logits;
- Gaussian emission location/scale parameters;
- Student-t emission parameters where public training supports them;
- end-to-end causal and non-causal log likelihood.

The numerical reference must perturb the actual parameter under test and recompute the scalar objective from fresh forward evaluation.

**Acceptance:** all maintained differentiable parameter groups pass the documented gradient tolerance. Parameters intentionally excluded from optimization are checked separately for frozen/no-gradient semantics.

**Stop condition:** finite gradients alone are insufficient; numerical derivative disagreement is a correctness failure.

## Gate 7 — Property-based state-space exploration

**Goal:** cover combinations not practical to enumerate manually.

Introduce Hypothesis only if it is accepted as a project dev dependency; otherwise use a deterministic in-repo generator with shrinking/reproduction metadata.

Properties:

- posterior normalization;
- finite outputs for valid finite inputs;
- exact preservation of impossible support;
- causal prefix invariance;
- right-padding invariance;
- batch-vs-single equivalence;
- streaming-vs-batch equivalence;
- self-transition age reset;
- maximum age never exceeds `D`;
- likelihood monotonic decomposition identities used by the reference model.

Every randomized failure must print or persist a deterministic reproducer.

**Acceptance:** the maintained property budget passes repeatedly without flakes.

## Gate 8 — Numerical precision envelope

**Goal:** distinguish mathematical correctness from implementation precision.

Measure the same reference matrix under Float64 and Float32.

Record separately:

- likelihood absolute/relative error;
- finite log-posterior error;
- joint posterior probability error;
- state marginal error;
- age marginal error;
- batch-vs-filter/runtime error as sequence length grows;
- extreme-logit behavior near deterministic support.

Float64 establishes the tight mathematical-reference baseline. Float32 limits must be empirical upper bounds with margin, not arbitrary round numbers chosen to make tests pass.

**Acceptance:** bounds are documented in tests or `testing.md`, remain below the level at which state/MAP/boundary semantics change, and long-sequence recursion remains stable.

## Gate 9 — Statistical recovery remains separate

**Goal:** confirm that mathematically correct inference still recovers known synthetic structure.

Re-run only the controlled package validation materially affected by HSMM-core changes:

- duration recovery;
- latent-state recovery;
- learning/prediction;
- transition-context recovery when transition semantics are touched.

This gate validates identifiability/usefulness of the implementation under known data-generating processes. It is not evidence for the recurrence itself; Gates 1–8 provide that evidence.

**Acceptance:** applicable maintained validation thresholds remain green without weakening thresholds because of the correctness work.

## Gate 10 — Final package acceptance

Required final checks:

```bash
ruff check nhsmm tests scripts
black --check nhsmm tests scripts
git diff --check
pytest -q
```

Before declaring the milestone complete, review that:

- reference code does not share production inference logic;
- every fixed discrepancy has a permanent regression test;
- no tolerance was widened without measured evidence;
- current `model.md`, `testing.md`, and `state.md` match observed behavior;
- no downstream Nautilus result is used as evidence for core mathematical correctness.

**Milestone completion claim:**

> The maintained HSMM recurrence and outputs are independently verified against exhaustive small-model reference calculations, gradient checks, structural invariants, and measured numerical precision bounds.

Anything stronger, such as proof for arbitrary floating-point executions or universal statistical identifiability, requires a separate formal claim and evidence.

## Recommended execution order

Execute Gates 0–4 first as one mathematical-reference workstream. Gates 5–8 then harden structural, differential, and numerical behavior. Gate 9 is empirical revalidation. Gate 10 is the final package gate.

A gate does not pass because later gates are green. Any failure returns work to the earliest gate whose assumption or implementation is invalidated.
