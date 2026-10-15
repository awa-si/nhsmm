# V3 validation path audit — 2026-10-15

Scope: nhsmm/validation on v3, based on
0a5dbd43d5db5a87fe8d11af67f712a45780f86d. Probabilistic model, inference,
training objective, artifact schema and frozen historical evidence are unchanged.

## Corrections

- Reject negative transition weights before diagonal removal/normalization.
  Row scaling prevents overflow when finite weights approach float64 limits.
- A constant effect cannot be selected under zero amplitude/correlation thresholds.
  Strictly positive amplitude is a necessary condition in both replicas.
- Capture callback output arrays immediately; a reused scratch buffer previously
  erased earlier context points and could turn a replicating effect into a null.
- Use scaled Pearson correlation: vector/finite/non-empty checks, zero for
  singleton/constant vectors, no arbitrary absolute standard-deviation cutoff.
  Very large finite and very small nonconstant effects retain finite correlations.
- Validate non-empty alignment centers and permutation bijections. Scale assignment
  costs to bound intermediate potentials; reject unrepresentable squared distances.
- Transition extraction rejects extra batch/time axes and invalid state/support
  dimensions rather than silently selecting [0,0].
- Preserve each PyTorch submodule's train/eval flag on success and failure for
  all four component probes and snapshots.
- Share split-fit preparation between transition/state effects. Validate integer
  indices and the evidence grid before fitting; reject reuse of one model instance.
  Independent observations, weights, preprocessing and seeds remain caller duties.

## Verification

Before correction: the first 39 new cases produced 38 failures and one pass.
Two additional scratch-buffer cases then failed independently. A separate
large-amplitude direction probe reproduced FloatingPointError in multiplication.
These were defect demonstrations, not successful acceptance runs.

Focused final command (74 passed, warnings treated as errors):
python -m pytest -q -W error tests/test_validation_regressions_v3.py
tests/test_validation_hardening.py tests/test_context_evidence.py
tests/test_state_context_evidence.py tests/test_validation_public_api.py
tests/test_validation_nhsmm_api.py tests/test_validation_snapshot.py

Final complete core suite, no exclusions: 1286 passed, three CUDA-only skips,
zero deselections (218.43s). Final installed regressions: 44 passed
with warnings treated as errors (5.71s); pip check clean. Final host suite:
84 passed, one no-GBM-profile skip and 102 existing warnings (45.00s).
The earlier 1285-test broad run also passed before the last sign-comparison
correction; the final full run above includes that correction and its new test.
Host warnings are existing NumPy/pandas timedelta deprecations and controlled
trusted-artifact warnings; they were not suppressed.

Final run jobs:
- focused: 656f4f9839644e25b05aeb4ac3e96b73
- full core: c783278356b24dbe86f94d9bd402123d
- installed and host: 36c4e2160a144780898c51744966ca69

The exact source/install SHA-256 identities and observed gates are in
[v3-validation-path-final-20261015.json](v3-validation-path-final-20261015.json).
Core and host gates ran in parallel with two OMP/MKL threads each.

All nine installed validation-module files were byte-checked against the candidate.
Static checks: Python compileall, Ruff, Black and git diff --check.
The new tests include an exhaustive assignment reference for K=1..5, permutation
and null guards, callback-buffer reuse, strict shapes, split preflight and mixed
module-mode restoration on both success and exceptions.

## Interpretation

Replicating context responses are descriptive diagnostics. Threshold selection,
multiple comparisons and context-grid choice are not statistically calibrated.
Disjoint array slices alone do not establish independent data or prevent temporal
overlap/leakage in a callback. Alignment by emission means is ambiguous when
states have indistinguishable centers. Snapshots summarize retrospective
likelihood, Viterbi paths and health, not deployable causal filtered features.

No fit/research-quality improvement is claimed. Oracle remains OFF; the existing
late holdout and original reports/proofs were not changed or rescored.
