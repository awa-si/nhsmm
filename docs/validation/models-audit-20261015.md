# Audit of nhsmm/models — 2026-10-15

Seven reproduced findings: one P1 and six P2. No model fixes are applied in this
checkpoint. Further learning experiments remain paused pending the correctness
repairs and their required verification.

## Scope and source identity

Audited the complete models directory on regime-v2@e9900c4cf46a27531b99e595c4b208bf92830536:
models/__init__.py (exports) and all1851 lines of models/base.py.
Base Git blob47571aca9bcaea304464a925d96e9ad0bd911df5 was fetched through the
connected GitHub connector and independently matched to the executing worktree.
The local pre-audit mirror39a831a has the same repository tree as remotee9900c4.
Repository instructions and material model/testing contracts were already loaded.

Review covered component construction, context/causal timing, causal and
noncausal forward, finite and geometric-tail recursion, all Viterbi branches,
likelihood/predict/decode dispatch, input shape/dtype/length handling, K-Means and
restart initialization, snapshots, losses, chunk accumulation, scoring,
convergence wiring and transition refinement. Coupled dependencies received
targeted inspection/tests; this is not a new audit of every dependency.

## Findings

### M01 — P1: Empty observations corrupt the training objective and score

Source: nhsmm/models/base.py, lines 1259, 1553, 1651, 1690.

Nonempty training score -12.54644775390625 becomes -3.4028234663852886e38 with one empty sequence and -inf with two. A zero-sequence batch is accepted with score0 and an optimization history entry.

Impact: Restart ranking and convergence can be dominated by the empty-sequence sentinel instead of observed data.

Proposed repair: Define and enforce empty-input semantics before mutation: reject empty training batches; either reject empty entries or make supported empty entries neutral in summed objectives.

### M02 — P2: Missing refinement context is rejected after joint training mutates weights

Source: nhsmm/models/base.py, lines 1706, 1837.

optimizer_initialization=current, max_iter1, context_dim1, transition_refine_steps1 and context=None raises ValueError after weights have changed.

Impact: An invalid request returns failure with a partially updated model.

Proposed repair: Validate refinement prerequisites before any initialization or optimizer step; assert state preservation on invalid requests.

### M03 — P2: Internal encoder Float64 conversion fails because observations are forced to Float32

Source: nhsmm/models/base.py, lines 1336, 1343, 1384.

model.double().log_likelihood(float64 observations) raises RuntimeError: Input type (float) and bias type (double) should be the same.

Impact: The regular nn.Module dtype conversion fails for the internal-encoder path.

Proposed repair: Use the live model parameter dtype/device consistently at input and restart boundaries, and verify encoder Float64 plus Float32 controls.

### M04 — P2: Noncausal segment emission sums clip valid Float64 density and gradients

Source: nhsmm/models/base.py, lines 646, 705.

K1,D1,x1e20,Gaussian mean0, variance softplus(0): expected LL -7.213475493573564e39; noncausal LL -3.4028234663852886e38 and mean gradient0. Causal finite/tail controls retain the density and nonzero gradient.

Impact: The segment score cap changes the estimator for finite Float64 values beyond the Float32 lower bound; Viterbi does not use the same cap.

Proposed repair: Remove categorical score clipping from accumulated continuous emission likelihoods; test independent density and gradient agreement.

Practical limit: Extreme Float64 counterexample. Moderate Float32 controls agree; no evidence that this affected the standardized frozen Trading artifact.

### M05 — P2: Causal normalization creates NaNs for zero evidence and the public wrapper hides them

Source: nhsmm/models/base.py, lines 534, 578, 1270.

A valid-shaped SequenceSet with K1 emissions[-inf,0] produces two NaN alpha entries; the public likelihood wrapper returns finite Float32 minimum.

Impact: A mathematically impossible event becomes NaN internally and a finite sentinel externally.

Proposed repair: Guard zero normalizers and preserve impossible evidence or raise a clear support error; test parity with the validating filter.

Practical limit: Exact-zero emission support supplied through the public forward score container/custom component extension, rather than the stock positive Gaussian density.

### M06 — P2: Finite noncausal Viterbi returns an arbitrary path when no terminal path exists

Source: nhsmm/models/base.py, lines 1118, 1161.

Supported injected Duration subclass hard-masks to duration2; T1 has LL -inf, yet decode returns[0].

Impact: A returned path need not have positive support under the same model.

Proposed repair: Check terminal MAP score finiteness before backtracking; cover finite/tail and duration-independent/dependent transitions.

### M07 — P2: Empty-sequence forward and prediction contracts disagree across branches

Source: nhsmm/models/base.py, lines 596, 682, 707, 1284.

Causal forward returns shape[1,0,1,1]; finite noncausal forward raises IndexError and tail noncausal raises stack-empty RuntimeError. predict(mode=log_likelihood) returns a list for T0 whereas direct log_likelihood returns Tensor.

Impact: Dispatch changes shape/type or raises implementation errors for empty data rather than applying an explicit contract.

Proposed repair: Handle emptiness consistently before branch-specific recursion, and dispatch likelihood mode before Viterbi-specific empty returns.

## Verification

- Entire model source parsed successfully.
- Existing affected suite:398 passed,3 CUDA-only skips,35.73 seconds. This includes
  tests/reference, test_training, optimizer accumulation/controls, causal tails,
  model/filter and context hardening. It is not an unrestricted package-suite run.
-20 additional bounded synthetic probe cases reproduce the seven findings.
  Moderate-density and causal controls are included, not only failure cases.
- The persistent reproducer's assertions, Ruff and git diff --check pass.
- No core source or existing test was modified. The new reproducer is a frozen
  audit tool, not a CI acceptance test: it asserts that the documented bugs remain
  observable on the pinned model blob and refuses a changed model source.
- Probes include tiny synthetic optimizer calls to expose validation side effects;
  no research fit, dataset tuning, GT input or holdout scoring was performed.
  Oracle remains OFF. The frozen Trading artifact is untouched.
- Existing five historical evidence-test failures are outside this focused run
  and remain unresolved.

Reproduce from the repository root with the prepared environment:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 PYTHONPATH="$PWD" \
/work/env/python/bin/python scripts/audit_model_boundaries_20261015.py \
  --output /work/checkpoints/models-audit-reproduction-20261015.json
```

Machine evidence: models-audit-20261015.json. Additional logs/JUnit are under
/work/checkpoints/models-audit-{existing,reproduction}-20261015.*.

## Interpretation and next gate

The398 existing passes validate covered ordinary cases, not the newly exposed
boundaries. M01 matters before learning-budget work because corrupted training
scores can invalidate restart selection and stopping diagnostics. Fix M01 and
M02 first, then zero-support/decoding correctness M05/M06 and dtype/empty-input
contracts M03/M07; remove continuous-score clipping M04 with density references.
Each repair needs a failing regression first and the applicable model/core suite.
Do not start the learning-budget study until the repaired boundary contracts
are verified.

Do not classify the deliberate noncausal completed-segment objective versus the
causal censored-prefix objective as a newly discovered mismatch: current
independent reference tests explicitly distinguish them. Geometric D+ tails and
same-state episode renewals are also established semantics.

No finding proves the cause of the frozen Trading result. M04 is an extreme
Float64 case; M05 requires exact-zero emission support. The known frozen
experiment uses bounded standardized Gaussian observations without an internal
encoder, and no empty-window evidence was established in this audit.
