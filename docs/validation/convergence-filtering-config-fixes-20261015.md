# Convergence, filtering and configuration fixes — 2026-10-15

All eight findings in [the source audit](convergence-filtering-config-audit-20261015.md)
are repaired in one coherent batch. Source base: awa-si/nhsmm@regime-v2,
cf97f6162459b494fc9de43355d4a146e2b91553 (audit documentation); executable
baseline da657e6385c1d83f318ebc2fed231416dbd184e4.

## Changes

| Finding | Repair |
| --- | --- |
| F01 | Reject zero/non-finite evidence before returning an unchecked filter state; cover public/internal kernels, mixed batches and disjoint support. |
| C01 | Keep scores, deltas, callbacks and best-score bookkeeping in Float64 independently of model dtype. |
| F02 | Validate normalized public state mass with absolute log-mass tolerance max(1e-12,8*finfo(dtype).eps), rtol=0. |
| F03 | Validate every valid trace row and exact -inf padding, allowing empty batch/time dimensions and requiring positive latent dimensions. |
| C03 | Export missing history/best scores as null and enforce strict JSON serialization. |
| G02 | Reject invalid scenario names before mapping construction; preserve distinct string identifiers. |
| G03 | Validate model seeds against PyTorch's signed/unsigned 64-bit domain, validation seeds against the runner's legacy NumPy range, and derived evaluation seeds against non-negative default_rng input. |
| G01 | Require an integer, excluding bool, before accepting the validation schema version. |

The strict trace check exposed finite-emission Float32 cancellation in the
existing raw log-normalizer subtraction. Filter normalization now uses
PyTorch log_softmax; four additional large-offset regressions cover both
initialization and subsequent updates. The normalization tolerance was not
relaxed to hide this issue.

The forecast's existing defense against an invalid state remains tested:
construct a valid state, mutate its tensor storage, and verify forecast rejection.
The constructor now rejects that invalid mass earlier during ordinary input.

Artifact/state_dict representations, model parameter dtype, D+ tails,
same-state episode renewals and context timing are unchanged. Valid config schema
version remains 1. Invalid boolean/float schema versions and lossy identifier
coercions are deliberately rejected.

## Validation

- Final focused suite: **213 passed**, 8.75 seconds.
- Broad applicable suite: **1189 passed, 3 CUDA skips, 5 explicit deselections**, 184.08 seconds.
- Updated installed snapshot: **48 passed**, 5.08 seconds; pip check reports no broken requirements.
- Ruff on changed sources/tests, Black on the new regression module, syntax and git diff checks passed.

New regression owner: tests/test_cfc_audit_fixes.py (48 cases). On the original
code, the initial 44 cases yielded 31 failures and 13 passes. The first repair
focused run yielded 208 passes and one failure: the large finite-emission
normalization residual described above. The final implementation includes the
stable normalization repair and four explicit offset cases.

Five previously established historical execution-identity tests are explicitly
excluded from the broad applicable suite; their test code and evidence are
unchanged:

- tests/test_block_mass_smoothing_audit.py::test_completed_batch
- tests/test_block_mass_smoothing_audit.py::test_raw_bytes_tampering
- tests/test_block_mass_smoothing_audit.py::test_missing_replay_certificate
- tests/test_optimizer_bridge_verification.py::test_changed_source_manifest_rejected
- tests/test_optimizer_bridge_verification.py::test_rebound_initial_objective_rejected

Their frozen Python/NumPy/Torch/source identity gates do not match the current
runtime. This batch does not claim an unrestricted all-green historical suite.

## Bounded runtime check

Same host, CPU Torch 2.14.1+cpu, one thread, K=4,F=5,D=64, 32 warmup steps and
256 measured steps repeated three times, fresh seed723 model per process.

| Statistic | Baseline microseconds | Final candidate microseconds |
| --- | ---: | ---: |
| Mean | 2324.408 | 2046.413 |
| p50 | 1842.425 | 1821.631 |
| p95 | 4492.606 | 2968.341 |

This short noisy measurement shows no observed material slowdown; it is not a
general performance improvement claim. Final posterior support is identical and
maximum absolute probability difference is 2.085468906662391e-7. Stable Float32
normalization changes rounding; bitwise equality is not claimed.

## Evidence and boundary

Focused/full logs and JUnit output are retained at
/work/checkpoints/cfc-fix-focused-final-20261015.txt,
/work/checkpoints/cfc-fix-full-20261015.txt and
/work/checkpoints/cfc-fix-full-20261015.xml. Runtime measurements are retained at
/work/checkpoints/cfc-runtime-{baseline,candidate_final}-20261015.json.
Machine-readable summary: convergence-filtering-config-fixes-20261015.json.

Oracle OFF. Reference holdout unchanged and not rescored. No new research
training, experiment selection or statistical acceptance claim; ordinary unit
tests include their bounded synthetic fitting fixtures. CUDA execution remains
unverified. The audit's latched stop_training observation and exact plateau
window semantics were clarified rather than changed.
