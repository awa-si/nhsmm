# Models audit repairs — 2026-10-15

All seven findings from models-audit-20261015.md are repaired as one batch on
regime-v2. Baseline0e9d167eacf7102f93a96a419237468fa669ff88; model blob
47571aca9bcaea304464a925d96e9ad0bd911df5 becomes d21c928d6bdb1294b410420f68161b691740a49c.

## Behavior

| Finding | Repair |
| --- | --- |
| M01 | Empty entries contribute LL0; wholly empty training data is rejected before a restart or update. Feature width and finiteness are checked before training. |
| M02 | Refinement without required explicit context is rejected before weights change. |
| M03 | Input conversion, distribution initialization, emission initialization and restart construction use live parameter dtype/device. Internal Float64 encoder training is covered. |
| M04 | Continuous segment emission sums retain their density and gradients without categorical score clipping. |
| M05 | Causal initial/step normalization rejects nonfinite or zero evidence. Public likelihood raises on NaN rather than hiding it behind a finite sentinel. |
| M06 | All five Viterbi terminal-selection sites validate the terminal MAP score before backtracking. Impossible paths raise ValueError. |
| M07 | Empty forward returns [B,0,K,D], likelihood prediction returns a tensor, and Viterbi returns empty paths. Empty encoder calls bypass sequence encoding; invalid modes/timestep arguments remain rejected. |

Public empty-sequence likelihood intentionally changes from the old finite
negative sentinel to0 (probability1). Its existing prediction-contract test is
migrated explicitly; the change prevents neutral empty records from corrupting
sum objectives, restart ranking and stopping scores. Completely empty training
batches remain invalid.

Ordinary finite-support and geometric D+ semantics, same-state episode renewals,
causal-prefix versus noncausal-completed-segment objectives, artifact schema and
parameter/state keys are retained. The historical audit/reproducer remains
frozen evidence and refuses the changed model source; it is not a current CI
acceptance test.

## Verification

- New regressions on the old source:30 failed,2 passed, covering all seven
  findings. The final file has33 cases, including preservation of causal
  timestep validation on empty inputs.
- Final focused smoke plus prediction contracts:39 passed.
- First unrestricted suite:1139 passed,6 failed,3 CUDA skips. Five failures are
  the already known frozen-evidence execution/software identity issues. The
  sixth was the obsolete empty-sequence expectation, now migrated to LL0.
- Final applicable complete package run:1141 passed,3 CUDA skips,5 explicit
  historical deselections,195.13 seconds.
- Nautilus/NHSMM/Qlib/GBM integration:49 passed,1 skip requiring an isolated
  no-GBM profile,22 existing NumPy/pandas deprecation warnings,40.84 seconds.
  This preceded only the final empty-causal-timestep guard ordering hardening;
  the final core run covers that guard and all affected model logic.
- Frozen HSMM artifact925bc1d35721d031d5385e9fa52ffc381eb789e1abfe225d18468878f6b2130b:
  the old installed source and candidate produce bitwise-identical posterior
  probabilities on synthetic linspace(-2,2,320), shape[1,64,5]; max difference0.
  No holdout was rescored and no fitting updates were performed for this probe.
- AST, model Ruff F checks, full Ruff on changed tests, Black on the new
  regression file and git diff --check passed. Whole-repository style cleanliness
  is not claimed.
- A first patch-transfer invocation exceeded the container argv limit before
  execution; a compact transfer succeeded. No partial core edit resulted.

The five known failures are explicitly scoped, not fixed or suppressed in test
configuration:

- tests/test_block_mass_smoothing_audit.py::test_completed_batch
- tests/test_block_mass_smoothing_audit.py::test_raw_bytes_tampering
- tests/test_block_mass_smoothing_audit.py::test_missing_replay_certificate
- tests/test_optimizer_bridge_verification.py::test_changed_source_manifest_rejected
- tests/test_optimizer_bridge_verification.py::test_rebound_initial_objective_rejected

They failed with the same execution/software guard messages in the unrestricted
run. Their frozen evidence and guards are unchanged. The unrestricted suite is
therefore not claimed green.

Logs/JUnit: /work/checkpoints/models-fix-{red,smoke,final-smoke,full,final,integration}-20261015.*
Artifact probe: /work/checkpoints/models-fix-parity-20261015/result.json.
Machine summary: models-fixes-20261015.json.

No learning-budget experiment, GT fitting or test-based selection was run.
Verification tests perform their existing bounded synthetic training calls.
Oracle stays OFF. CPU verification only; CUDA remains untested. The repaired
core may proceed to a separately preregistered train/validation learning study.
