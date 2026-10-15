# V3 public optimization summary — 2026-10-15

Base: 22467b1642cd0bc3ac03b03b1100df07fdc8f415 (v3).

The optional host Qlib trainer needs observed update counts without reading
private optimizer histories. NHSMM.optimization_summary now exposes independent
successful-fit telemetry. Main counters increment after Adam.step, restart
selection preserves first-on-tie behavior, and refinement counts are published
only after successful refinement. A new attempt invalidates earlier telemetry;
partial fits never claim a completed summary. No parameter update, objective,
model semantic, configuration default or inference artifact key changes.

Focused validation: 49 passed in 11.62s, including observed early stopping,
defensive copies and invalidation on a failed subsequent fit.
Broad suite: 1217 passed, 3 CUDA-only skips, 5 explicit historical exclusions in 201.38s; no warnings..
The same five historical execution-identity exclusions remain explicit:
- tests/test_block_mass_smoothing_audit.py::test_completed_batch
- tests/test_block_mass_smoothing_audit.py::test_raw_bytes_tampering
- tests/test_block_mass_smoothing_audit.py::test_missing_replay_certificate
- tests/test_optimizer_bridge_verification.py::test_changed_source_manifest_rejected
- tests/test_optimizer_bridge_verification.py::test_rebound_initial_objective_rejected

Eight exact replays against the pre-extraction model also pass with zero
tolerance: parameters, Adam state, convergence/scheduler state, histories,
restart scores, best snapshots, likelihood and RNG. This new telemetry was not
present in that baseline and is tested separately, not claimed identical.
Baseline source SHA256:
10f4ca09722d06423ed1bd3361ec268316f9fdb39ccc0882a9f86e78e4c705da.
Candidate model SHA256:
26eb6ffc2fa7f96f97b693cfcced19a03c77726066fab6896260b246d911e2d5.
Candidate training SHA256:
5b85980fa9319c0f3882f859fa176241650845f45facdfdc1dca38040e35acb1.

AST/Ruff/Black/diff checks pass. The syntax gate caught a draft indentation error
before source application; the corrected source passed all subsequent checks.
Installed snapshot is refreshed; the actual optional Qlib trainer exercises this
public summary in its synthetic fit/recorder acceptance tests.

Logs:
- /work/checkpoints/core-summary-focused-20261015.txt
- /work/checkpoints/core-summary-full-20261015.txt and .xml
- /work/checkpoints/core-summary-parity-20261015.json
- /work/checkpoints/qlib-nhsmm-model-smoke-20261015.txt

Oracle OFF; old holdout/artifacts untouched. No controlled benchmark, GPU test,
new market-data experiment or improved holdout claim.
