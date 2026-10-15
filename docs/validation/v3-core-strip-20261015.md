# NHSMM core strip — 2026-10-15

Base: awa-si/nhsmm v3 2d521c362dc18b16ccceffa4388f1f69a592ed74.
Host verification: awa-si/nautilus main 9bdc25310ba99e0a81f3ada61af5fe063dc8c1ba.

## Removed and retained

- Remove nhsmm.data and its 13 toy generator/Dataset-only tests: no maintained
  model, script or host imports it. Three prospective source inventories stop
  listing deleted data/benchmark files that were not numerical dependencies.
- Move the existing CPU benchmark into scripts/benchmark_runtime.py. Its four
  class/function definitions are AST-identical; the existing regression test
  follows the CLI helper. No benchmark module or root export remains in wheels.
- Remove Convergence callback protocol/list/lock/execution, JSON exporter,
  any_converged and the unused early_stop/stop_training bookkeeping. The actual
  training-loop convergence_stop decision, delta/plateau predicates, LR-aware
  patience, scheduler, objective precision and optimization_summary remain.
- Remove unused ContextRouter get_* / select_features methods, write-only
  dimension cache and SequenceSet.total_timesteps. Tensor fields are canonical.
- Preserve fractional score precision coverage; retire only the two JSON-export
  cases. Add two package-boundary checks for absent retired modules.

Installed package Python: **8927 → 8448 lines (479 fewer, 5.4%)**. Benchmark code
is retained outside the package; the unused toy data implementation is deleted.
No compatibility aliases recreate these retired APIs. See docs/api.md.

## Exact baseline comparison

A separately executed baseline replay uses the unchanged old convergence and
training source from the base commit. Across 96 monitor updates (delta/plateau,
with/without scheduler, two restarts), every convergence flag, effective
patience, score/delta/best-score tensor and scheduler score matches exactly.
Eight complete paired fits cover both modes, scheduler on/off and early stopping
on/off. All parameter/state tensors, retained parameter gradients, likelihoods
and actual update/restart summaries match with rtol=0 and atol=0.
Script/result: /work/checkpoints/nhsmm-strip-parity-20261015.{py,json}.

## Gates

- Focused boundary/training/context/benchmark/runner gate: 136 passed,
  3 CUDA-only skips, 26.18 seconds.
- Full core: **1302 passed, 3 CUDA-only skips, zero warnings**, 240.55 seconds.
  Command: python -m pytest -q tests --tb=short.
  Previous 1315 - 13 toy-data cases - 2 export cases + 2 boundary cases = 1302.
- Unchanged host Qlib/GBM/NHSMM/actor/provider integration: **169 passed,
  5 subtests passed, 1 existing GBM-profile skip**, 170 existing timedelta/
  trusted-pickle warnings, 160.38 seconds. Full Nautilus execution is not claimed.
- Verification wheel built successfully:
  nhsmm-0.1.dev734-py3-none-any.whl, 93048 bytes,
  SHA256 fe7aa22c29abcd1778722ce30be2e9084dae8eaba4421dfa0ff4cabb0c1038aa.
  Built from the tested worktree before commit; not a published release.
- Installed-wheel smoke from a separate target outside the checkout passed:
  retired modules/root export absent, no Qlib/MLflow/Nautilus imports, a real
  one-update fit, artifact reload with exact likelihood parity and normalized
  causal streaming posterior. The wheel was not published.

Logs: /work/checkpoints/nhsmm-strip-{core,host}-full-20261015.log.
Wheel checks: /work/checkpoints/nhsmm-strip-wheel-verification-20261015.json.

Ruff, syntax, pip check and diff checks passed. The full core command uses no
deselection list. The 15 retired tests cover only removed toy-data/export APIs;
all model/distribution/reference/causality tests remain. Runtime/benchmark
semantics are retained and active source manifests no longer reference deleted
paths. Archive reports, proof bytes, protocol JSON, source evidence index and
the frozen late holdout remain unchanged. Historical numerical replay still
requires the matching pinned source/software; an archive audit is separate.

No new empirical regime-quality, OOS, cost or trading qualification is claimed.
Oracle remains off. NumPy remains required by generic validation; Qlib and
MLflow remain host dependencies. Full Nautilus strategy/catalog/execution
suites are outside this affected-adapter gate.
