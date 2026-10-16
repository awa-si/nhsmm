# Testing and verification

The model package owns mathematical and runtime correctness. Research datasets,
training tasks, experiment comparison and native execution acceptance belong to
[the Nautilus ML stack](https://github.com/awa-si/nautilus/tree/main/adapter/qlib).
Core tests remain necessary even when Qlib owns experiment execution.

## Core checks

```bash
python -m pytest -q
ruff check nhsmm tests scripts
black --check nhsmm tests scripts
```

Run syntax/static and focused checks before longer work. Run the complete core
suite for changes to shared inference, distributions, context, configuration,
training or artifacts. Do not use deselections or loosen numerical tolerances
to conceal regressions. CUDA-only checks may skip on CPU hosts; report them.
The manually dispatched smoke workflow checks import, core tests and the source
checkout runtime benchmark. Retired synthetic acceptance jobs are removed.

`tests/reference/` retains its independent probability-space HSMM implementation
and checks likelihood, Viterbi, filtering, gradients, structural invariants,
finite/D+ duration boundaries and numerical precision against production code.
Ordinary core tests cover causal prefixes, padding/ragged batches, context
contracts, initialization/restarts, optimizer/convergence, fit summaries,
artifacts, forecasts and model-health/context-evidence snapshots.

Tests for retired scripts and their scenario configuration are removed with the
implementation. Mixed test files retain every test of a surviving core behavior;
for example, tail-boundary optimizer recovery and Gaussian initialization remain.

## Retained Oracle controls

Oracle stays OFF and opt-in. To honor the retained control contract, the eight
source-checkout scripts in the dependency closure of
`compare_compact_duration_em.py` and `diagnose_trading_oracle_structure.py` remain.
They are frozen control/reference tooling, not the active experiment stack and
are not packaged as `nhsmm` modules. Full duration support remains the default;
restricted true support requires explicit `--support-mode oracle`. Existing support,
output-overwrite, recursion and gradient tests for these controls remain.
Do not run them automatically or tune against their outer/frozen holdout.
The frozen `data/reference/trading-regimes-v1` bytes remain unchanged.

## Host acceptance

From a Nautilus checkout with both editable model cores and optional research
dependencies installed, run its NHSMM/Qlib and native execution tests. The host
owns real DatasetH/TrainerR/MLflow and Nautilus lifecycle tests. Successful core or
synthetic host checks do not establish market usefulness or live readiness.

Historical experiment scripts, protocols, proofs and long execution logs removed
from this branch remain available at
[NHSMM before cleanup](https://github.com/awa-si/nhsmm/tree/c0178857da54f7099d1cb39cb76a7c5a0a6680a8).
Reproducing those studies requires that pinned source and matching software;
they are not current test instructions. Current cleanup evidence is recorded in
[ml-stack-cleanup.md](ml-stack-cleanup.md).
