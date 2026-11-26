# Canonical synthetic NHSMM model tests

This is the NHSMM-owned mathematical and synthetic model suite. It does not import Nautilus market data, Qlib or adapter modules.

The test tree deliberately retains the `tests.ml_stack.nhsmm` Python import namespace used by independent mathematical-reference helpers and cross-test fixtures.

Run all tests locally:

```bash
python -B -m pytest -q -p no:cacheprovider tests/ml_stack/nhsmm
```

For mandatory AWA workspace verification, run `python scripts/nhsmm_test_groups.py --verify`; then launch all five disjoint test groups (`filtering-a`, `filtering-b`, `reference`, `training`, `support`) as independent parallel jobs, no more than five containers per `exec_group`. Require all five exit codes zero and report skips separately.

The migration qualification on 2026-11-26 collected 1,588 tests: 1,582 passed, 6 optional/CUDA skips, zero failures. Subsequent core changes require fresh source-bound runs.

New **model-internal synthetic** tests belong here. Market labels, Qlib validation, trading context, replay, regime-calibration and host integration tests belong in Nautilus. The Nautilus model-test copy is retained only as a temporary compatibility mirror.

See [test ownership and limits](../../../docs/synthetic-test-ownership.md).
