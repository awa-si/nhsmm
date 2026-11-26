# Synthetic NHSMM tests owned by model repository

Owner: awa-si/nhsmm. Source migration: Nautilus `66f6f17b78ccd547f05aba5b7afb6db0da67a846`, 2026-11-26.

## Ownership contract

- NHSMM owns deterministic, synthetic model and mathematical-reference tests
  for its distributions, likelihood, filtering, runtime, training, artifacts,
  model modes, configuration and numerical invariants.
- Nautilus owns market and data-dependent tests, regime-label calibration,
  Qlib/RD-Agent integration, adapters and validation-snapshot logic.
- NHSMM core code and tests do not import Nautilus, Nautilus data fixtures,
  model-specific market labels or trading platform APIs.

Exactly **90 synthetic model-core test files** plus **11 support Python files**
were copied into `tests/ml_stack/nhsmm/` with original byte content and
historical internal cross-test import namespace preserved. Two package
`__init__.py` files were added in `tests/` and `tests/ml_stack/`.

The 9 original core-tree files importing `adapter.qlib.nhsmm_validation`
and the Nautilus-only `test_parallel_test_groups.py` were **not** moved.
They remain Nautilus-owned. Nautilus retains an additional 34 NHSMM-oriented
host-level tests, including market/research/adapter tests. No trading-data
fixtures were migrated.

## Mandatory workspace tests

The model-local `scripts/nhsmm_test_groups.py` selects **all** synthetic
test files and assigns each once to one of five required groups:

```bash
python scripts/nhsmm_test_groups.py --verify
group=reference
mapfile -t paths < <(python scripts/nhsmm_test_groups.py --group "$group")
python -B -m pytest -q -p no:cacheprovider "${paths[@]}"
```

Launch all five groups as independent containers in the NHSMM workspace
(maximum five per AWA `exec_group`) and require every exit code zero.
No GitHub Actions are used or modified.

## 2026-11-26 source migration qualification

AWA workspace `f606e654ae5148d2a83f10c1afc65f8c`, slot 3,
source revision `9268dad905a36f00e4f3ab63ecb2ccc2649f2967`
plus the new uncommitted copied tests, verified before publishing.
`exec_group` `34013fef88ca474e8cfedbdb6837e01c` completed all five:
- filtering-a: 71 passed
- filtering-b: 268 passed
- reference: 408 passed, 3 skipped
- training: 262 passed
- support: 573 passed, 3 skipped

**Total: 1,582 passed, 6 skipped, no failures**, 1,588 tests collected.
Skipped tests are the optional native-Rust research reference and CUDA
cases; the CPU model-core suite is not a CUDA qualification.

## Transitional compatibility

Nautilus currently retains its historical copy of the 90 synthetic
model-core test files. This is a **temporary compatibility mirror**, not
the authoritative owner. Its existing unmodified GitHub Actions workflow
still references `tests/ml_stack/nhsmm`; deleting the files without a
separately authorized redirect would silently weaken that workflow.
The requested workspace-only scope does not authorize changing or invoking
GitHub Actions. Final deduplication must preserve that gate by an
explicitly authorized separate change. New model-core tests should
be introduced only in NHSMM; new host integration tests only in Nautilus.
