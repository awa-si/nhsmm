# NHSMM documentation map

NHSMM owns probabilistic **HMM, HSMM and neural-NHSMM** model semantics, training, runtime inference, likelihood backends and **synthetic model tests**. Nautilus owns market data, Qlib validation, trading/regime labels, replay and integration tests.

| Task | Canonical documentation |
| --- | --- |
| Model, runtime, Rust backend and forecast semantics | [Model contract](model.md) |
| Model lifecycle and public API | [API](api.md), [Configuration](configuration.md), [Modes](model-modes.md) |
| Mathematical verification and duration | [HSMM mathematics](hsmm-mathematical-contract.md), [Duration tail](duration-tail-contract.md) |
| Synthetic tests and five-group AWA workspace execution | [Synthetic test ownership](synthetic-test-ownership.md), [Verification and evidence](testing.md) |
| Operating limits | [Production qualification](production.md) |
| Historic development and measurements | [Migration history](v3.md), dated reports in [Verification](testing.md) |
| Package publication | [Release procedure](release.md) |

## Current ownership and interfaces

- `tests/ml_stack/nhsmm/` is the **canonical independent model-core suite** (90 modules migrated on 2026-11-26). Its five mandatory groups are selected by `scripts/nhsmm_test_groups.py`.
- Nautilus owns Qlib validation, training-label-based regime calibration, market research and host integration tests. Its existing model-core copy is a **temporary compatibility mirror**; do not add model tests to that mirror.
- `HSMMFilterRuntime.forecast()` requires a five-regime calibration matrix. Its `duration` estimates **latent-episode** survival, not market-regime persistence. See the [forecast contract](model.md#calibrated-next-step-market-regime-forecast).
- `set_backend("auto")` uses the compatible packaged Rust likelihood path when available, otherwise Python. This is not a blanket speed, hardware or trading-performance guarantee.
- NHSMM mathematical correctness does not qualify trading deployment; empirical model-admission evidence belongs to Nautilus.

## Workspace verification (no GitHub Actions)

From the NHSMM checkout, verify ownership then run **all five groups concurrently** in separate AWA containers (maximum five per `exec_group`), requiring every exit code zero.

```bash
python scripts/nhsmm_test_groups.py --verify
group=reference
mapfile -t paths < <(python scripts/nhsmm_test_groups.py --group "$group")
python -B -m pytest -q -p no:cacheprovider "${paths[@]}"
```

Historical results are qualified only for the exact revisions recorded in their receipts. GitHub Actions are excluded unless separately requested.
