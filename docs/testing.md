# Model verification

Canonical model, mathematical reference, distribution, training, runtime, artifact
and validation-snapshot tests have migrated to the [Nautilus ML stack](https://github.com/awa-si/nautilus/tree/main/tests/ml_stack/nhsmm).
There is no duplicate tests/ tree in this model package. Model implementation
remains here; this core never imports the host.

Install this source checkout editable in the research environment, then run from
Nautilus:

```bash
python -m pytest -q tests/ml_stack/nhsmm
python -m pytest -q tests/ml_stack/nhsmm/reference
```

The independent probability-space HSMM reference moved unchanged. Assertions,
numerical tolerances, gradient/likelihood/filtering/Viterbi/finite-D+ checks, and
all surviving core behavior tests are retained. Run the full model suite for
changes to shared inference, distributions, context, configuration or artifacts.
Do not hide failures with deselections or changed tolerances; report CUDA skips.

Oracle/benchmark helpers now live in Nautilus adapter.qlib.nhsmm_research.
Frozen fixtures/input reports live in tests/ml_stack/fixtures/nhsmm. Tests no
longer require research files in this core checkout. Run CLIs from Nautilus
with python -m and the new module name. Oracle remains OFF and opt-in.

Local core static checks remain ruff check nhsmm scripts and
black --check nhsmm scripts. Smoke/release CI checks out the pinned external
test owner and runs the NHSMM subtree against the core commit being built.
A successful model suite verifies contracts, not trading performance.

The test-owner repository is private. GitHub jobs require the symbolic secret
ML_STACK_TESTS_READ_TOKEN with read-only access to awa-si/nautilus. The default
core-repository token cannot authorize that checkout. Jobs fail closed before
checkout if the secret is absent. Secret provisioning and hosted execution
are not verified by local migration checks. Never commit the token value.

The canonical test_forecast_precision.py module checks rare finite/D+ end and state-change risks against 80-digit Decimal references in Float32/Float64, hard-support forecast gradients against independent probability formulas, geometric first/second derivatives, result identities, and separate duration/transition target learning.


## Central tunable verification

The canonical test_central_tunables.py checks invalid/valid overrides, actual probability laws, context/encoder propagation, Adam/scheduler/patience settings, supervised precedence and current/legacy artifact predictions. Host test_nhsmm_config_tuning.py checks the bounded search policy, grid, split separation, request guards and native Qlib/MLflow recording.

Alternate optimization-config regressions verify the actual annealing temperatures,
main/refinement Adam and clipping settings, and rejection of ignored construction
or supervised-only overrides before any parameter update.

2026-10-21 optimizer-config verification: 125 focused tests passed. The full
model plus tuning/auditor run passed 1474 tests with three CUDA skips; one
metadata setup error was resolved by exporting the original pyproject.toml.
That unchanged test passed separately (1475 distinct passes total). No model
code changed between those runs; Ruff and Black passed for the complete core.

Temperature-boundary verification (2026-10-21): test_temperature_numerics.py
adds 24 cases, including 100-digit Decimal schedule references, bit-exact
ordinary arithmetic, actual extreme annealed/constant training temperatures,
and rejection of an underflowing positive ratio/target combination before
model mutation. The complete model plus tuning/auditor suite passed 1499 tests
with three CUDA skips. Initial focused testing exposed an invalid zero-ratio
test setup; it was corrected to a valid underflowing configuration, without
changing configuration bounds. Corrected focused run: 39 passed.
