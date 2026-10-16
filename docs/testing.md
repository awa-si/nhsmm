# Model verification

Canonical model, mathematical reference, distribution, training, runtime, artifact
and validation-snapshot tests have migrated to the [Nautilus ML stack](https://github.com/awa-si/nautilus/tree/c07bb12374bbfb3a8a5aec419112290ef5b60d80/tests/ml_stack/nhsmm).
There is no duplicate tests/ tree in this model package. Model implementation
and public validation APIs remain here; this core never imports the host.

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

Oracle stays OFF by default and opt-in. Its eight control helper scripts, frozen
data/reference/trading-regimes-v1 bytes and two input reports remain here. The
host tests resolve these fixtures from the imported editable source and restore
cwd after the compact CLI test. Restricted support requires --support-mode oracle.
No long Oracle experiment or frozen holdout scoring is introduced.

Local core static checks remain ruff check nhsmm scripts and
black --check nhsmm scripts. Smoke/release CI checks out the pinned external
test owner and runs the NHSMM subtree against the core commit being built.
A successful model suite verifies contracts, not trading performance.

The test-owner repository is private. GitHub jobs require the symbolic secret
ML_STACK_TESTS_READ_TOKEN with read-only access to awa-si/nautilus. The default
core-repository token cannot authorize that checkout. Jobs fail closed before
checkout if the secret is absent. Secret provisioning and hosted execution
are not verified by local migration checks. Never commit the token value.
