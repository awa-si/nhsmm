# V3 core slimming and Qlib ownership — 2026-10-15

Base: awa-si/nhsmm v3@48b49f284674dc9460027c9742ec82050d5894d2.
Host base: awa-si/nautilus main@dc27b8d00d5672409ec762e2f0e642d7ad0b0c2b.

## Change

The custom search engine, Choice/IntRange/FloatRange domains, ConfigTuner,
TuneEvaluation/TuneTrial/TuneReport and apply_config_overrides exports are removed.
There is no compatibility shim and no new Qlib/MLflow dependency in the core.
Package Python lines: 9266 -> 8912, a reduction of 354 (336 tuner + 18 exports).

Use ModelConfig.with_overrides and ValidationConfig.with_overrides for strict
validated copies. The synthetic validation CLI retains --set path semantics via
script-local parsing. Its maintained contract tests remain. The current source
closure list drops the deleted file; original recorded manifests/proofs are
unchanged, and numerical replay still requires its frozen source/software.

Integrated research uses ordinary Qlib model/dataset/record tasks and native
TrainerR in the existing Nautilus host. Explicit candidates use ordinary Python
enumeration; no replacement custom tuner, ranking abstraction or scheduler is
introduced. NHSMMRecord consumes core evaluate_validation_snapshot and persists
validation.pkl alongside existing health/manifest/model artifacts and metrics.

## Ownership

| NHSMM | Nautilus / Qlib |
| --- | --- |
| Model/health/validation configuration | Candidate lists and preregistered selection policy |
| Distributions, HSMM inference, causal runtime | Dataset provenance and chronological folds |
| Optimize and model-internal restarts | Native TrainerR task/recorder lifecycle |
| Artifacts, generic health/validation snapshots | MLflow recording and cross-model comparisons |
| Standalone synthetic mathematical acceptance | Integrated market-data research and execution |

ValidationConfig and synthetic math checks remain model-domain utilities.
The historical research scripts are not installed package modules and are not
replaced with Qlib for mathematical reference work. New integrated experiments
belong to the host. No probabilistic, training-objective or artifact-schema change.

## Verification

Complete core: 1285 passed, three CUDA-only skips, zero deselections
(237.07s), job 67778659631248b2b42acf2986c30215.
Focused core: 73 passed (16.55s), job f6956a97ad884e9db97d867495a9e1e1.
That read-only run emitted two pytest-cache write warnings; the full writable
gate above is clean. Installed boundary: eight passed with warnings as errors,
job 82770260ff914323bf26b85c4a554895; pip check clean.
Host: 85 passed, one no-GBM-profile skip, 117 warnings (71.75s),
job 46e7e34166464778bf7d317ba6efb0ec. These are existing timedelta
deprecations plus controlled trusted-artifact loads in the native recorder test.
No warnings or acceptance thresholds were suppressed/relaxed.
Syntax, Ruff, Black and diff checks pass. Core and host full gates ran in
parallel with two OMP/MKL threads each.

The native host test constructs two independent one-update fits, with distinct
seeds, fit directories and recorders. It verifies completion tags, loaded model
identity, actual updates and snapshot fingerprints/likelihood. A frame guard
rejects reserved test access by every model/record step. It uses synthetic
observations, not market data, and performs no automated winner selection.

The former tuner API tests and one Cartesian-consumption regression are retired
with their implementation; eight new configuration/research-boundary cases replace
them. The full suite has zero deselections. Original reference/math, causal,
historical-report and proof checks remain.

Installation preparation removed a stale installed tuning.py and synchronized
the root exports and script consumers. The first focused run identified the
stale file (74 passed, one failed); a synchronized retry passed. Installation
tests run from the installed checkout because their CLI-helper source is not a
distribution module. An initial collection attempt from the host checkout and a
missing ephemeral temp-file attempt are not counted as successful tests.

Source/install equality, actual jobs and test summaries are recorded in the
paired final JSON. Final documentation changes do not alter the tested code.

Oracle remains OFF; old holdout, model artifacts and original evidence are not
rescored, rewritten or removed. No new model-quality or speedup claim.
