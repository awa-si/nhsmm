# Repository consistency audit — 2026-10-04

Scope: every tracked repository file on `develop`.

## Result

The repository was checked file-by-file for package-layer ownership, imports/exports, duplicate definitions, stale/deleted paths, documentation links, workflow alignment, validation-script overlap, and test/package gates.

Resolved in this audit:

- removed the internal `models/base.py -> nhsmm` package-root self import;
- replaced Artifact cross-module use of a private inference helper with the shared `validate_finite_state_tensors` helper;
- routed artifact serialization/loading and validation model fingerprinting through `ModelConfig.to_dict()/from_dict()`;
- updated README/public architecture for configuration, diagnostics, snapshots, and tuning;
- updated normative `docs/agent-domain.md` ownership/boundary directives;
- added the maintained learning/prediction usefulness gate to the package-validation workflow;
- extended release installed-package smoke to cover the public `ValidationConfig` / `ConfigTuner` surface;
- explicitly classified the Freqtrade-style dataframe loader in `nhsmm/data.py` as a legacy, non-root helper pending a breaking API window.

Static evidence:

- tracked files: **121**;
- package Python modules: **28/28 import successfully**;
- Python source AST parse: **68/68**;
- root `__all__`: no missing or duplicate exports;
- package import graph: no cycles;
- tracked bytecode/cache artifacts: none;
- local Markdown links: no broken relative links.

## Remaining intentional boundary

`nhsmm/data.py::load_dataframe(data_dir, pair, timeframe)` still encodes a Freqtrade-style `*-futures.feather` filename convention. It is not exported from package root and has no repository consumer. It was not removed or signature-changed in this audit to avoid an unannounced external breaking change. New framework-specific loaders in core are prohibited; migration/removal belongs to an explicit breaking API window.

## File inventory

| File | Classification | Audit result |
| --- | --- | --- |
| `.github/CODE_OF_CONDUCT.md` | GOVERNANCE | Repository governance/template. |
| `.github/CONTRIBUTING.md` | GOVERNANCE | Repository governance/template. |
| `.github/ISSUE_TEMPLATE/bug_report.yml` | GOVERNANCE | Repository governance/template. |
| `.github/PULL_REQUEST_TEMPLATE.md` | GOVERNANCE | Repository governance/template. |
| `.github/SECURITY.md` | GOVERNANCE | Repository governance/template. |
| `.github/workflows/package-validation.yml` | CURRENT WORKFLOW | Manual empirical package gates; now includes multicomponent and learning/usefulness jobs. |
| `.github/workflows/release.yml` | CURRENT WORKFLOW | Release/test/build/PyPI contract; installed-package smoke includes config/tuner public API. |
| `.github/workflows/smoke.yml` | CURRENT WORKFLOW | Manual package/runtime smoke. |
| `.gitignore` | REPO META | Repository metadata. |
| `AGENTS.md` | REPO META | Repository agent instructions. |
| `LICENSE` | REPO META | Repository metadata. |
| `README.md` | REPO META | Current public package overview; updated for config, diagnostics, snapshots, and tuner. |
| `docs/agent-domain.md` | CURRENT DOC | Current repository documentation/contract. |
| `docs/configuration.md` | CURRENT DOC | Current repository documentation/contract. |
| `docs/handoff.md` | CURRENT DOC | Current repository documentation/contract. |
| `docs/model.md` | CURRENT DOC | Current repository documentation/contract. |
| `docs/package-validation.md` | CURRENT DOC | Current repository documentation/contract. |
| `docs/release.md` | CURRENT DOC | Current repository documentation/contract. |
| `docs/state.md` | CURRENT DOC | Current repository documentation/contract. |
| `docs/testing.md` | CURRENT DOC | Current repository documentation/contract. |
| `docs/tuning.md` | CURRENT DOC | Current repository documentation/contract. |
| `docs/validation-api.md` | CURRENT DOC | Current repository documentation/contract. |
| `docs/validation/code-audit-validation-api-2026-09-28.md` | HISTORICAL EVIDENCE | Immutable empirical/audit record; historical counts are not current-suite claims. |
| `docs/validation/code-scan-2026-09-28.md` | HISTORICAL EVIDENCE | Immutable empirical/audit record; historical counts are not current-suite claims. |
| `docs/validation/general-context-detect-refine-confirmatory-2026-09-27.json` | HISTORICAL EVIDENCE | Immutable empirical/audit record; historical counts are not current-suite claims. |
| `docs/validation/general-context-detect-refine-package-core-confirmation-2026-09-27.json` | HISTORICAL EVIDENCE | Immutable empirical/audit record; historical counts are not current-suite claims. |
| `docs/validation/general-context-detect-refine-package-core-confirmation-2026-09-27.md` | HISTORICAL EVIDENCE | Immutable empirical/audit record; historical counts are not current-suite claims. |
| `docs/validation/general-context-detector-rapid-2026-09-27.md` | HISTORICAL EVIDENCE | Immutable empirical/audit record; historical counts are not current-suite claims. |
| `docs/validation/general-context-evidence-gating-2026-09-27.md` | HISTORICAL EVIDENCE | Immutable empirical/audit record; historical counts are not current-suite claims. |
| `docs/validation/general-context-package-diagnostics-2026-09-27.md` | HISTORICAL EVIDENCE | Immutable empirical/audit record; historical counts are not current-suite claims. |
| `docs/validation/general-context-robustness-2026-09-27.json` | HISTORICAL EVIDENCE | Immutable empirical/audit record; historical counts are not current-suite claims. |
| `docs/validation/general-context-robustness-2026-09-27.md` | HISTORICAL EVIDENCE | Immutable empirical/audit record; historical counts are not current-suite claims. |
| `docs/validation/general-context-shared-duration-ablation-2026-09-27.json` | HISTORICAL EVIDENCE | Immutable empirical/audit record; historical counts are not current-suite claims. |
| `docs/validation/general-context-shared-duration-ablation-2026-09-27.md` | HISTORICAL EVIDENCE | Immutable empirical/audit record; historical counts are not current-suite claims. |
| `docs/validation/general-context-shared-linear-effectfloor-confirm-2026-09-27.json` | HISTORICAL EVIDENCE | Immutable empirical/audit record; historical counts are not current-suite claims. |
| `docs/validation/general-context-shared-linear-effectfloor-confirm-2026-09-27.md` | HISTORICAL EVIDENCE | Immutable empirical/audit record; historical counts are not current-suite claims. |
| `docs/validation/general-context-split-fit-replication-confirm-2026-09-27.md` | HISTORICAL EVIDENCE | Immutable empirical/audit record; historical counts are not current-suite claims. |
| `docs/validation/general-context-split-replication-confirm-2026-09-27.json` | HISTORICAL EVIDENCE | Immutable empirical/audit record; historical counts are not current-suite claims. |
| `docs/validation/general-context-split-replication-confirm-2026-09-27.md` | HISTORICAL EVIDENCE | Immutable empirical/audit record; historical counts are not current-suite claims. |
| `docs/validation/learning-prediction-2026-10-04.md` | CURRENT EVIDENCE | Immutable empirical/audit record; historical counts are not current-suite claims. |
| `docs/validation/multicomponent-component-matched-package-15seed-2026-09-27.json` | HISTORICAL EVIDENCE | Immutable empirical/audit record; historical counts are not current-suite claims. |
| `docs/validation/multicomponent-component-matched-package-15seed-2026-09-27.md` | HISTORICAL EVIDENCE | Immutable empirical/audit record; historical counts are not current-suite claims. |
| `docs/validation/multicomponent-component-matched-package-core-2026-09-27.json` | HISTORICAL EVIDENCE | Immutable empirical/audit record; historical counts are not current-suite claims. |
| `docs/validation/multicomponent-component-matched-package-core-2026-09-27.md` | HISTORICAL EVIDENCE | Immutable empirical/audit record; historical counts are not current-suite claims. |
| `docs/validation/multicomponent-component-matched-package-screen-2026-09-27.md` | HISTORICAL EVIDENCE | Immutable empirical/audit record; historical counts are not current-suite claims. |
| `docs/validation/multicomponent-package-core-confirmation-2026-09-27.json` | HISTORICAL EVIDENCE | Immutable empirical/audit record; historical counts are not current-suite claims. |
| `docs/validation/multicomponent-package-core-confirmation-2026-09-27.md` | HISTORICAL EVIDENCE | Immutable empirical/audit record; historical counts are not current-suite claims. |
| `docs/validation/multicomponent-reference-deep-dive-2026-09-27.json` | HISTORICAL EVIDENCE | Immutable empirical/audit record; historical counts are not current-suite claims. |
| `docs/validation/multicomponent-reference-deep-dive-2026-09-27.md` | HISTORICAL EVIDENCE | Immutable empirical/audit record; historical counts are not current-suite claims. |
| `docs/validation/public-api-stress-2026-09-28.md` | HISTORICAL EVIDENCE | Immutable empirical/audit record; historical counts are not current-suite claims. |
| `nhsmm/__init__.py` | CURRENT PACKAGE | Root public export surface; __all__ resolved with no duplicates/missing exports. |
| `nhsmm/__version__.py` | CURRENT PACKAGE | Installed metadata version accessor. |
| `nhsmm/artifact.py` | CURRENT PACKAGE | Artifact v1 contract; now uses ModelConfig.to_dict/from_dict and shared finite-state validator. |
| `nhsmm/benchmark.py` | CURRENT PACKAGE | CPU incremental runtime benchmark contract. |
| `nhsmm/config.py` | CURRENT PACKAGE | Canonical ownership for model, health, and validation configuration contracts. |
| `nhsmm/context.py` | CURRENT PACKAGE | Sequence/context routing and context encoder support. |
| `nhsmm/convergence.py` | CURRENT PACKAGE | Training convergence tracking; depends only on config constants/logger. |
| `nhsmm/data.py` | CURRENT PACKAGE | Synthetic/optional data helpers. Legacy Freqtrade-style loader retained as non-root internal API; migration deferred to breaking window. |
| `nhsmm/diagnostics.py` | CURRENT PACKAGE | Model-health metrics using central ModelHealthThresholds. |
| `nhsmm/distributions/README.md` | CURRENT PACKAGE | Canonical distribution layer/documentation. |
| `nhsmm/distributions/__init__.py` | CURRENT PACKAGE | Canonical distribution layer/documentation. |
| `nhsmm/distributions/default.py` | CURRENT PACKAGE | Canonical distribution layer/documentation. |
| `nhsmm/encoder.py` | CURRENT PACKAGE | Canonical DefaultEncoder and streaming state. |
| `nhsmm/filtering.py` | CURRENT PACKAGE | Canonical filter state/trace and normalized causal filter recursion. |
| `nhsmm/inference.py` | CURRENT PACKAGE | Inference preparation/loading and shared finite-state validation. |
| `nhsmm/models/__init__.py` | CURRENT PACKAGE | Canonical model package. |
| `nhsmm/models/base.py` | CURRENT PACKAGE | Canonical NHSMM/DistributionSet; direct module imports, no package-root self import. |
| `nhsmm/runtime.py` | CURRENT PACKAGE | Incremental causal runtime and scoring/forecast integration. |
| `nhsmm/survival.py` | CURRENT PACKAGE | Active-episode survival forecasts. |
| `nhsmm/transitions.py` | CURRENT PACKAGE | One-step transition forecasts. |
| `nhsmm/tuning.py` | CURRENT PACKAGE | Public validated configuration tuner. |
| `nhsmm/validation/__init__.py` | CURRENT PACKAGE | Validation implementation; importable and covered by validation tests. |
| `nhsmm/validation/alignment.py` | CURRENT PACKAGE | Validation implementation; importable and covered by validation tests. |
| `nhsmm/validation/api.py` | CURRENT PACKAGE | Validation implementation; importable and covered by validation tests. |
| `nhsmm/validation/context_evidence.py` | CURRENT PACKAGE | Validation implementation; importable and covered by validation tests. |
| `nhsmm/validation/nhsmm.py` | CURRENT PACKAGE | Validation implementation; importable and covered by validation tests. |
| `nhsmm/validation/snapshot.py` | CURRENT PACKAGE | Reproducible model/data/context validation snapshots using config contract serialization. |
| `nhsmm/validation/split_fit.py` | CURRENT PACKAGE | Validation implementation; importable and covered by validation tests. |
| `nhsmm/validation/state_effect.py` | CURRENT PACKAGE | Validation implementation; importable and covered by validation tests. |
| `pyproject.toml` | REPO META | Current package/build/dependency contract. |
| `scripts/benchmark_runtime.py` | CURRENT SCRIPT | Artifact-based production CPU runtime benchmark entry point. |
| `scripts/install_cpu_dev.py` | CURRENT SCRIPT | CPU-only development bootstrap. |
| `scripts/validate_duration_context.py` | CURRENT SCRIPT | Duration-context mechanism validation. |
| `scripts/validate_learning_prediction.py` | CURRENT SCRIPT | Maintained tuneable learning + structural usefulness acceptance harness. |
| `scripts/validate_multicomponent_identifiability.py` | CURRENT SCRIPT | Multicomponent identifiability gate used by package-validation workflow. |
| `scripts/validate_state_recovery.py` | CURRENT SCRIPT | Independent state-recovery mechanism evidence; retained because scope differs from usefulness harness. |
| `scripts/validate_transition_context.py` | CURRENT SCRIPT | Transition-context mechanism validation. |
| `tests/test_artifact.py` | CURRENT TEST | Regression/contract coverage; collected by full pytest suite. |
| `tests/test_benchmark.py` | CURRENT TEST | Regression/contract coverage; collected by full pytest suite. |
| `tests/test_categorical_rsample.py` | CURRENT TEST | Regression/contract coverage; collected by full pytest suite. |
| `tests/test_causal_dynamic_hazard.py` | CURRENT TEST | Regression/contract coverage; collected by full pytest suite. |
| `tests/test_config_contract.py` | CURRENT TEST | Regression/contract coverage; collected by full pytest suite. |
| `tests/test_context_evidence.py` | CURRENT TEST | Regression/contract coverage; collected by full pytest suite. |
| `tests/test_context_hardening.py` | CURRENT TEST | Regression/contract coverage; collected by full pytest suite. |
| `tests/test_convergence.py` | CURRENT TEST | Regression/contract coverage; collected by full pytest suite. |
| `tests/test_default_distributions.py` | CURRENT TEST | Regression/contract coverage; collected by full pytest suite. |
| `tests/test_diagnostics.py` | CURRENT TEST | Regression/contract coverage; collected by full pytest suite. |
| `tests/test_duration_indexing.py` | CURRENT TEST | Regression/contract coverage; collected by full pytest suite. |
| `tests/test_encoder.py` | CURRENT TEST | Regression/contract coverage; collected by full pytest suite. |
| `tests/test_filtering.py` | CURRENT TEST | Regression/contract coverage; collected by full pytest suite. |
| `tests/test_general.py` | CURRENT TEST | Regression/contract coverage; collected by full pytest suite. |
| `tests/test_incremental_runtime.py` | CURRENT TEST | Regression/contract coverage; collected by full pytest suite. |
| `tests/test_inference.py` | CURRENT TEST | Regression/contract coverage; collected by full pytest suite. |
| `tests/test_inference_context_fastpath.py` | CURRENT TEST | Regression/contract coverage; collected by full pytest suite. |
| `tests/test_learning_prediction_validation.py` | CURRENT TEST | Regression/contract coverage; collected by full pytest suite. |
| `tests/test_model_filter.py` | CURRENT TEST | Regression/contract coverage; collected by full pytest suite. |
| `tests/test_public_runtime_api.py` | CURRENT TEST | Regression/contract coverage; collected by full pytest suite. |
| `tests/test_runtime.py` | CURRENT TEST | Regression/contract coverage; collected by full pytest suite. |
| `tests/test_runtime_scoring.py` | CURRENT TEST | Regression/contract coverage; collected by full pytest suite. |
| `tests/test_runtime_transition.py` | CURRENT TEST | Regression/contract coverage; collected by full pytest suite. |
| `tests/test_scan_bugfixes.py` | CURRENT TEST | Regression/contract coverage; collected by full pytest suite. |
| `tests/test_state_context_evidence.py` | CURRENT TEST | Regression/contract coverage; collected by full pytest suite. |
| `tests/test_survival.py` | CURRENT TEST | Regression/contract coverage; collected by full pytest suite. |
| `tests/test_training.py` | CURRENT TEST | Regression/contract coverage; collected by full pytest suite. |
| `tests/test_transitions.py` | CURRENT TEST | Regression/contract coverage; collected by full pytest suite. |
| `tests/test_tuning.py` | CURRENT TEST | Regression/contract coverage; collected by full pytest suite. |
| `tests/test_validation_hardening.py` | CURRENT TEST | Regression/contract coverage; collected by full pytest suite. |
| `tests/test_validation_nhsmm_api.py` | CURRENT TEST | Regression/contract coverage; collected by full pytest suite. |
| `tests/test_validation_public_api.py` | CURRENT TEST | Regression/contract coverage; collected by full pytest suite. |
| `tests/test_validation_snapshot.py` | CURRENT TEST | Regression/contract coverage; collected by full pytest suite. |
| `workspace.ini` | REPO META | Current AWA development workspace resources. |
