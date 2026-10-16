# ML-stack extraction — 2026-10-16

Core packages own numerical training/inference, causal OOF construction,
calibration, schema/provenance admission and sealed artifact IO.
RD-Agent orchestrates candidates, Qlib owns tasks/datasets, MLflow records
runs. None is a model dependency.

| Previous namespace | Current owner |
| --- | --- |
| gbm.offline / evaluation / utility | adapter.qlib.gbm_research |
| gbm.ablation_profiles / head_ablation* / architecture_challenger | adapter.qlib.gbm_research |
| gbm.holdout_research / oof_shift / oos_provenance / research_dataset / synthetic_dataset | adapter.qlib.gbm_research |
| gbm.production_profiles | gbm.defaults: numeric defaults; host production_profiles: selection metadata |
| nhsmm.validation / diagnostics / ModelHealthThresholds | adapter.qlib.nhsmm_validation |
| NHSMM research scripts | adapter.qlib.nhsmm_research |
| NHSMM frozen data and input reports | tests/ml_stack/fixtures/nhsmm |

Imports intentionally change; model packages carry no compatibility wrappers.
The host's publish_registry_training_bundle_v4 and train/evaluate/publish helpers
orchestrate retained core artifact primitives. StagedModelTrainer stays core:
its dependency-ordered OOF chaining is part of the estimator.

Model equations, ModelConfig, numeric estimator defaults, causal/purge rules,
core artifact classes and bundle schemas are unchanged. Research dataclass
module paths change. Historical experimental pickles require their pinned
pre-extraction environment; no silent artifact conversion is introduced.
Core artifact security/admission and historical bundle readers remain.

The host ledger docs/ml-stack-extraction-20261016.json records source revisions
and per-file checksums. Frozen inputs move byte-for-byte. The previous test
migration manifest remains historical provenance. Oracle stays opt-in.
No market experiment, holdout rescoring or model promotion is part of this work.

## Verification

Coordinated host/test commit: [54c16d00c081](https://github.com/awa-si/nautilus/commit/54c16d00c0814a1d6512ce723ddbe04641905bd3).
Full combined suite: 2,135 passed + 177 subtests; 11 optional skips.
Final Backtest/ablation gate: 55 passed. Numerical defaults/ModelConfig,
frozen input hashes and independent mathematical assertions are preserved.
Offline wheel builds and isolated imports pass; no host dependency enters core.
Detailed provenance and validation are recorded in that host commit.
