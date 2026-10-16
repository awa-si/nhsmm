# ML-stack ownership cleanup — 2026-10-16

Research orchestration is centralized in
[awa-si/nautilus/adapter/qlib](https://github.com/awa-si/nautilus/tree/main/adapter/qlib):
ChatGPT/RD-Agent bounded candidates, Qlib datasets/tasks, MLflow recording and
NautilusTrader execution validation. Model math, training primitives, artifacts
and inference contracts remain core-owned.

Retire 74 standalone experiment scripts, the synthetic ValidationConfig /
ValidationDataConfig / ValidationScenarioConfig surface, their dedicated tests,
manual synthetic acceptance CI jobs and historical report/protocol trees.
Mixed tests retain surviving model/configuration/optimizer behavior. Reusable
nhsmm.validation context evidence, alignment and snapshot APIs remain for the
host recorder. ModelConfig and ModelHealthThresholds retain their behavior.

The Oracle control dependency closure remains available, unchanged and OFF by
default, together with its support/gradient/recursion checks and the frozen
reference data. The source-checkout CPU benchmark/install helper remain.
Retained Oracle/reference controls are not the active research execution path.

Historical implementations and their exact evidence remain in
[the pre-cleanup tree](https://github.com/awa-si/nhsmm/tree/c0178857da54f7099d1cb39cb76a7c5a0a6680a8).
Restore that pinned source/software to reproduce old studies. The cleanup does
not retrain on market data, rescore a frozen holdout or qualify any model for
trading. Verification results will be recorded after the local gates.

Two immutable input reports and their short index remain solely for the retained
Oracle CLI. No new model work or holdout scoring is performed to restore them.
