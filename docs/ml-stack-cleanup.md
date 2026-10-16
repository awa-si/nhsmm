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
trading. Verification results follow below.

Two immutable input reports and their short index remain solely for the retained
Oracle CLI. No new model work or holdout scoring is performed to restore them.

## Verified source and execution

| Repository | Tested code revision | Full suite |
| --- | --- | --- |
| nhsmm (v3) | [44d606809678](https://github.com/awa-si/nhsmm/commit/44d6068096787ff64e66fb104ba2b7490e083561) | 1,029 passed; 3 skipped (CUDA unavailable); 108.08 s. |
| gbm (main) | [ac5852da52d7](https://github.com/awa-si/gbm/commit/ac5852da52d799682dd84aad834c83bfa51f3148) | 181 passed + 48 subtests; no skips; 16.24 s. |
| nautilus (main) | [95ba29102249](https://github.com/awa-si/nautilus/commit/95ba2910224939a1dd79c005eead6965988b3ee2) | 916 passed + 129 subtests; 14 optional skips; 225 warnings; 201.92 s. |

All three runs used Python 3.14.7, Torch 2.14.1+cpu, vendored
Qlib 0.9.9.dev0, MLflow 3.15.2 and RD-Agent 1.0.0, with bounded CPU threads.
Commands: `python -m pytest -q --tb=short -ra tests` in each checkout;
`python -m pip check` passed after every suite and after packaging setup.
The host suite exercised actual fitted model admission, causal artifact replay,
Qlib/MLflow recorders, and RD-Agent model/tracking/review/resume integration.

Nautilus skips: seven optional TA-Lib comparisons, one isolated no-GBM profile,
and six unbuilt Rust bridge checks. Its warnings concern retained Bundle-v3
compatibility, dependency deprecations and explicit trusted fixture reloads.
No test failure was suppressed. The previously missing Oracle input report
was restored byte-for-byte with its paired report; the full NHSMM suite,
including the retained compact Oracle CLI regression, then passed.

Both core wheels built offline with `pip wheel --no-deps --no-build-isolation
--no-index` from clean copies. NHSMM's declared wheel build dependency was
installed first. Archive inspection retained the NHSMM validation API and
GBM dataset admission contract, excluded retired modules, and confirmed no
host ML-stack runtime dependencies were introduced into either core.
NHSMM used version override 0.0.0 only for this packaging check.

| Wheel | SHA-256 |
| --- | --- |
| gbm-0.1.0-py3-none-any.whl | ded2949c1b7e1c2de9611cb185aab4fd424e8ac9bbb9a4ec25d0c553b31477b3 |
| nhsmm-0.0.0-py3-none-any.whl | 550dbaa20b3e1c4d16f8a76eb1694d7ae2b6d72c1fca5f740abd303cbc95fd09 |

Changed-source syntax/format/lint and dangling-import gates passed. NHSMM's
ModelConfig, ModelHealthThresholds and strict dataclass parser, plus the moved
ResearchProtocol class, were AST-identical to their pre-cleanup definitions.
Remote code trees matched the staged local trees before checkpoint publication.
This evidence verifies the cleanup and integration contracts; it does not
establish trading performance or authorize a model promotion.
