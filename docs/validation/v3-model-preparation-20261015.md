# V3 model preparation audit — 2026-10-15

Base: `f807a00cef0bd26754007ea46a7a6c4b0831ad6e` (`awa-si/nhsmm@v3`).

## Finding and change

The public `log_likelihood` called `_ensure_tensor` to obtain batch size,
then `_build_sequence_set` repeated normalization, finite-input validation,
ragged padding and mask allocation. The first prepared tensor/mask were discarded.

Batch size now comes from the validated SequenceSet. Observation/context
preparation retains one owner; forward recursion and terminal reduction are
unchanged. This removes an entire redundant preparation pass per likelihood call,
without introducing a prepared-input API or another module.

The audit also inspected model initialization, encoder/context preparation,
causal and noncausal forward/Viterbi dispatch, and training helpers against
`nhsmm/training.py` and `nhsmm/inference.py`. Model-specific initialization,
loss and snapshot helpers remain with the model. External candidate orchestration
remains in Nautilus/Qlib. No mathematical defect or need for a new module is
established by this refactoring.

Decoding still has its separate entry preparation and empty-input short circuit;
this batch does not alter that behavior. A future consolidation must preserve
the distinction between unpadded lengths and padded tensors, as well as empty
decode validation ordering.

## Verification

The 22 new cases cover causal/noncausal, finite/tail durations, dense/ragged
batches, reduced/per-sequence scores, and empty batch/time inputs. Terminal
evidence is reduced independently from forward tensors. Likelihoods and all
trainable-parameter, observation and context gradients match exactly on the
16 nonempty cases. A call-count guard detects redundant preparation.

The initial new test harness incorrectly requested gradients for frozen
parameters and failed before comparison; filtering by requires_grad corrected
the harness. Production code did not change in response.

No timing benchmark or empirical model-quality improvement is claimed.
Oracle remains OFF; frozen holdout and historical evidence are unchanged.

Observed gates (Python 3.14.7 / CPU Torch shared environment):

| Gate | Result |
| --- | --- |
| Full core `python -m pytest -q --tb=short` | 1307 passed, 3 CUDA skips, no deselections; 201.46 s |
| Preparation + existing config/context regressions | 37 passed; 5.57 s |
| Installed package, new tests, `-W error` | 22 passed; 6.45 s |
| Host `tests/test_nhsmm*.py tests/test_qlib*.py tests/test_gbm_qlib_bridge.py` | 85 passed, 1 profile skip, 117 warnings; 59.23 s |
| Ruff, new-test Black, pip check, diff check | Passed |

Host warnings are NumPy/pandas timedelta deprecations and controlled trusted-pickle
test loads, matching the preceding host run. Host source is unchanged at
`f7df0c068613a48ab3aa4f7ca5d8ab05ce35dab1`.
The package delta is one line fewer; the measured improvement is preparation
call count (two to one), not end-to-end execution speed.

AWA execution jobs: core `6d19bf713e424dd89a48cb35cd4ab798`,
host `23bef2bac00745d784cf7d6d1571eff8`,
focused `1edafe5071bf406297eb721c249f525b`,
installed `cb4c27a4a6364aa4843083db518499fa`.
