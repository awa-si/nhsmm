# V3 batch 1 — shared causal frontier

Base: awa-si/nhsmm@v3, fceb4781a574fc3e14f9cb7a49cd614e97f35c69.
Status: implemented and verified.

## Change and preserved semantics

Batch forward and streaming/filter previously duplicated initialization and
state-age propagation. Both now call _initialize_causal_frontier and
_causal_filter_step in nhsmm/filtering.py. Each returns a normalized log posterior
and incremental log evidence. _normalize_causal_frontier owns finite-evidence
checking and stable log_softmax normalization.

The kernel accepts canonical normalized component scores. Batch forward keeps
its precomputed hazards and active-length masks, accumulates evidence separately,
and reconstructs the public unnormalized alpha output. Streaming computes its
hazards from the previous boundary scores and consumes the same kernel.
The public filter initializer still accepts scores and does not expose an
evidence API. Public arbitrary-score filter validation is preserved.

There is one implementation of continuation, episode renewal and D+ tail
combination. Age-independent transitions broadcast across the age axis without
changing the reduction order. Same-state transitions still reset episode age.
Initial age is one, boundary timing remains past-only, exact impossible support
is retained, and empty active batches return empty evidence/posterior tensors.

Noncausal completed-segmentation likelihoods and Viterbi recurrences are separate
mathematical contracts and are untouched. No public API, configuration default,
model parameter, artifact key or dependency is added. Model batch normalization
now uses the same stable normalization as streaming; numerical equality is
verified rather than assuming bitwise equivalence of every batch intermediate.

## Independent verification

tests/test_v3_causal_kernel.py adds 22 cases. Its reference enumerates distinct
continuation/renewal paths in probability space; it never calls the production
hazard/step helpers. Checks cover finite/D+ support, D=1 and D=3, age-dependent
and age-independent transitions, impossible duration mass, every prefix
posterior/evidence and gradients with respect to initial/duration/transition/
emission/tail parameters. Float64 tolerances are 2e-12 for posterior/evidence
and 3e-11 for gradients. Additional cases cover large Float32/Float64 offsets,
non-finite evidence and an empty active batch.

- Focused reference/model/training block: **474 passed**, 42.96 seconds.
- Installed snapshot: **22 passed**, 5.38 seconds; pip check clean.
- Broad applicable suite: **1211 passed, 3 CUDA skips, 5 explicit historical deselections**, 200.07 seconds.
- Existing host/Qlib integration: **54 passed, 1 skip**, 21.34 seconds; 22 existing deprecation warnings.
- AST parsing, Ruff, Black on new tests and git diff checks pass.

Five known historical execution-identity tests remain explicitly deselected:
test_block_mass_smoothing_audit::{test_completed_batch,test_raw_bytes_tampering,
test_missing_replay_certificate} and test_optimizer_bridge_verification::{
test_changed_source_manifest_rejected,test_rebound_initial_objective_rejected}.
Their test code and historical evidence are unchanged. This is not a claim that
the unrestricted historical suite is green.

## Host/Qlib integration

Host source: awa-si/nautilus@e527734bab719038ec3820014f7c9743e4cfca93,
using the updated editable NHSMM source. Existing shared Qlib, NHSMM, GBM bridge,
adapter, actor recovery and fixture tests pass, including real Qlib DatasetH,
LGBModel fit/predict and MLflow recorder interaction. One isolated no-GBM profile
case is skipped because this environment includes GBM; 22 existing NumPy/pandas
deprecation warnings are reported separately.

The initial host attempt used a read-only repository working directory and
failed when MLflow tried to write /workspace/mlruns. Its preserved failure log is
v3-causal-host-integration-20261015.txt. Repeating the same tests from
/work/artifacts/v3-causal-20261015/host passes without test or adapter changes.
This confirms the need for an explicitly writable recorder artifact location.

## Bounded runtime observation

Same host, one CPU thread, Torch2.14.1+cpu, K4/F5/D64, warmup32 and256 measured
steps repeated3 times. Runtime posterior probabilities are identical to the
pre-refactor synthetic baseline (max absolute delta0; identical support).

| Microseconds | Baseline | Candidate |
| --- | ---: | ---: |
| Mean | 2561.950 | 2098.327 |
| p50 | 1833.599 | 1753.884 |
| p95 | 3361.573 | 3342.160 |

This short noisy workload shows no observed material slowdown; it does not
establish a general speedup or batch-throughput guarantee.

## Checkpoint and limits

Evidence JSON: v3-causal-frontier-20261015.json.
Focused/full/host logs: /work/checkpoints/v3-causal-{focused,full,host-final}-20261015.txt.
Full JUnit: /work/checkpoints/v3-causal-full-20261015.xml.

Oracle OFF; old late test/artifacts unchanged and not rescored. No new research
training, model selection or statistical recovery acceptance. Unit tests include
bounded synthetic training fixtures. CUDA execution remains unverified.

Next batch: separate optimizer/training ownership, then implement the thin Qlib
training boundary in its owning host adapter.
