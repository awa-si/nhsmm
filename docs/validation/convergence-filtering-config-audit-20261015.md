# Convergence, filtering and configuration audit — 2026-10-15

Status: audit complete; findings open. Production code unchanged.

## Source and scope

Repository: awa-si/nhsmm, branch regime-v2.
Audited remote commit: da657e6385c1d83f318ebc2fed231416dbd184e4.
Remote and executed workspace tree: 8005895d3058c6be5f0cda222e8236d4aa8b3fd9.
The local offline mirror has commit b563b09a6e51196ec081a75d544364d9dbe306fc;
commit metadata differs, but the complete Git tree and target blobs match.

| Module | Verified Git blob |
| --- | --- |
| nhsmm/convergence.py | ac5498542eb0fb7a0fc6c37be22d941f1f96690d |
| nhsmm/filtering.py | 3e59661d3b554ad245f5f71febb7fb408c616ed9 |
| nhsmm/config.py | cf3d165bf432611b79505ad47aa8f3de00dd30bc |

Read all three modules and material model construction/training, runtime and
validation consumers. Reviewed the applicable AGENTS.md, domain, model and test
contracts. No fitting, reference-holdout scoring or experiment selection occurred.
Oracle remains OFF.

## Confirmed findings

### F01 — P1: impossible evidence produces a NaN filter state

Owner: nhsmm/filtering.py, _filter_step_normalized, posterior normalization
around lines 299–304; public filter_step delegates at lines 357–363.

Initialize K=D=B=1 with zero initial/emission scores; then call filter_step with
emission=-inf, duration=0 and transition=0. The public API permits negative-infinite
emission scores, but the computed normalizer is -inf. Subtracting it yields NaN.
The unchecked state constructor bypasses the public state validator.

Expected: explicitly reject zero/non-finite evidence before division, as
initialize_filter already does. Cover mixed batches and disjoint supported states,
not only a completely -inf emission row. Share the guard with internal consumers
without weakening exact impossible-support semantics.

This is demonstrated for the public score API. The tested built-in Gaussian
runtime with a finite extreme observation (1e20) already raises
"emission log_prob contains NaN/Inf"; that path did not demonstrate runtime
state corruption. Do not generalize the public-kernel counterexample into a
confirmed default-runtime poisoning claim.

### C01 — P1: Float32 bookkeeping changes convergence and best-iteration decisions

Owner: nhsmm/convergence.py, tracking arrays at lines 96–100 and _record_score /
_plateau_convergence at lines 168–180 and 202–214.

Scores arrive as Python floats, including scores from double-precision models,
but history and best_scores are unconditionally Float32. Inputs
[100000000,100000001,100000002] become three identical stored scores.
With plateau_window=3 and plateau_tol=0, the monitor reports convergence and
keeps best_iter=0 although each input strictly improves. Conversely, repeating
the identical score 100000000.1 with delta mode, patience=2 and zero tolerances
produces deltas approximately 0.1 and never converges.

Expected: retain the precision of supplied scalar scores consistently for
history, deltas and comparisons; Float64 bookkeeping is a practical bounded
choice. Keep monitor state independent of model parameter dtype.

NHSMM.optimize uses the returned convergence decision when convergence_stop=True,
so premature termination is an actual affected caller. Its final restart winner
uses separate Python-float scoring; this finding does not prove that winner
selection itself is corrupted by monitor.best_scores.

### F02 — P2: public filter state does not enforce normalized probability mass

Owner: nhsmm/filtering.py, HSMMFilterState.__post_init__, lines 19–33.

HSMMFilterState(full((1,1,1),log(2))) is accepted: the constructor checks only
finite total log mass. state_posterior sums to 2 and
next_episode_end_probability(state,zeros(1,1,1)) returns 2.0.

Expected: reject materially unnormalized public states, with a documented
dtype-aware tolerance; do not silently interpret arbitrary scores as posteriors.
Keep the internal trusted construction path tied to a checked normalizer.

### F03 — P2: trace constructor accepts NaN and invalid posterior/padding content

Owner: nhsmm/filtering.py, HSMMFilterTrace.__post_init__, lines 71–89.

A [1,2,1,1] trace filled with NaN and lengths=[1] is accepted, and
state_posterior returns NaNs. Checks cover shape/dtype/length bounds but omit
probability validity, normalization of valid rows and the documented -inf
padding contract.

Expected: validate valid rows and padded rows separately. Preserve permitted
zero-length sequences/empty time dimensions; do not normalize padding into mass.

### C03 — P2: convergence export emits non-standard JSON for unfinished restarts

Owner: nhsmm/convergence.py, Convergence.export.

For n_init=2, updating only initialization zero and calling export writes
best_scores=[1.0,-Infinity]. The score matrix converts missing entries to null,
but best_scores uses raw tolist(). A strict JSON reader rejects the result.

Expected: represent missing best scores as null consistently, and serialize with
allow_nan=False so future non-finite export paths fail explicitly. Cover an
untouched monitor and a partially completed multi-initialization monitor.

### G02 — P2: from_dict silently merges distinct scenario identifiers

Owner: nhsmm/config.py, ValidationConfig.from_dict, lines 560–567.

The mapping {1:{"separation":1},"1":{"separation":2}} becomes one scenario named
"1" with separation=2. The direct constructor requires non-empty string names;
from_dict first coerces names with str(), defeating that contract and silently
discarding a validation case.

Expected: enforce the same string-name rule before constructing the output
mapping. This affects Python Mapping input; JSON object keys already are strings.

### G03 — P2: accepted seeds exceed actual consumer domains

Owner: nhsmm/config.py, ModelConfig seed validation around lines 165–169 and
ValidationConfig seed validation around lines 496–501.
Consumers: NHSMM.__init__ and scripts/validate_learning_prediction.py run_one.

ModelConfig(seed=2**64) succeeds, but NHSMM(config) raises an overflow error in
torch.manual_seed. ValidationConfig(seeds=(-1,)) likewise succeeds, while the
validation runner's np.random.seed(-1) raises before generating data.

Expected: validate seeds against their respective consumer domains, including
derived evaluation seeds seed+oos_seed_offset. PyTorch model seeds and legacy
NumPy validation seeds do not have identical domains; do not reject all negative
model seeds merely because the NumPy validation runner cannot use them.

### G01 — P3: schema version accepts a boolean and changes serialized type

Owner: nhsmm/config.py, ValidationConfig.__post_init__, lines 490–495.

ValidationConfig.from_dict({"schema_version":true}) accepts True because
True==1; serialization retains a JSON boolean instead of the declared integer
version. A float 1.0 follows the same equality-only validation path.

Expected: require an integer excluding bool before checking the supported value.

## Observations requiring a contract decision

Convergence.stop_training remains true after a subsequent rejected non-finite
update clears all_converged. This is reproduced in the evidence, but a latched
stop request until reset can be intentional; it is not counted among the eight
confirmed defects. NHSMM.optimize uses the update return value rather than this
flag. Likewise, adaptive patience currently applies only to delta mode, while
plateau uses its exact rolling window. Clarify that mode distinction before
changing behavior; the existing exact-window test is an intentional constraint.

## Validation and limits

Focused unchanged-code suite: **148 passed in 8.21 seconds** on CPU.
Command (thread limits OMP/MKL/OPENBLAS=1 and PYTHONDONTWRITEBYTECODE=1):

    /work/env/python/bin/python -m pytest -q -p no:cacheprovider \
      tests/test_convergence.py tests/test_filtering.py \
      tests/test_config_contract.py tests/test_model_filter.py \
      tests/reference/test_exact_filtering.py \
      tests/reference/test_production_filtering_equivalence.py

The passing suite includes independent probability-space filtering equivalence,
finite and D+ duration handling, same-state renewals, prefix/padding behavior,
ordinary gradients and current configuration contracts. It does not cover the
counterexamples above.

The separately staged reproduction program passed syntax checking and explicit
assertions for the current defects and the rejected Gaussian-runtime suspicion.
Machine-readable observations: convergence-filtering-config-audit-20261015.json.
No production fix is claimed, no full canonical suite was rerun for this audit,
and CUDA was not exercised.

Suggested repair batches: (1) F01/F02/F03 with support, padding, gradients and
runtime atomicity checks; (2) C01/C03 with scalar precision, restart history and
strict export checks; (3) G01/G02/G03 with roundtrip and consumer-boundary checks.
After shared inference/configuration changes, run the broader canonical suite
and report historical evidence-identity failures separately.

Reproduction program SHA256: 014ce5b8ffe5f2a796adc79f9048be4b62a656166987aa51fdc42bcad0676e02.
Terminal reproduction job: 9b32540e495d463ab269c0380e17fde5; all assertions passed.
