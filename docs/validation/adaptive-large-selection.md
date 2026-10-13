# Large-sample adaptive penalty selection — Step22a

Status: predeclared, not executed. Audit base: `1a57fd534ffcbc37f14569ac9fb5b31616997e9b`.
Machine-readable contract: [adaptive-large-selection-protocol.json](adaptive-large-selection-protocol.json).

## Source audit and question

`scripts/compare_adaptive_duration_em.py::select_penalty` selected factors
0.5/1/2 on a 48/16 split with six updates per candidate.
`scripts/compare_adaptive_warmstart.py::paired_selection` compared uniform
and warm paths on the same 48/16 split, with nine updates per path (warm3+fit6).
`scripts/compare_adaptive_sample_size.py` then held factor2 fixed at 1024
sequences; it did not renew inner selection. Step21's one-of-three recovery
result therefore tests that frozen rule, not large-exposure model selection.

The next controlled question is whether the same training-only likelihood
criterion, with larger exposure and matched warm3+fit27 paths, chooses a
different penalty and improves recovery on fresh data. Penalty controls
adaptive block capacity; no new partition algorithm is introduced here.
Predictive likelihood selection still cannot guarantee fine-PMF recovery.

## Frozen experiment

Use fresh seeds1906/1907/1908. Generate1024 independent300-row observation
sequences per seed with the existing sigma2 known-nuisance generator.
Permute sequence indices using seed+600000 and split768/256; do not split
rows within an episode. Existing generator offsets remain +400000 for
training and +500000 for24 independent outer sequences.

Fit one three-update unpenalized warm state on the768 inner-fit observations.
Copy it into each factor0.125/0.5/2 candidate, then run27 penalized updates
with penalty factor*log(768). Score the256 inner-validation observations
with marginal likelihood. Maximum total validation LL selects the factor;
an exact tie selects the stronger factor. Retain all candidates, hashes,
indices, PMFs, learned blocks and phase histories.

Freeze the selected factor before constructing outer observations or invoking
true_law for evaluation. Refit on all1024 training observations using a new
shared warm3 state and fit27 with factor*log(1024). Compare against frozen
factor2 using that same warm state and budget. If selection chooses2, reuse
the identical completed fit rather than recomputing a duplicate control.

All ages1..260 remain available for every state; oracle support stays OFF.
Positive pseudocount1e-6, nuisance parameters, valid episode starts and terminal
right censoring remain unchanged. No labels or true support/PMF enter learning
or selection. Previous seeds1903..1905 are historical evidence, never selection
inputs or fresh confirmation data.

## Acceptance, cost and recovery

Keep each-state PMF TV<=0.12 and positive outer gain over full-support uniform
as separate recovery gates per seed; preserve existing public likelihood,
gradient and filtering numerical gates. Report selected factors and all seed
failures. Diagnose learned-partition TV bounds only after completed fits.
Do not change factors, gates or update budgets after seeing outer outcomes.

Each seed needs3 shared inner warm +3*27 candidate +3 full-data warm +27
selected-refit updates:114 actual M-steps. A distinct factor2 control adds27,
giving at most141 per seed (342..423 across three). These counts exclude
scoring/E-step replay checks. Execute one worker; checkpoint every existing
five-update boundary and each phase/candidate completion. Bind resume to
source/contract/data identities and preserve already completed candidates.
A recycled runtime is not evidence of a continuing or completed fit.

## Verified milestone and remaining work

Step22b implements scripts/compare_adaptive_large_selection.py using the existing
fit_adaptive, score/evaluation owners and atomic-save primitive. Seed, protocol,
software and source hashes bind resume; training and inner split hashes bind
phase replay. Completed phases must be an ordered prefix. Every five-update
checkpoint and phase completion preserves PMF/partition/history; resume checks
positive normalized full support and re-scores the last objective. Completed
phases are re-scored without additional M-steps. Selected factor is saved before
outer data construction. A selected factor2 reuses its fit for the control.

Use --max-phases to bound newly completed phases per invocation; existing fits
are retained. --resume continues the same source/contract/data identity. Completed
evaluation is a no-op. Separate seed/output files permit independent later work,
but this frozen experiment retains one compute worker. Connector reads and
independent checks can run in bounded parallel lanes.

Fifty required current source/doc/test files are restored and Git-blob verified
(one evidence-backed dependency reopen for the existing warmstart test helper).
Declared CPU dependencies and editable install succeed: Python3.12.14,
NumPy2.5.4, Torch2.14.1+cpu, pytest9.1.1. AST/Ruff/Black pass44 scoped Python
files. Focused tests include real interruption/resume equality at both phase
start/end, phase limits, data/contract/PMF rejection, tie/control reuse, separate
control fitting, selection-only observations, completed no-op and a real short
fit/public likelihood/gradient/filter evaluation smoke. No large recovery result
or improvement is claimed by this preparation. Next: bounded large phases and
verified checkpoint writeback. Production/joint Trading acceptance is unchanged.

## Step22c — verified seed1906 execution checkpoint

Raw checkpoint: results/adaptive-large-selection-seed1906-native-20261013.json.
Source3966159902a48bfc2c1fad6a22c375a15cd71dee remains frozen. The first bounded
invocation exits0 after inner_warm updates0..3 on768 training observations.
All13 source identities, protocol/software/seed binding,1024-data and768/256
split replay hashes, positive normalized full support, partition consistency
and fixed-phase objective monotonicity pass. Terminal E-step replay reproduces
LL-499446.2814603957 and objective-499446.28935215634; blocks254/250/228/129.
Selection/evaluation remain null. This is execution evidence, not recovery.

Next bounded command, from the repository root:

```bash
timeout 1200 ../nhsmm-env23/bin/python -m scripts.compare_adaptive_large_selection \\
  --base-commit 3966159902a48bfc2c1fad6a22c375a15cd71dee --seed 1906 \\
  --output docs/validation/results/adaptive-large-selection-seed1906-native-20261013.json \\
  --resume --max-phases 1
```

This starts candidate_0 (factor0.125), reusing the verified inner warm state.
Completed phases are re-scored without new M-steps. If interrupted, the same
command resumes the active candidate from its last atomic five-update snapshot.
Native checkpoints are durable only after connector writeback. The source commit
argument stays3966159 while later commits modify only documentation/evidence.
Seeds1907/1908 are not started. No outer outcome has changed the frozen protocol.

Step22d persists an active candidate_0 snapshot at update10/27,67/73/57/36
blocks. Source/contract/prior warm and split identity, active budget/history,
full-age PMFs and block consistency pass. Independent terminal scoring and
selection remain pending. The raw file is a recovery checkpoint; native fitting
continues under unchanged3966159. No outer data/truth evaluation is performed.
