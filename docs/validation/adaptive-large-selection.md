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

## Step22e — complete first candidate,2026-10-13

Candidate_0 finishes27/27 updates at factor0.125 and penalty0.830473716643459.
Its final LL/objective are-499451.2684951689/-499641.4594240784; learned block
counts are67/73/57/36. Independent terminal replay, all13 source identities,
full training/inner split replay, phase indices, fixed-objective monotonicity
and positive normalized full-support partition checks pass. The prior warm
record is unchanged. Total completed work is30 M-steps.

Inner validation LL, independently re-scored on the256 inner-validation
observations, is-166868.3480591295. This is one candidate score; no factor
is selected yet. Both selection and evaluation remain null, so no outer recovery
claim follows. SHA-bound verification is in
results/adaptive-large-selection-seed1906-candidate0-verification-native-20261013.json.
The raw result supersedes the active Step22d snapshot with a completed phase.
The same bounded resume command above next executes candidate_1, factor0.5.
Source3966159 and all frozen settings, including oracle OFF, remain unchanged.

Step22f captures candidate_1, factor0.5, at update10/27 with
18/13/19/9 blocks. Thirteen source hashes, unchanged prior completed phases/
data/contract identity, active budget/history, normalized positive full-age PMFs
and partition consistency pass. Snapshot SHA256a4895fdd084f4cb236c6e1c431b62884e1f010b3c2c84097a4c2d66d13489e19.
Terminal independent scoring and selection are pending; fitting continues.

## Step22g — complete second candidate,2026-10-13

Candidate_1 finishes27/27 updates with factor0.5, penalty3.321894866573836,
LL-499629.0984813526, objective-499811.8135767134 and18/13/19/9 blocks.
All13 source hashes, training/split replay, full positive normalized PMF and
partition/history/objective checks pass. Independent terminal E-step replay
reproduces the score; prior warm and candidate_0 records are unchanged.
Total completed work57 M-steps. Inner validation LL is-166776.27502647758,
improving on candidate_0 by92.07303265191149 nats on the same256 validation sequences.
No final factor is selected: candidate_2 (factor2) is still pending. Selection
and outer evaluation remain null. Phase verification is in
results/adaptive-large-selection-seed1906-candidate1-verification-native-20261013.json.
The same resume command above next runs candidate_2 at unchanged3966159.
The raw completed report supersedes Step22f's active checkpoint.

Step22h saves candidate_2 factor2 at update10/27 with5/4/7/5 blocks.
Source/data/contract and prior completed phases are preserved; active budget,
phase indices, fixed-objective monotonicity and positive normalized full-age
PMF/partition coverage pass. SHA256bd646722f7613d41b03a61854498e5fca926086df7fd3897d492eeceb2c74856. Selection/outer evaluation
are still null. The native bounded seed1906 batch continues under3966159.

## Step22i — selection and active full-data refit checkpoint

All three inner candidates finish; their marginal validation LLs are:

| Factor | Inner validation LL |
| --- | ---: |
|0.125|-166868.3480591295|
|0.5|-166776.27502647758|
|2|-166718.363173187|

The predeclared criterion selects2, before any outer data or truth evaluation.
Its gain over0.5 is57.911853 nats on the same256 validation sequences.
The independent full-data warm3 is complete. Active selected_refit reaches
5/27 with6/4/8/5 blocks, penalty2*log(1024). Because selected factor
is2, this same completed fit will represent the matched factor2 control.

Thirteen source hashes, original completed phase/data/contract identities,
indices, monotonic fixed objectives, normalized positive full-age PMFs, block
coverage/densities, scalar objective recalculation and phase continuity pass.
Snapshot SHA256c337e6e6fac1ec8a73d038fd0ba7a1007708884823133026c4308d0c5da1e5bc. This is an execution checkpoint; terminal
independent E-step/public numerical/recovery verification remains pending.
Outer evaluation is null and the native bounded batch continues at3966159.

## Step22j — complete seed1906 result,2026-10-13

The complete native batch at3966159 uses114 actual M-steps. All three inner
validation scores and factor2 selection are independently reproduced. Both
selected and frozen factor2 control refer to the identical completed1024 fit;
no performance difference between these arms is claimed. The initial warm and
candidate_0/1 records remain unchanged. Seeds1907/1908 remain pending.

| Metric | Selected/control result |
| --- | --- |
|Fine-PMF TV, bull/bear/range/chaotic|0.095938 /0.108081 /0.143096 /0.126634|
|Mean RMSE|1.448104|
|Outer gain over full-support uniform, nats/row|+0.005741339|
|Learned block counts|6 /4 /8 /5|
|Full recovery gate|FAIL: range/chaotic|

Independent verification passes all114 phase budgets/histories, source/data/
split replay and objective/PMF/partition invariants. Candidate_2, full_warm and
selected_refit terminal E-steps reproduce saved evidence without new M-steps.
Outer observations are generated only after complete training and selection;
24-sequence replay, baseline/fitted likelihood, metrics/gates and public numerical
checks pass. Gradient/public-recursion LL/state-probability maxima are
1.4433e-15 /4.5475e-12 /1.0148e-13. Completed resume is a no-op.

Final SHA-bound evidence:
results/adaptive-large-selection-seed1906-verification-native-20261013.json;
raw report: results/adaptive-large-selection-seed1906-native-20261013.json.
Raw SHA2567f3c68610d2dc4fac25d507a8829dce43f198f9ad747a523c1d4304dc85599f5.

Post-fit median-TV bounds by learned partition are0.093883/0.106256/0.125738/
0.125731; normalized forward-KL projection TVs are0.095822/0.113457/0.131687/
0.128146. Half the sum of deviations from each block's true-PMF median is a
valid lower bound for every density constant on those blocks, even before its
normalization constraint. Range and chaotic bounds exceed0.12, proving that
those fitted partitions cannot pass solely through additional mass optimization.
The forward-KL projection is feasible, not the minimum normalized-TV solution.
These bounds concern the learned partitions, not all possible adaptive partitions.

Predictive improvement does not establish fine-PMF recovery. Adequate-exposure
inner likelihood selection still chooses2 and leaves insufficient partition
resolution in this first fresh seed. No thresholds, oracle support, candidate
factors or learning parameters are changed in response to this outcome. Complete
the frozen1907/1908 replications before broad conclusions or a new estimator
change; known nuisance and production/joint Trading limitations remain unchanged.

## Step22k — warm-start checkpoints for1907/1908,2026-10-13

Both remaining fresh seeds complete inner_warm3 at unchanged3966159. Independent
terminal E-step/source/data/split/PMF/partition/objective replay passes. Both
active/selection/evaluation fields are null, and no outer data is evaluated.
The datasets have distinct hashes. Total new work is6 M-steps; completed1906
remains an independent failed recovery result under the same frozen protocol.

| Seed | Warm blocks bull/bear/range/chaotic | Checkpoint |
| --- | --- | --- |
|1907|260/253/233/186|results/adaptive-large-selection-seed1907-native-20261013.json|
|1908|254/254/245/153|results/adaptive-large-selection-seed1908-native-20261013.json|

Next resume each checkpoint with source3966159902a48bfc2c1fad6a22c375a15cd71dee,
its matching --seed and --output, --resume and --max-phases1; this begins
candidate_0 at factor0.125 without repeating the warm M-steps. Keep separate
output files and unchanged statistical settings. Later phase/selection/refit/
outer gates remain pending; no three-seed conclusion is made yet. Admin layers
are fully refreshed and the active regime-v2 branch is retained.

Step22l captures seed1907 candidate_0 (factor0.125) at10/27 with
76/63/65/34 blocks. Thirteen source hashes, preserved prior warm/data/contract,
phase budget/indices, positive normalized full-age PMFs and contiguous partition
checks pass. Fixed-objective history is monotonic. Snapshot SHA256c86b6d4fdfa9765d5a3f18543c4db58bc91d4863b9251eff1e7f602136f5ab7d.
This is active execution evidence; terminal replay and inner validation scoring
are pending. Source3966159 is unchanged; native fitting continues. Seed1908
has its durable warm3 and has not yet begun candidate_0.

## Step22m — seed1907 candidate_0 complete

Factor0.125 completes27 updates with76/63/65/34 blocks. All source/data/split/
PMF/partition/history/objective checks and independent terminal E-step replay
pass; the warm record is unchanged. Final LL/objective are-499961.7578440498/
-500156.10102691303. Independent inner validation LL is-166754.64328071033.
Total1907 work30 M-steps. No factor is selected and no outer evaluation occurs.
SHA-bound evidence: results/adaptive-large-selection-seed1907-candidate0-verification-native-20261013.json.
Next seed1908 executes candidate_0 from its separately verified warm3; all source,
method, support and acceptance settings remain fixed. Cross-seed absolute inner
likelihoods are not treated as a ranking of method performance.

Step22n captures seed1908 candidate_0 at10/27 with68/80/64/37 blocks.
Thirteen source hashes and unchanged prior warm/data/contract, active budget/
history, positive normalized full-support PMFs and partitions pass. Fixed
objective history is monotonic. Snapshot SHA256273107f1b5c123d87531e084f415a3ae8e85597397a27161ae5b5be0ad499625. This is execution
recovery evidence only; terminal independent score and inner validation remain
pending. Seed1907's first candidate is durably complete; source3966159 and all
method settings remain unchanged. Native seed1908 fitting continues.

## Step22o — first-candidate batch complete for1907/1908

| Seed | Factor | Blocks bull/bear/range/chaotic | Inner validation LL |
| --- | ---: | --- | ---: |
|1907|0.125|76/63/65/34|-166754.64328071033|
|1908|0.125|68/80/63/37|-167330.094205874|

Both complete27 updates. All source/software/contract/data/split/history/PMF/
partition/objective checks and independent terminal E-step replays pass; each
prior warm record is unchanged. The source/software/protocol match across seeds
and their training hashes differ. These are separate validation datasets; the
absolute LL values do not rank method performance across seeds.

Total54 new candidate M-steps,60 including the previous warm starts. SHA-bound
joined evidence: results/adaptive-large-selection-candidate0-batch-native-20261013.json.
Seed1908 phase evidence: results/adaptive-large-selection-seed1908-candidate0-verification-native-20261013.json.
Selection and outer evaluation remain null. Next bounded resume each seed with
its matching output/source3966159 starts candidate_1 (factor0.5). No method,
support, threshold or budget changes are made; oracle remains OFF.
