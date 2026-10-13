# Large-sample adaptive penalty selection — Step22a

Status: predeclared, not executed. Audit base: `1a57fd534ffcbc37f14569ac9fb5b31616997e9b`.
Machine-readable contract: [adaptive-large-selection-protocol.json](adaptive-large-selection-protocol.json).

## Source audit and question

`scripts/compare_adaptive_duration_em.py::select_penalty` selected factors
0.125/0.5/2 on a 48/16 split with six updates per candidate.
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

Current source audit and protocol arithmetic are complete; training is pending.
The prior env21 runtime is absent, and available system Python lacks Torch.
No new training, smoke result, recovery result or improvement is claimed.

Next implementation should compose existing fit_adaptive, emission_scores,
aggregate, evaluation and atomic-save primitives in a new resumable research
driver. Restore and verify its material source closure and declared runtime;
pass syntax/static and a short real-fit/resume smoke before long execution.
Persist verified stages through the connector under the standing stepwise
writeback instruction. Public package and joint Trading integration remain
outside this experiment.
