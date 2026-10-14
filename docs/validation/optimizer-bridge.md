# Optimizer bridge — Step35

## Step39 —300-row resource smoke completed

The fixed fresh401918 training-only smoke runs the actual public optimizer on
4x300 rows, one constant-temperature Adam update at lr0.01, full positive
duration support and frozen known nuisance/current initialization. It takes
1.795389s in the observed native environment; the whole process peaks at
581704 KiB (568.07 MiB). Terminal target/public/reference LL error is
9.094947017729282e-13. It uses no held-out data or labels to choose a model.
Source d2d2389a and exact data/core/runner hashes are recorded in the raw report.

Next resource boundary is exact gradient accumulation over bounded sequence
chunks: sum the sequence likelihood gradients, then clip and take one Adam
step per full batch. Test against unchunked float64 loss/gradients/updates on a
small problem, including uneven final chunks and variable lengths. Fixed-target
monitoring must also aggregate over the full training sample. A full-size fit
requires a separately frozen statistical protocol and measured bounded memory;
this single4-sequence timing does not justify linear memory extrapolation.

## Step38 — pilot complete and independently verified

Source21428b6f, fresh seeds1915/1916/1917,4 training and2 outer sequences of60
rows, sigma2,2 updates per arm,18 total updates. All21 source identities and
data hashes, terminal training/public target likelihoods, independent outer
LL/gradient/filter replays and byte-preserving completed CLI no-op pass.
The verifier performs0 fitting updates and validates saved endpoints; it does
not reproduce the optimizer trajectory. Frozen nuisance preservation is
checked during the fits and its certificate is checked during verification.

| Seed | Constant Adam gain/row | Annealed Adam gain/row | EM gain/row |
| --- | ---: | ---: | ---: |
| 1915 | +0.000148862 | +0.000148224 | +0.003463649 |
| 1916 | +0.000114517 | +0.000114034 | -0.094563277 |
| 1917 | +0.000077814 | +0.000077500 | +0.002064989 |

All three arms fail every seed's full recovery gate (TV<=0.12 in each state
and positive outer gain). Constant Adam has a slightly larger outer gain than
annealed Adam on these three runs, but the tiny budget/sample does not establish
statistical superiority, convergence, equal compute, or a cause of earlier
recovery failures. No arm or hyperparameter is selected from these outcomes.

### Post-fit ground-truth exposure audit

Training labels are used only after completed fits to count observed completed
episodes, treating each final run as censored.

| Seed | Completed bull/bear/range/chaotic | Terminal-censored bull/bear/range/chaotic |
| --- | --- | --- |
| 1915 | 0/0/1/1 | 2/1/1/0 |
| 1916 | 0/0/1/0 | 2/1/1/0 |
| 1917 | 0/0/1/0 | 2/1/1/0 |

Under the configured ground truth, P(duration>60) is
[0.9830170255,0.7730295881,0.5688167179,0.0888006994].
These short sequences provide very little completed-episode evidence for long
durations. Censoring is handled by the likelihood; correct censoring does not
create missing information. These counts do not prove mathematical
non-identifiability, but make the2-step pilot unsuitable for recovery claims.

Next execution gate: measure a300-row actual public optimizer smoke before
allocating a larger-exposure study. Then freeze fresh data/budgets/target model
family and optimizer comparisons before outcomes; compare matching duration
families and report actual compute. Free emissions/transitions/initial and
neural context only after duration-only validation. Oracle remains OFF.

The completed fixed-grid study is accepted only for duration learning with
known nuisance parameters. Both unsmoothed and smoothed arms pass on3/3 seeds;
smoothing improves matched outer LL on3/3 seeds, but does not improve every
state TV or every mean RMSE. These observations do not prove a universal
smoothing benefit, a sample-size effect, or joint neural recovery.

## Bounded temperature diagnostic (2026-10-14)

No fitting, no labels, full positive configured ages1..260, no true-law support
mask. Generate2 sequences of40 observations at sigma2 with seed401915;
use a uniform duration PMF and the existing known-nuisance public model helper.
The helper retains an all-true support buffer; it excludes no configured age.

At the same unchanged parameters, public `log_likelihood` and `_compute_loss`
with both temperature bounds1 and loss_bias0 return exactly
`-177.68856980552732` (absolute difference0).
With `_compute_loss`'s default schedule, iterations0/1/2 of a3-step budget
return `[-177.6885917737634, -177.68913551351594, -177.69322209619298]`.
The fixed-parameter LL span is0.0046303224295911605.
The two regression tests also verify state tensors remain unchanged.

This establishes an objective mismatch, not a faulty likelihood recursion:
temperature divides distribution logits and changes the model being scored.
`NHSMM.optimize` calls this scheduled objective for both updates and final
restart scoring. Thus convergence traces and restart scores are not generally
the public temperature1 likelihood. This diagnostic alone does not establish
that annealing causes the observed statistical recovery failures.

## Next bounded implementation boundary

1. Expose an explicit temperature1 optimizer control, retaining existing
   annealing as a named option. Test config validation and default compatibility.
2. Score final restart snapshots using the declared target temperature, so
   traces identify scheduled versus target objectives explicitly. Do not remove
   the causal encoder or change censoring/runtime semantics.
3. Before any long fit, verify the actual optimize path (including initialization,
   frozen parameters, final snapshot and restart selection), then preregister a
   fresh duration-only optimizer comparison. Match data, initialization and target
   model family; Adam steps and EM M-steps are not equivalent compute budgets.
4. Only after that comparison, progressively free emissions, transitions and
   initial probabilities, then neural context. Report each boundary separately;
   do not jump from known-nuisance EM to Trading acceptance.

Step35 does not alter production optimizer defaults, the frozen study source,
raw outcomes or gates; no further long study has started. Oracle remains OFF.
