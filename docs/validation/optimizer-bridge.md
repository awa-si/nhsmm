# Optimizer bridge — Step35

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
