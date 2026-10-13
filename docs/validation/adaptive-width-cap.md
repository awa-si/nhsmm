# Adaptive width-cap mechanism pilot

The completed large-selection experiment has positive outer likelihood gains
but zero full recovery passes in three seeds. Every failing state has a
learned-partition TV lower bound above0.12. Block-mass optimization alone cannot
fix those particular partitions. This does not rule out the adaptive family.

The new estimator restricts each constant-density block to at most8 configured
ages, retaining full support1..260 and pseudocount1e-6. The exact M-step uses the
same augmented-count log-density objective and block-count penalty, but limits
admissible starts in the prefix dynamic program. It remains globally optimal
within this restricted partition class; an observed-objective monotonicity check
is mandatory. The unrestricted control uses the unchanged existing fitter.

A shared unpenalized warm PMF is fitted on training observations. Deterministic
splitting of wide warm blocks preserves every PMF value and provides a feasible
initial width8 partition. No label, true support, boundary or true-PMF value
defines the splits. Both arms use factor2*log256 and27 M-steps after warm3.

The committed JSON contract freezes fresh seeds1909/1910/1911,256 training
sequences and24 independent outer sequences,300 rows, known Gaussian nuisance
parameters and concurrency1. It is a matched mechanism pilot rather than a
replacement1024-sequence acceptance result. Evaluate both arms without selection.
Do not compare their penalized objectives as if their feasible classes were
identical; compare per-seed outer LL and duration-PMF metrics explicitly.

Histogram model selection and dynamic-program cut-point optimization are
established methods: Kontkanen and Myllymaki,2007,
[MDL Histogram Density Estimation](https://proceedings.mlr.press/v2/kontkanen07a.html).
Their NML criterion differs from this experiment's frozen simple penalty.
Liu,Li and Wong,2023,
[Density Estimation via Adaptive Partitioning](https://www.jmlr.org/beta/papers/v24/20-060.html)
also distinguish approximation spaces and partition complexity; their theory
does not establish finite-sample HSMM recovery here. Width8 is our predeclared
mechanism intervention, not a published optimality claim.

Current checkpoint: protocol committed before fitting. Next implement the
restricted M-step, verify exhaustive small cases, actual short-fit/resume,
then execute and independently replay all three paired fits. Oracle stays OFF.

## Step26 — implementation verified before fitting

Step26 implements the separate research-only width-cap M-step/fitter and checkpointed paired driver. Exhaustive bounded partition optimality, inactive-cap equivalence, positive zero-exposure PMF, probability-preserving warm refinement, real censored short EM/interrupted resume, guarded source/contract/seed/prefix/history, post-fit public numerics and completed no-op checks pass.44 focused tests pass in12.94s; new3-file Ruff/Black/syntax checks pass. Existing adaptive/production owners and prior frozen source hashes remain unchanged. Next execute frozen256-sequence paired pilot, concurrency1, oracle OFF.


## Step27-1909-active

Width-cap pilot seed1909 checkpoint: unrestricted at10/27,
blocks[4, 3, 5, 4]. Source/software/contract/data/phase/history/
objective/PMF/partition and shared warm-PMF continuity checks pass. This is
recovery evidence; final terminal replay/outer/numerical and recovery gates
remain pending. Frozen source8fc245e,256 training sequences, oracle OFF.

## Step27-1909-active-width8

Width-cap pilot seed1909 checkpoint: width8 at10/27,
blocks[33, 33, 33, 33]. Source/software/contract/data/phase/history/
objective/PMF/partition and shared warm-PMF continuity checks pass. This is
recovery evidence; final terminal replay/outer/numerical and recovery gates
remain pending. Frozen source8fc245e,256 training sequences, oracle OFF.
