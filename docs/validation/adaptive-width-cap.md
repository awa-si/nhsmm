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

## Step27-1909-complete

Width-cap pilot seed1909 completes57 actual M-steps (warm3 plus27 per arm).
All16 source/software/contract/training/outer identities, shared initial PMF,
matched budgets, phase/history/objective/positive full-age PMF/partition checks,
independent terminal E-step replays, regenerated outer evaluation, public
likelihood/gradient/filtering and byte-preserving completed no-op pass.
Width8 minus unrestricted outer LL: 15.788184175442439 nats.
Main report SHA2564d37e4ba58d861d44c6e6ee18d104e97159da263561ccd61ccae7f17d94cf5bc. All results are pilot evidence,
not1024-sequence or production/joint-neural/legacy NPZ acceptance. Oracle OFF.
unrestricted: blocks[4, 3, 5, 4], TVs[0.1310162390517595, 0.14796260880209172, 0.17532581976170813, 0.17586775837164556], gain0.005149776424069488 nats/row, full gateFAIL, partition TV lower bounds[0.12470794605793709, 0.14701677785676828, 0.17078985610860292, 0.17272074620117248].
width8: blocks[33, 33, 33, 33], TVs[0.185082841536105, 0.15511084872243294, 0.1490903352546657, 0.14827088207990077], gain0.007342579781769827 nats/row, full gateFAIL, partition TV lower bounds[0.02390236738745256, 0.02409627774125917, 0.05118498048109232, 0.05096372213525549].

## Step27-1910-active-unrestricted

Width-cap pilot seed1910 checkpoint: unrestricted at15/27,
blocks[4, 4, 5, 4]. Source/software/contract/data/phase/history/
objective/PMF/partition and shared warm-PMF continuity checks pass. This is
recovery evidence; final terminal replay/outer/numerical and recovery gates
remain pending. Frozen source8fc245e,256 training sequences, oracle OFF.

## Step27-1910-active-width8

Width-cap pilot seed1910 checkpoint: width8 at5/27,
blocks[33, 33, 34, 33]. Source/software/contract/data/phase/history/
objective/PMF/partition and shared warm-PMF continuity checks pass. This is
recovery evidence; final terminal replay/outer/numerical and recovery gates
remain pending. Frozen source8fc245e,256 training sequences, oracle OFF.

## Step27-1910-complete

Width-cap pilot seed1910 completes57 actual M-steps (warm3 plus27 per arm).
All16 source/software/contract/training/outer identities, shared initial PMF,
matched budgets, phase/history/objective/positive full-age PMF/partition checks,
independent terminal E-step replays, regenerated outer evaluation, public
likelihood/gradient/filtering and byte-preserving completed no-op pass.
Width8 minus unrestricted outer LL: -28.033464356243712 nats.
Main report SHA25636a9e6e24cf2188fd5ced16008bf18f057a8b367d5747efd57178636281a6e57. All results are pilot evidence,
not1024-sequence or production/joint-neural/legacy NPZ acceptance. Oracle OFF.
unrestricted: blocks[4, 4, 5, 4], TVs[0.1512682124673306, 0.11415891768091632, 0.19243455817006525, 0.14708438297226797], gain0.005200034136191284 nats/row, full gateFAIL, partition TV lower bounds[0.13949747652704678, 0.11036037126376683, 0.18967903496131433, 0.1358535170371385].
width8: blocks[33, 33, 34, 33], TVs[0.16021754190356455, 0.22176273932976012, 0.21373524132051733, 0.12752437326266428], gain0.0013064974200463237 nats/row, full gateFAIL, partition TV lower bounds[0.023814457857965984, 0.02430026697124297, 0.04952418313143421, 0.05010040354108249].

## Step27-1911-active-unrestricted

Width-cap pilot seed1911 checkpoint: unrestricted at5/27,
blocks[3, 4, 5, 4]. Source/software/contract/data/phase/history/
objective/PMF/partition and shared warm-PMF continuity checks pass. This is
recovery evidence; final terminal replay/outer/numerical and recovery gates
remain pending. Frozen source8fc245e,256 training sequences, oracle OFF.

## Step27-1911-active-width8

Width-cap pilot seed1911 checkpoint: width8 at5/27,
blocks[33, 33, 33, 33]. Source/software/contract/data/phase/history/
objective/PMF/partition and shared warm-PMF continuity checks pass. This is
recovery evidence; final terminal replay/outer/numerical and recovery gates
remain pending. Frozen source8fc245e,256 training sequences, oracle OFF.
