# Fixed-bin block-mass smoothing

Step31: scripts/compare_block_mass_smoothing.py now owns atomic checkpoints,
strict phase/source/software/contract/data/solver history guards and bounded
warm/unsmoothed/smoothed phases. It reuses the existing full-support matched
outer/public evaluator through an immutable arm-name translation.65 focused
cases and new-file static checks pass. Freeze the runner commit before large
fits, then invoke --source-commit with that immutable SHA, --seed1912/1913/1914,
and a separate docs/validation/results/block-mass-smoothing-seed<seed>-native-20261014.json.
Resume only matching source/software/contract/output with --resume; --max-phases
bounds newly completed phases. Completed experiments are no-ops. Oracle OFF.

Current checkpoint (Step30): the M-step/objective is implemented separately in
scripts/block_mass_smoothing.py. All58 focused tests (14 new) and the new-file
static gates pass. The preflight evidence binds exact source/test/protocol
hashes at matching runtime versions. The1024-sequence study is not started;
next build and test the checkpointed matched runner, then freeze its source
before bounded training phases. No empirical recovery result is claimed here.

The width-cap pilot proved that its partitions admit normalized approximations
to the true duration PMF with TV below0.12, yet every fitted width8 state failed
that gate. Increasing partition resolution alone was not a robust improvement.
The next predeclared experiment isolates regularization of masses within a fixed
full-age grid; it does not assume that the public HSMM recursion is defective.

For bin log densities z and widths w, log p_age=z_bin-logsumexp(z+log w).
With augmented posterior duration counts C_b and total N, the M-step maximizes
sum(C_b*z_b)-N*logsumexp(z+log w)-lambda/2*||Rz||². R computes differences
between neighboring log-density slopes at bin centers; the shorter final bin is
accounted for. No cyclic difference or virtual endpoint neighbor is used.

The Hessian is -N*(diag(m)-m*m^T)-lambda*R^T*R, with bin masses m.
It is negative semidefinite; fixing the last z to0 removes the additive shift.
Positive per-age pseudocounts give a finite strictly concave anchored problem.
Lambda0 has the exact analytic augmented block-mass solution. Positive lambda
uses the existing Torch float64 LBFGS solver, not a new optimizer dependency.
The initial normalized negative Hessian whitens its coordinates. Up to3 native
Newton corrections address floating-point line-search stagnation near the
optimum; they consume the same solver iteration budget and may not decrease Q
beyond1e-7. The original-coordinate gradient certificate remains unchanged.
Acceptance requires normalized-gradient stationarity and no Q decrease; the
subsequent EM owner must independently check the penalized marginal objective.
These are numerical checks, not claims of statistical recovery.

The approach is inspired by difference-penalized density smoothing in
Eilers and Marx,1996,
[Flexible Smoothing with B-splines and Penalties](https://sites.stat.washington.edu/courses/stat527/s13/readings/EilersMarx_StatSci_1996.pdf).
This experiment uses fixed constant-density bins, not their complete spline
estimator; its finite-sample HSMM performance must be measured independently.

The JSON contract freezes fresh seeds1912/1913/1914,1024 training sequences,
24 outer sequences, warm3 plus27 updates per arm, lambda0 versus log1024.
Both arms start from the same fixed-grid warm PMF. There is no hyperparameter
selection, oracle support, label-based fitting or post-outer adjustment.

Implementation gate: verify the solver's gradient/Hessian, analytic lambda0,
unequal widths, full-support normalization, constant-density block semantics,
stationarity failure and a real short censored EM/public numerical smoke before
building/running the checkpointed1024-sequence driver. Existing production and
historical experiment owners stay unchanged.

Numerical refinement is frozen before any fresh experiment outcome. The first
short smoke exposed badly scaled joint L-BFGS (1000 iterations without the
required certificate); whitening reached a near-stationary point but its
4e-8 residual still failed the declared1e-8 gate. The bounded native correction
passes the original gate. No lambda, width, data budget or recovery threshold
was changed to make the smoke pass.

## Step32-1912-checkpoint-unsmoothed

Smoothing experiment seed1912 checkpoint: unsmoothed at0 updates.
All19 source/software/contract/training/phase/solver/history/PMF/partition and
shared-warm continuity checks pass. Exact saved snapshot is recovery evidence;
outer/final independent/numerical/recovery gates remain pending. Source140ff55,
1024 training sequences, fixed33-bin full-age support, oracle OFF.

## Step32-1912-checkpoint-smoothed

Smoothing experiment seed1912 checkpoint: smoothed at0 updates.
All19 source/software/contract/training/phase/solver/history/PMF/partition and
shared-warm continuity checks pass. Exact saved snapshot is recovery evidence;
outer/final independent/numerical/recovery gates remain pending. Source140ff55,
1024 training sequences, fixed33-bin full-age support, oracle OFF.

## Step32-1912-checkpoint-smoothed-15

Smoothing experiment seed1912 checkpoint: smoothed at15 updates.
All19 source/software/contract/training/phase/solver/history/PMF/partition and
shared-warm continuity checks pass. Exact saved snapshot is recovery evidence;
outer/final independent/numerical/recovery gates remain pending. Source140ff55,
1024 training sequences, fixed33-bin full-age support, oracle OFF.

## Step32-1912-complete

Smoothing experiment seed1912 completes57 actual M-steps: warm3 and27
per matched fixed-grid arm. All19 source/software/contract/training/outer
identities, phase/budget/solver certificate/objective/PMF/grid checks, independent
terminal E-step and outer/public likelihood/gradient/filtering replays and
byte-preserving completed no-op pass. Both arms share initial PMF, grid and
observations. Smoothed minus unsmoothed outer LL0.2281040401685459 nats.
Main report SHA256bb78904407ad2421854d484052a471daae62cb6ef3e17fc0706dfad0a87410d0. Oracle OFF, source140ff55 and
all declared widths/strengths/budgets/gates unchanged. No production/joint-neural
or legacy NPZ acceptance follows.
unsmoothed: TVs[0.07355676669601013, 0.08641623132355288, 0.09008035408903947, 0.06766988698701032], gain0.007734397346692074 nats/row, full gatePASS, fixed-partition TV lower bounds[0.02422778085700007, 0.025163683254851587, 0.05228327082775793, 0.05424085851097055].
smoothed: TVs[0.04234832628750705, 0.06414683287189094, 0.07021739228445997, 0.07326639128710745], gain0.0077660784633821495 nats/row, full gatePASS, fixed-partition TV lower bounds[0.02422778085700007, 0.025163683254851587, 0.05228327082775793, 0.05424085851097055].

## Step32-1913-checkpoint-unsmoothed-27

Smoothing experiment seed1913 checkpoint: unsmoothed at27 updates.
All19 source/software/contract/training/phase/solver/history/PMF/partition and
shared-warm continuity checks pass. Exact saved snapshot is recovery evidence;
outer/final independent/numerical/recovery gates remain pending. Source140ff55,
1024 training sequences, fixed33-bin full-age support, oracle OFF.

## Step32-1913-checkpoint-smoothed-15

Smoothing experiment seed1913 checkpoint: smoothed at15 updates.
All19 source/software/contract/training/phase/solver/history/PMF/partition and
shared-warm continuity checks pass. Exact saved snapshot is recovery evidence;
outer/final independent/numerical/recovery gates remain pending. Source140ff55,
1024 training sequences, fixed33-bin full-age support, oracle OFF.
