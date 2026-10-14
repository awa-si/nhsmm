# Fixed-bin block-mass smoothing

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
