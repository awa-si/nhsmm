# Stage 2V.7 — Unsupervised age-tied transition pilot

The initial attempt to attach the age-sharing gradient hook before `NHSMM.optimize()` was invalid: `_initialize_run_state()` replaces `model.dist` with a fresh `DistributionSet` on each restart, so the hook was attached to an obsolete parameter. The corrected harness installs the hook **after fresh-run initialization**, without changing production NHSMM code. The final age-invariance check confirms the age-tied transition probability rows are identical (`max deviation 0`). This is a harness integration correction, **not an NHSMM production bug**.

## Experimental scope

Dataset `trading-regimes-v1`; train 12 windows × 480 bars drawn from the first 60%; OOS test 4 windows × 480 bars from the last 20%; seed 901; 12 `optimize()` iterations, no transition context. Ground truth is used **only for metrics**, not for optimization. Both models use the same initial seed, configurations, and window split. The no-context transition component is either left fully age-dependent or gradient-tied across all 64 ages. This is a deliberately bounded diagnostic, not the canonical full 3-seed recovery acceptance.

| OOS metric | Free age-dependent | Age tied |
| --- | ---: | ---: |
| Log-likelihood per bar (higher better) | -10.248358 | -10.249608 |
| Matched state accuracy | 0.499479 | 0.498958 |
| ARI | 0.130756 | 0.130836 |
| Boundary F1 ±5 bars | 0.172249 | 0.175610 |
| Max age-row probability difference | 0.062569 | 0 |

## Interpretation

Age tying dramatically improved directly supervised transition probability generalization in Stage 2V.4, but **does not materially improve unsupervised regime identification or out-of-sample likelihood in this pilot**. Ground-truth boundary fitting and fully latent estimation answer different questions. Weak observation separability and training convergence remain unresolved. No production defaults or mathematical routines are changed. A fuller independent-seed comparison would be required before making a default decision.

Reproduce with `PYTHONPATH=/workspace /work/env/nhsmm/bin/python scripts/diagnose_trading_unsupervised_age_tie_pilot.py`.
