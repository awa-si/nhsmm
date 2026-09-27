# General-context detector rapid search — 2026-09-27

Status: **OPEN / no accepted detector yet**.

This record continues the package-core general-context detector search after the canonical detect→refine gate remained PARTIAL on moderate 2D context and the shared-linear + effect-floor candidate failed confirmatory selection power.

All work remained local-only. No CI was used. Previously rejected thresholds were not retuned after seeing confirmatory blocks.

## Candidate 1 — fold-stability OR effect floor

Rule frozen from dev seeds `761..765`:

- select if at least `4/5` held-out folds have positive evidence **OR** pre-refine context amplitude `>=0.25`;
- `0.25` was inherited from the earlier frozen effect-floor candidate rather than retuned on the confirmatory set.

Fresh confirmatory seeds `771..775`:

- strong: `5/5` selected;
- moderate: `3/5` selected;
- null: `0/5` selected.

**Result: REJECTED.** Moderate selection power remained below the required `>=4/5`.

## Candidate 2 — single-permutation evidence

Per held-out fold, canonical context was compared with one deterministic permuted context. Selection required at least `4/5` folds with `ΔLL(true - permuted) > 0`.

Development seeds `781..785`:

- strong: `5/5` selected;
- moderate: `5/5` selected;
- null: `3/5` selected.

**Result: REJECTED on development.** Temporal context assignment alone did not control null false positives.

## Candidate 3 — permutation evidence AND effect floor

Rule frozen from the same development block:

- permutation-positive folds `>=4/5`;
- pre-refine context amplitude `>=0.25`;
- both conditions required.

Fresh confirmatory seeds `791..795` failed early because moderate selected only `2/5`. Null completion was intentionally stopped once the candidate had already failed the frozen moderate-power requirement.

**Result: REJECTED.**

## Candidate 4 — permutation-rank exact test

To reduce single-permutation variance, each of 5 held-out folds compared true context against all 7 nontrivial cyclic batch permutations, yielding 35 paired wins/losses. Selection used a one-sided exact Binomial test against `p=0.5` at `alpha=0.05`; no effect-size threshold was added.

Development seeds `801..805`:

- strong: `5/5` selected;
- moderate: `5/5` selected;
- null: `0/5` selected.

Fresh confirmatory seeds `811..815`:

- strong: `5/5` selected;
- moderate: `5/5` selected, median transition corr `0.890`, median MAE `0.082`;
- null: `1/5` false-positive.

The false-positive was null seed `813` with exact binomial `p=0.0083`.

**Result: REJECTED.** Do not tighten `alpha` after observing this confirmatory block.

## Current boundary

The best rejected detector so far is the permutation-rank test: it restored full moderate power and strong recovery but missed the strict null requirement by one seed. The remaining problem is therefore not estimator expressiveness and not moderate power in isolation; it is replication-calibrated null control.

Next candidate: **split-fit direction replication**. Two independent fits on disjoint training halves must recover the same context-delta direction after state alignment. This adds an independent reproducibility requirement rather than another magnitude threshold.

- development seeds: `821..825`;
- if viable, freeze once on that block;
- fresh confirmatory seeds: `831..835`;
- no result has been accepted yet.
