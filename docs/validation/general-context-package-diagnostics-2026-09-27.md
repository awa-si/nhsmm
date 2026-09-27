# General-context package diagnostics after partial gate — 2026-09-27

Base package-core result: strong PASS, null PASS, moderate FAIL only on median transition-function correlation (`0.730 < 0.800`) under the frozen detect→refine gate.

All work was local-only. No CI and no frozen gate retuning.

## Diagnostics

1. **Pre vs post refinement**
   - moderate median corr before refinement: ~`0.754`
   - after 20 transition-context-only refinement steps: ~`0.730`
   - conclusion: refinement is not the primary cause; the fidelity gap already exists before refinement.

2. **No-self transition topology**
   - moderate selection degraded to `3/5`, median corr ~`0.600`
   - rejected.

3. **Shared-duration context modulation**
   - moderate corr ~`0.644`, MAE ~`0.134`
   - rejected.

4. **Reference-equivalent shared-linear estimator**
   - strong: selected `5/5`, median corr ~`0.985`, MAE ~`0.039`
   - moderate: selected `5/5`, median corr ~`0.877`, MAE ~`0.084`
   - null: selected `5/5` when used as its own detector
   - conclusion: excellent active-signal estimator, invalid null detector.

5. **Dual-stage candidate: canonical MLP detector → shared-linear estimator**
   - Dev seeds `741..745`, 15/15 cases completed.
   - strong: selected `5/5`, median corr `0.984`, MAE `0.044`, 5/5 per-seed recovery pass.
   - moderate: selected `5/5`, median corr `0.923`, MAE `0.076`, 5/5 per-seed recovery pass.
   - null: selected `3/5`, median context amplitude `0.169`, median MAE `0.038`, only 2/5 null per-seed passes.
   - **REJECTED**: active fidelity is solved, but detector null-generalization fails on new dev seeds.

## Current interpretation

The remaining package deficiency is not estimator expressiveness: a shared-linear estimator can recover strong and moderate 2D transition functions accurately. The unresolved issue is robust evidence-based detection/generalization of weak multidimensional context under null truth. The canonical MLP detector succeeded on seeds `731..735` (`0/5` null selected) but failed to generalize on new dev seeds `741..745` (`3/5` null selected).

Do not promote the dual-stage architecture. Next work must improve detector stability without weakening null control or changing the frozen reference gate after observing these results.
