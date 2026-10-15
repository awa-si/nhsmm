# Frozen Trading artifact: zero-update gradient-flow protocol

Preregistered 2026-10-15 before the material diagnostic. Core baseline:
95bc4b17995d43155db7458a7fcb156ede3534ff (regime-v2).
Runner: scripts/diagnose_trading_gradient_flow.py.

Question: are duration PMF, soft gate and D+ tail connected to the production
training objective at the existing ten-update artifact? How do their raw gradient
norms compare with emission/chain gradients before the global norm cap of five?

Use only the existing baseline protocol's twelve training and eight validation
windows, width480, original five features and frozen training-only scaler.
Dataset SHA256370e6e9e87e52d779ea208111eeb2e03b0bf21f8b7e88859fcc8c19ac2739fb4;
artifact SHA256925bc1d35721d031d5385e9fa52ffc381eb789e1abfe225d18468878f6b2130b.
Never extract labels or test rows. Oracle OFF.

No optimizer steps, no new fitting, no candidate selection. Two gradient passes:
target temperature1 and the historical schedule's last pre-update temperature,
derived from the saved max_iter10 configuration. The latter is a local probe at
the terminal weights, not a reconstruction of update10. Accumulate gradients
one training window at a time using the production training-loss helper, applying
the saved bias penalty once. Report finite/nonzero gradients by named parameter,
group L2 norms and fraction of squared global norm, plus the global clipping
factor and resulting group norm. Score observation likelihood per row on train
and validation at the same two temperatures. Preserve and compare all model state.

The short smoke uses one32-row window per split, followed by the unchanged
material protocol. Static checks must pass first. Stop on identity/bounds/finite
gradient/state preservation failures. No minimum gradient magnitude is selected
after results. Global clipping affects all raw gradients by the same factor;
Adam's per-coordinate moments prevent interpreting raw gradient proportions as
parameter-update proportions. The artifact contains neither optimizer moments
nor an update trace, so it cannot prove historical convergence or exact updates.

Existing budget pilots were inspected but use corrected episode-end fixtures,
different feature/context settings and test-scoring routines. They cannot be
silently reused for this v1 episode-start artifact. Reuse the existing duration
law/mean and production objective helpers instead. No core estimator changes.

Follow-up, outside this stage: design a matched finite-budget learning study
using fixed objective temperature, observed update counts, train/validation-only
checkpoint diagnostics and independent seed replication. Existing final-test
scores stay frozen.
