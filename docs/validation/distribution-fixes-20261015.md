# Distribution corrections — 2026-10-15

Base: `ec81825` on NHSMM `regime-v2`. Follow-up to the immutable distribution audit.

## Applied

- D08: Gaussian/Student-t locations retain finite-value checks but no longer inherit categorical logit clipping. Large joint translations and their gradients are tested.
- D01: safe exact suffix survival and all-impossible logsumexp reductions avoid NaN backward paths. Dense positive duration scores retain vectorized cumulative survival.
- D02/D05/D06: zero-mass categorical probability gradients, constructor validation order and label broadcasting corrected. Partial negative-infinity logit support is accepted; NaN,+inf and all-impossible rows remain rejected.
- D03: Student-t parameters broadcast before moment, entropy and event-dimension calculations.
- D04: Initial,Duration,Transition use exact negative-infinity hard support, rather than a finite sentinel.
- Coupled diagnostic consumer: censored-duration objectives select positive counts before multiplying log scores; zero counts outside support contribute exactly zero.

Twenty-four regression cases cover the reproduced defects. Existing independent public-marginal and posterior-score gradient tests cover sparse model recursion.

## Verification

| Gate | Result |
| --- | --- |
| Focused smoke | 85 passed |
| Sparse gradient/consumer follow-up | 36 passed |
| Applicable full NHSMM suite | 1066 passed;3 CUDA skips;5 explicit historical deselections |
| Nautilus/Qlib integration with changed core | 49 passed;1 isolated no-GBM profile skip |
| Frozen artifact fixed64-row probe | Posterior difference exactly0.0 against verified pre-fix source |
| AST/Ruff checks | Syntax and core F checks pass;new test file full Ruff passes |

The first unrestricted run found four sparse-support consumer/recursion failures in addition to five historical provenance tests. All four sparse-support failures pass after the coupled correction. The same five historical failures were independently reproduced on the unchanged pre-fix snapshot; thirteen other tests in those historical files pass.

The five exceptions remain explicit command-line deselections, not pytest configuration exclusions or weakened assertions. Current software identity differs from frozen report metadata, so provenance gates fail before some tamper-specific expected messages. Frozen evidence and guards are not modified. **The unrestricted suite is not claimed fully green.** Exact test IDs and outputs are recorded in the JSON evidence.

## Preserved / still open

Artifact schema,parameter names,initialization,D+ tail,self-renewal semantics and optimizer settings remain unchanged. No new research fit,holdout tuning or production promotion is part of this correction.

D07 remains open: generic Duration forward/expected_probs do not honor soft_dmax like log_matrix does. This needs a deliberate API decision. Age-versus-regime-age interpretation and the cause of oversegmentation remain separate research questions.

The original audit remains historical evidence of the pre-fix source. Its seven confirmed defects are addressed by this checkpoint; this document does not erase the audit or retrospectively alter frozen results.
