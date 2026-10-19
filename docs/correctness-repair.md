# Mathematical correctness repair

The 2026-10-19 independent audit of e28db29 found five reproducible defects. This repair normalizes noncausal public observation likelihoods, restores first-order tail gradients at h_tail=1, handles segment emission support zeros, rejects loss of reachable duration support, and preserves artifact precision.

The normalized noncausal contract is `W(X;C)/Z_T(C)` over complete segmentations, with fixed C. The partition is differentiated, not detached. Raw forward and MAP remain segment-potential interfaces. This is an intentional public score change, recorded by artifact v3 `schema.likelihood_semantics`. Old causal v2 and encoderless causal v1 artifacts retain their score interpretation. Old noncausal artifacts require retraining; the loader rejects silent objective migration. Raw state dictionaries have no objective provenance, which callers must establish.

For the exact tail boundary, log-domain forward/backward messages supply the derivative contribution from a single formerly zero-weight continuation. No approximate h, test-tolerance change, new trainable parameter or extra training attempt is used. Boundary score derivatives are first-order only; requesting a higher-order graph raises explicitly. Raw hard-zero log tensors are not the differentiable likelihood API.

Forward, decoding, segment emissions, objectives and boundary score derivatives have separate owners under `nhsmm/models/`. Public class methods retain their signatures and state-dict paths.

Canonical regression file: Nautilus `tests/ml_stack/nhsmm/test_ground_truth_fixes.py`. It independently enumerates probability products, including paths with zero value but nonzero derivative. Existing raw-forward oracle assertions remain; public likelihood assertions now additionally divide by an independently enumerated partition. Numerical tolerances are unchanged.

Validation results and source identities are retained by the host repair evidence. The prior training reference remains unchanged and is not promoted or retrained by this repair. The observed baseline state collapse has no newly established cause.

## Verification at the repaired source

- Full canonical suite: **1156 passed, 3 skipped** (CUDA unavailable), 74.20 s;
  execution `41480b3dc6214848a25a8ca7e3aa900f`.
- **59 new regression cases** protect normalization, support, dtype and exact
  first-order tail-boundary derivatives, including ragged sequences.
- Independent rational path enumeration: **540 forward/MAP configurations**,
  maximum absolute evidence residual `2.220446049250313e-16`.
- Ruff and Black pass. The model class is 923 lines; 22 Python source files
  retain the original parameter paths and separate inference responsibilities.
- The unchanged causal v2 reference artifact loads and gives exactly the same
  score before/after on a fixed synthetic prefix. No real training was run.
- Short CPU A/B (K=3, F=2, D=32, B=1, T=256, three repetitions): median forward
  0.1754 -> 0.1981 s, backward 0.1919 -> 0.1635 s. This bounded sample supports
  no general speedup claim. Normalizing the noncausal objective requires a
  numerator and partition pass; that additional work is required for correctness.

The host retains exact source fingerprints and execution identities in
`docs/nhsmm-correctness-repair.json`.
