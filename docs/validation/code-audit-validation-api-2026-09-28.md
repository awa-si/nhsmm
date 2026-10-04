# Validation API code audit — 2026-09-28

> Historical note: `nhsmm/models/training.py` referenced below was later consolidated into the canonical `nhsmm/models/base.py` implementation and removed.

Status: **PASS / behavior preserved**.

## Scope

Behavior-preserving audit of the universal validation surface introduced for split-fit context evidence and its immediate package-adjacent surfaces:

- `nhsmm/validation/alignment.py`
- `nhsmm/validation/context_evidence.py`
- `nhsmm/validation/split_fit.py`
- `nhsmm/validation/nhsmm.py`
- `nhsmm/validation/__init__.py`
- root public exports in `nhsmm/__init__.py`
- semantic validation tests and public API documentation
- adjacent package hygiene in `nhsmm/config.py` and `nhsmm/models/training.py`
- static review of imports/exception boundaries in `models/base.py`, `runtime.py`, `inference.py`, `artifact.py`, `convergence.py`, `context.py`, and `distributions/default.py`

The audit explicitly excluded model-semantic, threshold, training, likelihood, topology, detector-policy, numerical, and serialization changes.

## Findings

### Semantics

No semantic drift found in:

- emission-center state alignment;
- state permutation handling;
- context-effect vector construction;
- replicated direction correlation;
- replicated minimum amplitude selection;
- split-fit disjointness;
- duration-weighted transition integration;
- `duration_logits_bias` use;
- train/eval mode restoration;
- public root exports versus `nhsmm.validation` exports.

The accepted detector policy remains external and explicit through `ContextEvidenceConfig`; no research threshold was promoted into model-core behavior.

### Code quality

Behavior-neutral dead imports removed:

- unused `Optional` import in `nhsmm/validation/context_evidence.py`;
- unused `Dict`, `Any`, and `torch.nn` imports in `nhsmm/config.py`;
- unused `Any` and `logger` imports in `nhsmm/models/training.py`.

These are source-only cleanups. They do not alter runtime behavior, API signatures, type contracts, numerical output, exception behavior, training, selection policy, model state, or serialization.

The second static pass also checked broad exception boundaries and adjacent imports. Existing broad catches in artifact deserialization, distribution initialization, convergence scheduler introspection, and encoder-signature fallback were left unchanged because narrowing them could change failure behavior. Active logging in `models/base.py` was verified and retained.

No additional behavior-neutral refactor was justified. In particular, duplicated callable signatures and small helper boundaries were left unchanged because consolidating them would add churn without a correctness benefit.

## Verification

- validation package compiles successfully after the cleanup;
- focused semantic/API validation had previously passed `13/13` before this import-only audit pass;
- the import-only follow-up does not modify executable branches;
- full pytest materialization remains constrained by connector-only repository transport in the sparse local workspace; no binary repository checkout path is used;
- no CI was used.

## Conclusion

**PASS.** The audited validation/API surface and immediate package-adjacent code are internally coherent and behavior is preserved. Changes are limited to verified dead-import removal.
