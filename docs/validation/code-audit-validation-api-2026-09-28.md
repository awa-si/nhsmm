# Validation API code audit — 2026-09-28

Status: **PASS / behavior preserved**.

## Scope

Behavior-preserving audit of the universal validation surface introduced for split-fit context evidence:

- `nhsmm/validation/alignment.py`
- `nhsmm/validation/context_evidence.py`
- `nhsmm/validation/split_fit.py`
- `nhsmm/validation/nhsmm.py`
- `nhsmm/validation/__init__.py`
- root public exports in `nhsmm/__init__.py`
- semantic validation tests and public API documentation

The audit explicitly excluded model-semantic, threshold, training, likelihood, topology, and detector-policy changes.

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

One dead import was found and removed:

- unused `Optional` import in `nhsmm/validation/context_evidence.py`.

This is a source-only cleanup and does not alter runtime behavior, API signatures, numerical output, exceptions, selection policy, or serialization.

No additional behavior-neutral refactor was justified. In particular, duplicated callable signatures and small helper boundaries were left unchanged because consolidating them would add churn without a correctness benefit.

## Verification

- validation package compiles successfully after the cleanup;
- focused pytest collection in the sparse local workspace remains blocked by the known missing regular package modules imported from `nhsmm.__init__` (`nhsmm.inference` etc.);
- this is the pre-existing sparse-workspace limitation, not a regression introduced by the audit;
- no CI was used.

## Conclusion

**PASS.** The audited validation API is internally coherent and behavior is preserved. The only code change is removal of an unused import.
