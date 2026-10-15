# V3 config/context checkpoint — 2026-10-15

Base: fbfc4f9e508ca2ae3bf59e2effb105faa308f6df, branch v3.

## Change

The model now prepares observations/masks once for SequenceSet construction and
passes the validated mask to external-context alignment. Ragged list alignment
has one context-module owner, with the same finite-value, feature-width, length
and public broadcast checks. The existing model helper remains the established
boundary for its consumers. No context timing shift or new ambiguous shape
coercion is introduced.

A reproduced empty-input bug had canonical shape (2,0,3) for empty observations
with external context. The canonical shape is now [B,1,H] with zero context;
empty likelihood remains neutral. This fix is intentional; equivalence claims
apply to previously valid positive-length cases.

ModelConfig mapping parsing shares the existing dataclass parser. Strict unknown
fields remain errors, explicit strict=False still filters unknown fields, mixed
unknown key types produce a clear validation error, and non-string/unhashable
choice values raise ValueError instead of incidental hashing TypeError. Default
values and serialized/artifact/state_dict schemas remain unchanged.

## Verification

- AST/Ruff/Black/diff checks pass.
- 84 focused cases pass, 3 CUDA-only skips, 20.18s.
- Full applicable suite: 1232 passed, 3 CUDA-only skips, 5 explicit historical exclusions in 220.41s..
- Eight pre/post-refactor training/inference replays remain exact with tolerance
  zero for parameters, Adam/convergence state, scores, histories, likelihood and
  RNG. No general performance claim follows from this structural refactor.
- Fifteen new cases cover Float64 ragged/padded likelihood and context gradients,
  zero padding gradients, empty batch/time contexts in all supported forms,
  strict/permissive parsing and invalid non-string choices.
- Initial focused command referenced two nonexistent historical test filenames;
  that invocation ran no tests. It was corrected before reporting the actual
  passing focused and full runs.

The same five historical software/execution-identity exclusions remain explicit
for this isolated config/context checkpoint. They are being handled in the next
separate batch rather than hidden by modifying archived evidence:
test_block_mass_smoothing_audit.py: completed_batch, raw_bytes_tampering,
missing_replay_certificate; test_optimizer_bridge_verification.py:
changed_source_manifest_rejected and rebound_initial_objective_rejected.

Logs: /work/checkpoints/v3-config-context-focused-20261015.txt,
v3-config-context-full-20261015.txt and .xml, and
v3-config-context-parity-20261015.json. Actual source and Git trees are checked
before commit/push. Oracle OFF; frozen holdout/artifacts untouched. No new
market-data/holdout fit, statistical-quality improvement or GPU acceptance.
