# V3 historical identity cleanup and final validation — 2026-10-15

Context/config source checkpoint: d80e09c6c9d4a96d024dd93202af43d09e2da2ef.
Archive anchor: fbfc4f9e508ca2ae3bf59e2effb105faa308f6df.

## Root cause and correction

All five prior exclusions were reproduced before changes. Archived evidence
records Python 3.12.14 while the current interpreter is 3.14.7 (NumPy 2.5.4,
Torch 2.14.1+cpu). A metadata-only smoothing audit constructed expected execution
identity from the current interpreter. Two optimizer tampering tests invoked
the strict numerical verifier, whose correct software guard blocked them before
their intended historical checks. None of these failures proves a model drift.

A new historical-evidence index binds original schema, source/software identity,
contract and SHA256 report/proof/batch bytes to the existing Git history. All nine
original Git blob identities were independently checked through the connected
GitHub connector before indexing. No existing frozen manifest, report, proof,
source identity or recorded result was edited.

The smoothing metadata audit now uses recorded identity, retains its arithmetic,
budget/certificate checks and additionally verifies original report/proof/batch
bytes. The optimizer verifier adds an explicit audit API/--audit-only mode for
registered historical bytes/certificates. Its result explicitly says that no
independent terminal/outer/public numerical replay was performed. Default verify
still requires matching software and retains all numerical replay checks.

The two optimizer tests asserting historical source/initial-objective rejection
now call the operation that owns that contract. New regressions demonstrate no
sequence generation/public/gradient replay during metadata audits, rejection of
recorded-software tampering, continued strict replay software rejection, rejection
of resealed/unregistered report changes and rejection of altered proof-file bytes.
Archive audit output must be a new file; old artifacts cannot be overwritten
through the new CLI mode.

The index is a trusted repository record, not a digital signature or proof that
an arbitrary external source is authentic. Byte/certificate audits establish
preservation and recorded consistency, not a fresh statistical/model acceptance.

## Observed validation

- Before correction: all five exclusions reproduce their identity/software errors.
- Historical focused cases: 23 passed in 7.10s.
- Entire core suite without --deselect or skips added for historical cases:
  1242 passed, 3 existing CUDA-only skips, zero deselections, 204.26s.
- Consumer scope (all test_qlib*, test_nhsmm*, test_gbm_qlib_bridge.py):
  84 passed, one isolated no-GBM-profile skip, 58.11s.
- Installed snapshot: 15 new config/context regressions pass, 9.22s; pip check clean.
- Both archive CLIs run successfully; all nine original files retain their SHA256.
- The only source change after the broad run is clarification of the smoothing
  CLI help text. AST/Ruff/Black, CLI --help and the 23 historical regressions are
  rechecked on the final source; no model/test algorithm changes follow the run.
- Eight pre/post-refactor positive-length training/inference cases remain exact
  with zero tolerance. The empty-context canonical shape repair is intentional
  and separately tested. No performance or improved holdout claim.

The consumer run emits the existing NumPy/pandas timedelta and controlled
self-produced trusted-pickle warnings (102); no warning was suppressed.
All source/install copies and Git trees are checked before push.

Logs:
- /work/checkpoints/v3-historical-baseline-20261015.txt
- /work/checkpoints/v3-historical-focused-20261015.txt
- /work/checkpoints/v3-all-no-exclusions-20261015.txt and .xml
- /work/checkpoints/v3-cc-host-20261015.txt
- /work/checkpoints/v3-cc-installed-20261015.txt
- /work/checkpoints/v3-smoothing-archive-audit-20261015.json
- /work/checkpoints/v3-optimizer-archive-audit-20261015.json
- /work/checkpoints/v3-historical-focused-final-20261015.txt

Oracle OFF. Frozen holdout/artifacts unchanged; metadata audits add zero fitting
updates and no new holdout evaluation. Next research is a separately preregistered
matched-actual-update, multi-seed train/validation study with a fresh outer holdout
reserved. This code/test cleanup does not establish market quality or live use.
