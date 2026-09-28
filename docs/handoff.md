# NHSMM handoff

Operational continuation point for the next chat. `docs/state.md` remains the canonical package-status owner.

## Repository

- repository: `awa-si/nhsmm`
- branch: `develop`
- snapshot date: `2026-09-28`
- snapshot head before this handoff update: `7fa2462a12fb909c58ef8e4e3fd696843ffe2db7`

## Current state

Package-core mechanism research, normalized universal validation API, stress validation, packaging dry-run, GitHub metadata cleanup, full local regression, and the first repository code-scan/hardening pass are complete.

Do not reopen accepted detector tuning or change probabilistic semantics without new evidence.

### Package-core evidence

- duration context: PASS
- latent-state recovery: PASS
- transition context: PASS
- multi-component identifiability: PACKAGE-CORE PASS
- general-context robustness: PACKAGE-CORE PASS using split-fit direction replication

### Canonical public validation API

```python
context_effect(model, context, component=...)
evaluate_context_effect_replication(
    model_a,
    model_b,
    contexts,
    component=...,
    config=...,
)
split_fit_context_effect_evidence(
    observations,
    context,
    component=...,
    fit_model=...,
    evidence_contexts=...,
    config=...,
)
```

`component` is one of:

```text
initial | duration | emission | transition
```

Validation remains separate from `NHSMM.optimize()` and remains domain-neutral.

## Verification status

Normalized validation/API focused regression:

```text
23/23 PASS
```

Public API stress:

```text
STRESS_PASS
effects=4800
replications=160
optimize_splitfits=16
runtime=22.421 s
Python-tracked peak memory=53.47 MiB
```

Full repository pre-CI regression after the latest context/device hardening:

```text
99 passed, 3 skipped
```

The three skips are CUDA-only tests on a CPU runtime. No failing local test remains in that verified workspace.

Earlier GitHub smoke CI before the code-scan hardening was green:

```text
run: 36392227850
pytest: 96 passed
runtime benchmark: PASS
```

Do not claim CI coverage of the later code-scan hardening until a new CI run is actually executed.

## Packaging / PyPI

Packaging dry-run is verified:

```text
run: 36359725337
artifact: nhsmm-release-dry-run-distributions-36359725337
sha256: dd741ea982c7679d07357d2dd475516904b5acfeeeb85415b03fb0c8397e130f
```

Verified gates:

- `python -m build`
- `twine check dist/*`
- clean-venv wheel installation
- installed-package public API smoke
- artifact upload

Canonical release workflow:

```text
.github/workflows/release.yml
```

PyPI Trusted Publishing still requires one-time external setup for:

```text
owner: awa-si
repository: nhsmm
workflow: release.yml
environment: pypi
```

No production release tag was created.

## GitHub/repository hygiene

GitHub-specific files have been reviewed and normalized for a developer-oriented public repository:

- contributor guide
- PR template
- bug issue form
- security policy
- code of conduct
- Dependabot for pip + GitHub Actions
- manual smoke/package-validation workflows
- tag-only publish path in release workflow

README/documentation language was also reduced to technical scope/status rather than product claims.

## Latest code-scan hardening already applied

The repository code scan is recorded in:

```text
docs/validation/code-scan-2026-09-28.md
```

Safe behavior-preserving fixes already committed:

- ContextEncoder/ContextRouter/SequenceSet device placement hardened
- mask/index/padding helper tensors follow the active input device
- lazy attention/MHA modules are created on the active device/dtype
- temporary pooling overrides are restored on exceptions
- unused encoder logger import removed
- `tests/test_context_hardening.py` added

The corresponding local full suite is `99 passed, 3 skipped`.

## IMPORTANT: requested next change is NOT done yet

The user explicitly requested:

```text
Bump Python to 3.12. Correct bugs.
```

That work was interrupted by this handoff request. Do **not** assume it has been applied.

Current package metadata still declares Python `>=3.9` and old Python classifiers. The next chat should deliberately move the supported baseline to Python 3.12 and align packaging/tool configuration/workflows/docs accordingly.

## Open code-scan findings to fix next

Priority order:

### P0 — Python compatibility metadata

Current source already uses syntax such as `A | B`, while `pyproject.toml` still claims Python `>=3.9`.

User decision: **raise the project baseline to Python 3.12** rather than backport syntax.

Next chat should update at least:

- `project.requires-python` -> `>=3.12`
- Python classifiers -> remove 3.9/3.10/3.11; retain/add supported modern versions as deliberately chosen
- Black/Ruff/mypy target versions -> Python 3.12
- GitHub workflows -> use Python 3.12 consistently unless a deliberate matrix is introduced
- README/contributor/release docs if they state a Python version

Then rebuild/install-test the wheel under Python 3.12.

### P0 — `SequenceDataset(variable_length=True)` contract bug

`generate_gaussian_sequence()` returns a single observation tensor. `SequenceDataset` wraps `X` and optional `C` as sequence lists for `variable_length=True`, but leaves `states` as one tensor. `__getitem__()` therefore returns a scalar state where `collate_fn()` expects a state sequence for `pad_sequence()`.

Fix the data contract, add focused tests, and verify both context/no-context variable-length paths.

Also review whether `nhsmm/data.py` should remain package-core at all; it still contains a Freqtrade-specific `load_dataframe()` helper and is the apparent reason `polars` is a runtime dependency. Do not remove or relocate it silently without checking public/import consumers.

### P0 — lazy `attn` / `mha` training lifecycle

`ContextEncoder.reset()` clears lazy attention/MHA modules. During optimize/restarts this can cause parameters to be recreated after optimizer construction, leaving those recreated parameters outside the optimizer and restart snapshots.

Correct the lifecycle so trainable pooling parameters/modules are registered before optimizer construction and survive/reset only runtime caches, not model parameters. Add tests proving optimizer membership and restart/state-dict consistency for `pool="attn"` and `pool="mha"`.

### P1 — 2-D context contract

`ContextRouter` supports both `[T,H]` and `[B,H]`, while the normal NHSMM `_build_sequence_set()` 2-D context path effectively assumes `[T,H]` and broadcasts it across batch. Decide and enforce one unambiguous public contract with tests; avoid silent ambiguous broadcasting.

### P1 — config validation

Add early `ModelConfig` validation for invalid values such as zero/negative `n_states`, `n_features`, `max_duration`, `n_init`, `max_iter`, invalid learning rates/tolerances, etc., rather than failing later in training internals.

### Lower-priority scan findings

Review after P0/P1:

- transition refinement should explicitly handle missing context
- base `DistributionSet` constructor accepts injectable classes but currently instantiates canonical classes directly
- `build_artifact()` calls inference preparation and can alter model mode; decide whether artifact creation should be side-effect-free
- ContextEncoder caches may retain autograd graphs longer than necessary
- `emission_init_mode="randome"` is an old typo retained as compatibility behavior
- a small number of broad exceptions/assert-based runtime checks remain
- dead imports in `distributions/default.py` were observed previously; only remove after current-state proof

## Next-chat execution order

1. Read `agent.md`, `docs/state.md`, this handoff, and `docs/validation/code-scan-2026-09-28.md`.
2. Resolve current `develop` head before editing.
3. Bump Python baseline to **3.12** across package metadata/tooling/workflows/docs.
4. Fix `SequenceDataset(variable_length=True)` with focused tests.
5. Fix lazy `attn`/`mha` lifecycle with optimizer/state/restart tests.
6. Resolve the 2-D context contract and add tests.
7. Add `ModelConfig` validation.
8. Run focused tests.
9. Run full local pytest; require green before CI.
10. If local green, run the manual smoke/package validation CI as appropriate.
11. Update `docs/state.md`, `docs/handoff.md`, and code-scan status after verification.
12. Only after package bugs are closed return to PyPI release/versioning or downstream Nautilus evaluation.

## Semantic constraints to preserve

- causal boundary `t-1 -> t` uses only information from `F_{t-1}`; `x_t` enters after propagation through emission
- explicit-duration HSMM semantics must remain intact
- duration evidence includes `duration_logits_bias`
- validation remains separate from optimization policy
- state alignment uses learned emission centers, never latent truth
- no Nautilus/trading assumptions in NHSMM core validation
- no retuning of frozen detector thresholds on confirmatory data

## Source-of-truth files

Read first:

- `agent.md`
- `docs/state.md`
- `docs/handoff.md`
- `docs/validation/code-scan-2026-09-28.md`
- `docs/release.md`
- `docs/testing.md`
- `docs/validation-api.md`
- `docs/validation/public-api-stress-2026-09-28.md`
