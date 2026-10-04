# NHSMM State

Current package readiness and next package-level boundary. Detailed semantics: [`model.md`](model.md). Controlled evidence: [`package-validation.md`](package-validation.md). Verification policy: [`testing.md`](testing.md). Release procedure: [`release.md`](release.md).

## Current status

**Phase:** package-core mechanism research, context-effect validation API consolidation, packaging verification, repository/GitHub cleanup, full local regression, and first code-scan hardening are complete.

**Saved snapshot:** 2026-10-04 after code-scan fixes, context-invariant hardening, and full local regression.

- **Duration context:** PASS.
- **Latent-state recovery:** PASS.
- **Transition context:** PASS.
- **Multi-component identifiability:** PACKAGE-CORE PASS across the frozen 15-seed A/B/C/D gate.
- **General-context robustness:** PACKAGE-CORE PASS using split-fit direction replication.
- **Context-effect validation API:** initial, duration, emission, transition, and arbitrary state-indexed `[K,...]` context effects are supported without Nautilus/trading dependencies and without changing `NHSMM.optimize()` semantics.
- **Canonical public facade:** `component=` selects initial, duration, emission, or transition effects for extraction, replica evaluation, and split-fit evidence; component-specific helpers are internal.
- **Focused validation/API tests:** 23/23 PASS locally.
- **Normalized public API stress:** PASS — 4,800 effect extractions, 160 replication evaluations, and 16 real optimize-based split-fits across all four components without shape, finiteness, normalization, mode-restoration, determinism, or split-contract failures.
- **Full local repository regression on current `develop`:** 228 passed, 3 skipped in 23.98 s. The skips are CUDA-only tests on a CPU runtime.
- **Packaging dry-run:** PASS on GitHub runner; current local consistency scan also rebuilt sdist + wheel and passed `twine check` after migrating package license metadata to SPDX.
- **Production release workflow:** `.github/workflows/release.yml` gates build/publish on the full pytest suite; manual runs stop after verification, while PyPI publication is restricted to pushed `v*` tags.
- **AWA development workspace:** `workspace.ini` allocates 4 CPU, 8 GiB RAM, 8 GiB storage, 512 PIDs, and 1 GiB tmp. CPU-only development uses `scripts/install_cpu_dev.py`, which installs the official CPU PyTorch wheel before the normal editable dev environment and avoids unnecessary CUDA dependency resolution.

The Python 3.12 baseline, selected P0 code-scan fixes, default-distribution math/performance hardening, encoder/cache hardening, models-layer contract cleanup, convergence-monitor hardening, transition-forecast hardening, and repository-wide static/API consistency cleanup are complete. Downstream Nautilus integration remains a separate empirical validation boundary.

## Accepted package-core evidence

### Duration context

Controlled strong, moderate, and null duration-context gates: **PASS**.

### Latent-state recovery

Strong and moderate state recovery plus null descriptive control: **PASS**.

### Transition context

Binary external-context transition recovery at `max_duration=1`: **PASS** for strong, moderate, and null.

### Multi-component identifiability

Frozen 15-seed A/B/C/D package validation: **12/12 scenario gates PASS** across strong, moderate, and null conditions.

See [`validation/multicomponent-component-matched-package-15seed-2026-09-27.md`](validation/multicomponent-component-matched-package-15seed-2026-09-27.md).

### General-context robustness

Accepted detector: split-fit direction replication across two independently fitted models on disjoint training halves. State alignment is truth-free and uses learned emission centers only.

Frozen development policy for the recorded transition-context research profile:

- `replication_corr >= 0.55`;
- `min(amplitude_half1, amplitude_half2) >= 0.50`.

Fresh confirmatory result:

- strong: `5/5` selected;
- moderate: `4/5` selected;
- null: `0/5` selected.

Rejected detector variants remain rejected and must not be revived by retuning on their confirmatory blocks.

See [`validation/general-context-split-replication-confirm-2026-09-27.md`](validation/general-context-split-replication-confirm-2026-09-27.md).

## Context-effect validation API

### Canonical facade

Normal callers should use:

- `ContextEffectComponent = Literal["initial", "duration", "emission", "transition"]`
- `context_effect(model, context, component=...)`
- `evaluate_context_effect_replication(model_a, model_b, contexts, component=..., config=...)`
- `split_fit_context_effect_evidence(observations, context, component=..., fit_model=..., evidence_contexts=..., config=...)`

Shared result/policy types:

- `ContextEvidenceConfig`
- `ContextEvidence`
- `SplitFitEvidence`

Canonical semantic shapes:

- `initial` -> `[K]` normalized state prior;
- `duration` -> `[K,D]` normalized duration law;
- `emission` -> `[K,F]` state-conditioned means;
- `transition` -> `[K,K]` effective boundary-transition law.

Component-specific helpers are internal implementation details behind the canonical component-selected validation API.

Design rules:

- validation does not call or modify `NHSMM.optimize()`;
- training policy remains consumer-owned through `fit_model(...)`;
- no trading/domain assumptions;
- thresholds remain explicit policy, not package-core defaults;
- initial-state evidence uses normalized probabilities, not internal logits;
- duration evidence includes `duration_logits_bias`;
- duration-dependent transition evidence integrates over the current duration law;
- adapters restore the previous train/eval mode;
- state alignment uses learned emission centers rather than latent truth labels.

Documentation: [`validation-api.md`](validation-api.md).

## Validation/API verification

Focused semantic/API regression:

```text
23/23 PASS
```

Functional end-to-end smoke includes:

- real NHSMM initialization;
- all four normalized `component=` extraction paths;
- correct component dispatch and semantic effect shapes;
- replica-evidence dispatch;
- split-fit dispatch;
- real split-fit callbacks using `initialize_distributions()` + `NHSMM.optimize()` on disjoint halves.

Stress result:

```text
STRESS_PASS
effects=4800
replications=160
optimize_splitfits=16
runtime=22.421 s
Python-tracked peak memory=53.47 MiB
```

Persistent stress record: [`validation/public-api-stress-2026-09-28.md`](validation/public-api-stress-2026-09-28.md).

## Repository code scan / hardening

Full scan record: [`validation/code-scan-2026-09-28.md`](validation/code-scan-2026-09-28.md).

Safe behavior-preserving hardening already applied:

- ContextEncoder/ContextRouter/SequenceSet helper tensors follow active device placement;
- mask/index/padding tensors are device-safe;
- lazy attention/MHA modules are created on active device/dtype;
- temporary pooling overrides restore previous state on exceptions;
- unused encoder logger import removed;
- focused regression tests added in `tests/test_context_hardening.py`.

Verification after these changes:

```text
228 passed, 3 skipped
```

The three skips require CUDA and were run on a CPU-only local runtime.

### Remaining scan findings selected for next slice

P2:

1. Decide the long-term compatibility contract for currently unused public `ModelConfig.temperature` and `ModelConfig.debug` fields (wire, deprecate, or remove in a breaking release).

Already resolved by the code-scan maintenance pass: Python 3.12 packaging/tooling alignment, variable-length dataset state alignment, learned attention/MHA restart lifecycle, core Polars optionalization, and public numeric `ModelConfig` validation.

See `docs/validation/code-scan-2026-09-28.md` for the canonical scan status.

## Packaging / PyPI readiness

Packaging contract currently recorded in `pyproject.toml`:

- project name: `nhsmm`;
- PEP 517 backend: `setuptools.build_meta`;
- version source: Git metadata through `setuptools_scm`;
- package discovery: `nhsmm*`;
- build products: sdist + pure-Python wheel;
- runtime version: `importlib.metadata.version("nhsmm")`;
- release workflow: `.github/workflows/release.yml`;
- target: existing PyPI project `nhsmm`.

### Verified release dry-run — 2026-09-28

GitHub Actions run:

```text
36359725337
```

Verified gates:

- checkout with complete Git history/tags: PASS;
- build tooling installation: PASS;
- `python -m build`: PASS;
- `python -m twine check dist/*`: PASS;
- built wheel install into fresh venv: PASS;
- installed-package canonical public-API smoke: PASS;
- distribution artifact upload: PASS.

Dry-run artifact:

```text
name: nhsmm-release-dry-run-distributions-36359725337
size: 208303 bytes
sha256: dd741ea982c7679d07357d2dd475516904b5acfeeeb85415b03fb0c8397e130f
```

The temporary push-trigger workflow used to force this verification from the connector was removed immediately after the successful run. The canonical release workflow remains `.github/workflows/release.yml`.

### Remaining release setup

Configure PyPI Trusted Publishing for:

```text
owner: awa-si
repository: nhsmm
workflow: release.yml
environment: pypi
```

A GitHub `pypi` environment with manual approval is recommended.

No production release tag has been created by this packaging pass.

See [`release.md`](release.md).

## CI status and limits

An earlier GitHub smoke run was green before the latest code-scan hardening:

```text
run: 36392227850
pytest: 96 passed
runtime benchmark: PASS
```

The later hardening has local full-suite evidence (`228 passed, 3 skipped`) but has **not** yet been covered by a new CI run. Keep the established gate: local pre-test first; CI only after local green.

## Package boundary / next slice

Proceed in this order:

1. Decide the compatibility contract for unused public config fields `temperature` and `debug`.
2. Run focused tests for any resulting API change.
3. Run complete local pytest; require green before CI.
4. Run manual smoke/package-validation CI only when material.
5. Update state/handoff after verification.
6. Return to release/tagging or downstream Nautilus empirical evaluation only after the selected package slice is closed.

## Semantic constraints

Preserve unless explicitly justified by new research:

- causal boundary `t-1 -> t` uses only information from `F_{t-1}`; `x_t` enters only through emission after propagation;
- explicit-duration HSMM semantics remain intact;
- validation remains separate from `NHSMM.optimize()`;
- validation thresholds remain explicit policy, not package-core defaults;
- state alignment uses learned emission centers, not latent truth;
- duration evidence includes `duration_logits_bias`;
- no Nautilus/trading assumptions inside NHSMM package-core validation;
- frozen detector thresholds are not retuned on confirmatory blocks.
