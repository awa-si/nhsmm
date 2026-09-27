# NHSMM State

Current package readiness and next package-level boundary. Detailed semantics: [`model.md`](model.md). Controlled evidence: [`package-validation.md`](package-validation.md). Verification policy: [`testing.md`](testing.md). Release procedure: [`release.md`](release.md).

## Current status

**Phase:** package-core mechanism research, universal validation API translation, and packaging verification are complete.

**Saved snapshot:** 2026-09-28 after normalized-public-API functional/stress validation and successful package build/install dry-run.

- **Duration context:** PASS.
- **Latent-state recovery:** PASS.
- **Transition context:** PASS.
- **Multi-component identifiability:** PACKAGE-CORE PASS across the frozen 15-seed A/B/C/D gate.
- **General-context robustness:** PACKAGE-CORE PASS using split-fit direction replication.
- **Universal validation API:** initial, duration, emission, transition, and arbitrary state-indexed `[K,...]` context effects are supported without Nautilus/trading dependencies and without changing `NHSMM.optimize()` semantics.
- **Normalized public facade:** canonical `component=` API implemented for extraction, replica evaluation, and split-fit evidence; component-specific helpers remain behavior-compatible.
- **Focused validation/API tests:** 23/23 PASS locally.
- **Normalized public API stress:** PASS — 4,800 effect extractions, 160 replication evaluations, and 16 real optimize-based split-fits across all four components without shape, finiteness, normalization, mode-restoration, determinism, or split-contract failures.
- **Packaging dry-run:** PASS on GitHub runner — sdist + wheel build, `twine check`, clean-venv wheel install, installed-package public-API smoke, and artifact upload all succeeded.
- **Production release workflow:** `.github/workflows/release.yml` is present; manual runs are build/install dry-runs only, while PyPI publication is restricted to pushed `v*` tags.

The next package boundary is a complete-package regression run plus release gating. Downstream Nautilus integration remains a separate empirical validation boundary.

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

## Universal validation API

### Canonical facade

Normal callers should use:

- `ContextComponent = Literal["initial", "duration", "emission", "transition"]`
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

Existing component-specific helpers remain available as compatibility/advanced API.

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
- equality to component-specific helpers;
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

## Packaging / PyPI readiness

Packaging contract:

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

One external setup item remains before production publishing:

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

## Verification limits

The packaging path has now been verified from a complete GitHub checkout and clean wheel-install environment. The broad complete-repository pytest suite still needs a fresh final run against the intended release commit after the recent validation/API work.

Packaging CI evidence is release verification only; routine model/research development remains local-first and does not rely on CI.

## Package boundary / next slice

1. Run the complete discovered pytest suite against current `develop` from a complete checkout.
2. Fix only regressions actually exposed by that run; preserve accepted probabilistic and validation semantics.
3. Confirm PyPI Trusted Publisher + GitHub `pypi` environment.
4. Choose the next release version/tag only after full regression is green.
5. Run the canonical release workflow manually once more if the release commit changes materially.
6. Create the release tag; the tag-triggered workflow must rebuild, verify tag/version equality, and publish the exact verified distributions.
7. Treat downstream Nautilus integration as a separate empirical validation task.
