# NHSMM handoff

This file is the short operational handoff for the next work session. `docs/state.md` remains the canonical readiness/status owner.

## Current continuation point

Repository: `awa-si/nhsmm`

Branch: `develop`

Snapshot date: `2026-09-28`

The package-core mechanism research, universal validation/API translation, normalized public API, stress validation, and packaging dry-run are complete. Do not reopen detector tuning or redesign accepted validation semantics unless a later package change invalidates them.

## Current verified state

Package-core evidence:

- duration context: PASS;
- latent-state recovery: PASS;
- transition context: PASS;
- multi-component identifiability: PACKAGE-CORE PASS;
- general-context robustness: PACKAGE-CORE PASS using split-fit direction replication.

Normalized public validation facade:

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

`ContextComponent` is restricted to:

```text
initial | duration | emission | transition
```

Component-specific helpers remain available as compatibility/advanced API.

## Verification status

Focused semantic/API regression:

```text
23/23 PASS
```

Stress result:

```text
STRESS_PASS
effects=4800
replications=160
optimize_splitfits=16
runtime=22.421 s
Python-tracked peak memory=53.47 MiB
```

Persistent stress record:

`docs/validation/public-api-stress-2026-09-28.md`

## Packaging status

Packaging is now functionally verified from a complete GitHub checkout.

Canonical release workflow:

```text
.github/workflows/release.yml
```

Packaging stack:

- PEP 517 / `setuptools.build_meta`;
- versioning via `setuptools_scm`;
- sdist + pure-Python wheel;
- runtime version via installed metadata;
- manual workflow run = build/install dry-run only;
- pushed `v*` tag = production release path;
- PyPI publish job uses OIDC / Trusted Publishing.

### Verified dry-run

GitHub Actions run:

```text
36359725337
```

All gates PASS:

- build tooling install;
- `python -m build`;
- `twine check dist/*`;
- clean-venv wheel installation;
- installed-package public-API smoke for all four components;
- distribution artifact upload.

Artifact:

```text
name: nhsmm-release-dry-run-distributions-36359725337
size: 208303 bytes
sha256: dd741ea982c7679d07357d2dd475516904b5acfeeeb85415b03fb0c8397e130f
```

The temporary connector-trigger workflow used solely to force this dry-run was removed after success. Do not restore it; use the canonical manual trigger in `release.yml` for future rehearsals.

## Remaining production-release setup

Before a real PyPI release:

1. Run the complete discovered pytest suite against the intended release commit from a complete checkout.
2. Configure PyPI Trusted Publishing:

```text
owner: awa-si
repository: nhsmm
workflow: release.yml
environment: pypi
```

3. Create GitHub environment `pypi`; manual approval is recommended.
4. Choose a new version/tag only after the full regression is green.
5. Do not reuse an already published version.
6. Push the release tag; let `release.yml` rebuild and publish the exact verified distributions.

No production tag was created during the packaging verification pass.

## Important semantic constraints

Preserve these unless explicitly justified by new research:

- validation remains separate from `NHSMM.optimize()`;
- training policy is consumer-owned through `fit_model(...)`;
- no trading/Nautilus assumptions inside NHSMM validation;
- validation thresholds remain explicit policy, not package-core defaults;
- state alignment uses learned emission centers, not latent truth;
- initial-state evidence uses normalized probabilities, not internal logits;
- duration evidence includes `duration_logits_bias`;
- effective transition evidence integrates duration-dependent transition laws over the current duration distribution;
- adapters restore previous train/eval mode;
- frozen detector thresholds must not be retuned on prior confirmatory seeds.

## Current package boundary

The next task is **not more detector/API expansion and not another packaging redesign**.

Proceed in this order:

1. complete-package pytest regression;
2. fix only real regressions;
3. configure Trusted Publisher / `pypi` environment;
4. choose release version;
5. manual canonical release dry-run if release commit changed materially;
6. tag release;
7. only then return to downstream Nautilus empirical evaluation.

## Source of truth

Read:

- `docs/state.md`
- `docs/release.md`
- `docs/validation-api.md`
- `docs/validation/public-api-stress-2026-09-28.md`
- `docs/validation/general-context-split-replication-confirm-2026-09-27.md`
- `docs/validation/multicomponent-component-matched-package-15seed-2026-09-27.md`
