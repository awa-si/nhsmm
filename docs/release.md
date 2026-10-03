# Release procedure

NHSMM uses `setuptools`/PEP 517 packaging, `setuptools_scm` for tag-derived versions, and PyPI Trusted Publishing for production releases.

## Release contract

- package name: `nhsmm`
- source of version: Git tag via `setuptools_scm`
- accepted release tag form: `vX.Y.Z` or a prerelease form accepted by the configured tag regex
- build products: source distribution and pure-Python wheel
- publish workflow: `.github/workflows/release.yml`
- publish target: PyPI project `nhsmm`

The canonical workflow supports two modes:

- manual `workflow_dispatch`: build/install dry-run only; never publishes;
- pushed `v*` tag: full build/verification followed by PyPI publication.

## Verified packaging dry-run

A complete GitHub-runner dry-run was executed successfully on 2026-09-28.

Run:

```text
36359725337
```

Verified gates:

1. complete checkout suitable for `setuptools_scm`;
2. build tooling installation;
3. `python -m build`;
4. `python -m twine check dist/*`;
5. installation of the built wheel into a fresh virtual environment;
6. import of the installed package from outside the source checkout;
7. canonical public context API smoke for `initial`, `duration`, `emission`, and `transition`;
8. upload of the exact verified distributions as a workflow artifact.

Result: **PASS**.

Artifact:

```text
name: nhsmm-release-dry-run-distributions-36359725337
size: 208303 bytes
sha256: dd741ea982c7679d07357d2dd475516904b5acfeeeb85415b03fb0c8397e130f
```

A temporary push-trigger workflow was used only because the active connector could not issue a new `workflow_dispatch`. It was deleted immediately after the successful run. Do not restore it; future rehearsals should use the manual trigger in the canonical `release.yml` workflow.

## One-time PyPI setup

Before the first Trusted-Publishing release, configure a Trusted Publisher on the existing `nhsmm` PyPI project with:

```text
owner: awa-si
repository: nhsmm
workflow filename: release.yml
environment: pypi
```

Create the GitHub environment `pypi`. A manual approval rule is recommended before production publishing.

No long-lived PyPI API token is required once Trusted Publishing is configured.

## Pre-release checklist

Before creating a release tag:

1. Use a complete checkout of the intended release commit.
2. Run the complete discovered pytest suite.
3. Confirm the release commit contains the desired public API and documentation.
4. Confirm `git status` is clean in the release checkout.
5. Confirm the chosen version is new on PyPI.
6. Run the canonical `NHSMM release` workflow manually if the release commit changed materially since the last packaging verification.
7. Verify the manual build/install job is green before tagging.

Optional local check when a complete local checkout and package-index access are available:

```bash
python -m pip install --upgrade build twine
rm -rf build dist *.egg-info
python -m build
python -m twine check dist/*
```

`python -m build` is intentional: it creates an sdist and builds a wheel from that sdist, validating that the source distribution contains everything required for wheel construction.

## Production release

Choose the intended version and create a matching annotated tag on the release commit, for example:

```bash
git tag -a v0.0.5a0 -m "NHSMM 0.0.5a0"
git push origin v0.0.5a0
```

Pushing the tag starts `.github/workflows/release.yml`.

The tag-triggered workflow:

1. runs the complete pytest suite on Python 3.12;
2. checks out full Git history/tags for `setuptools_scm`;
3. builds sdist + wheel;
4. runs `twine check`;
5. installs the built wheel into a fresh venv;
6. verifies the installed package version exactly equals the pushed tag without its `v` prefix;
7. runs the installed-package public-API smoke;
8. uploads the exact verified distributions as an artifact;
9. publishes those same distributions to PyPI through Trusted Publishing.

The build job cannot run unless the full test job succeeds. The publish job cannot run unless build/verification succeeds, and it is gated to `refs/tags/v*`.

## Versioning

Do not manually edit a source version constant for releases. `pyproject.toml` declares a dynamic version and `setuptools_scm` derives it from Git metadata.

`nhsmm.__version__` is read from installed package metadata.

For a release build from tag `vX.Y.Z`, the installed version must be exactly `X.Y.Z`. The release workflow checks this before publishing.

The configured tag pattern accepts examples such as:

```text
v0.0.5a0
v0.1.0a0
v0.1.0
```

## Failure handling

If build, metadata validation, wheel installation, public-API smoke, or tag/version verification fails, do not upload manually as a workaround.

Instead:

1. fix the release commit;
2. rerun the complete regression/package verification;
3. choose a new version/tag if an immutable PyPI version was already consumed;
4. rerun the release process.

PyPI release files are immutable: never reuse a version that has already been uploaded.

## TestPyPI

TestPyPI is optional. The production workflow intentionally publishes only to PyPI after package validation. If a TestPyPI rehearsal is needed, use a separate temporary/manual workflow rather than weakening the production release job.
