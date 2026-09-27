# Release procedure

NHSMM uses `setuptools`/PEP 517 packaging, `setuptools_scm` for tag-derived versions, and PyPI Trusted Publishing for releases.

## Release contract

- package name: `nhsmm`
- source of version: Git tag via `setuptools_scm`
- accepted release tag form: `vX.Y.Z` or a prerelease form accepted by the configured tag regex
- build products: source distribution and pure-Python wheel
- publish workflow: `.github/workflows/release.yml`
- publish target: PyPI project `nhsmm`

The workflow can be started manually as a **build/install dry run**. Manual runs never publish. Production publishing happens only for pushed `v*` tags.

## One-time PyPI setup

Configure a Trusted Publisher on the existing `nhsmm` PyPI project with:

- owner: `awa-si`
- repository: `nhsmm`
- workflow filename: `release.yml`
- environment: `pypi`

Create the GitHub environment `pypi`. A manual approval rule is recommended for the environment before production publishing.

No long-lived PyPI API token is required when Trusted Publishing is configured.

## Pre-release dry run

Before creating a release tag, run the GitHub workflow `NHSMM release` manually. The manual run performs all packaging checks but skips the publish job.

It verifies:

1. full-history checkout for `setuptools_scm`;
2. sdist + wheel build;
3. `twine check` metadata validation;
4. installation of the built wheel into a fresh venv;
5. import of the installed package from outside the source checkout;
6. canonical public context API smoke for `initial`, `duration`, `emission`, and `transition`;
7. artifact creation for the exact verified distributions.

Also run the complete package pytest suite from a complete local checkout before release.

## Local build check

When a complete checkout is available locally:

```bash
python -m pip install --upgrade build twine
rm -rf build dist *.egg-info
python -m build
python -m twine check dist/*
```

`python -m build` is intentional: it first creates the sdist and then builds the wheel from that sdist, validating that the sdist contains the files needed for a wheel build.

## Release

Choose the intended version and create a matching annotated tag on the release commit, for example:

```bash
git tag -a v0.0.5a0 -m "NHSMM 0.0.5a0"
git push origin v0.0.5a0
```

Pushing the tag starts `.github/workflows/release.yml`.

The tag-triggered workflow:

1. repeats the complete build/install verification;
2. verifies the installed package version exactly equals the pushed tag without its `v` prefix;
3. uploads the exact verified distributions as an artifact;
4. publishes those same distributions through PyPI Trusted Publishing.

A publish job cannot run unless the build/verification job passes, and the publish job is gated to `refs/tags/v*`.

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

If build, metadata validation, wheel installation, public-API smoke, or tag/version verification fails, do not upload manually as a workaround. Fix the release commit, choose a new release version/tag if necessary, and rerun the release process.

PyPI release files are immutable: never reuse a version that has already been uploaded.

## TestPyPI

TestPyPI is optional. The production workflow intentionally publishes only to PyPI after local/full-package validation. If a TestPyPI rehearsal is needed, use a separate temporary/manual workflow rather than weakening the production release job.
