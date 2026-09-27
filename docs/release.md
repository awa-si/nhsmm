# Release procedure

NHSMM uses `setuptools`/PEP 517 packaging, `setuptools_scm` for tag-derived versions, and PyPI Trusted Publishing for releases.

## Release contract

- package name: `nhsmm`
- source of version: Git tag via `setuptools_scm`
- accepted release tag form: `vX.Y.Z` or a prerelease form accepted by the configured tag regex
- build products: source distribution and pure-Python wheel
- publish workflow: `.github/workflows/release.yml`
- publish target: PyPI project `nhsmm`

The release workflow is tag-only. It does not run on ordinary pushes or pull requests.

## One-time PyPI setup

Configure a Trusted Publisher on the existing `nhsmm` PyPI project with:

- owner: `awa-si`
- repository: `nhsmm`
- workflow filename: `release.yml`
- environment: `pypi`

Create the GitHub environment `pypi`. A manual approval rule is recommended for the environment before production publishing.

No long-lived PyPI API token is required when Trusted Publishing is configured.

## Pre-release checklist

Before creating a release tag:

1. Use a complete checkout of the intended release commit.
2. Run the complete discovered pytest suite.
3. Confirm `git status` is clean.
4. Confirm README/package metadata describe the current public API.
5. Confirm the release version is new on PyPI.
6. Run a local package build when possible:

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

The workflow:

1. checks out full Git history/tags for `setuptools_scm`;
2. builds sdist + wheel;
3. runs `twine check`;
4. installs the built wheel into a fresh virtual environment;
5. verifies installed package version equals the pushed tag;
6. runs a public-API import/context-effect smoke against the installed wheel;
7. uploads the exact verified distributions as an artifact;
8. publishes those same distributions through PyPI Trusted Publishing.

A publish job cannot run unless the build/verification job passes.

## Versioning

Do not manually edit a source version constant for releases. `pyproject.toml` declares a dynamic version and `setuptools_scm` derives it from Git metadata.

`nhsmm.__version__` is read from installed package metadata.

For a release build from tag `vX.Y.Z`, the installed version must be exactly `X.Y.Z`. The release workflow checks this before publishing.

## Failure handling

If build, metadata validation, wheel installation, public-API smoke, or tag/version verification fails, do not upload manually as a workaround. Fix the release commit, choose a new release version/tag if necessary, and rerun the release process.

PyPI release files are immutable: never reuse a version that has already been uploaded.

## TestPyPI

TestPyPI is optional. The production workflow intentionally publishes only to PyPI after local/full-package validation. If a TestPyPI rehearsal is needed, use a separate temporary/manual workflow rather than weakening the production release job.
