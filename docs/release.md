---
title: NHSMM package and release verification
description: Local qualification gates and retired GitHub Actions release automation.
type: procedure
status: active
language: en
owner: NHSMM
updated: 2026-11-23
---

# NHSMM release verification

NHSMM GitHub Actions smoke and release workflows were removed from `main`.
Automated CI and PyPI publishing are **not available** in this repository.
Previously configured workflow-specific PyPI Trusted Publishing cannot be
considered operational.

## Local package qualification gates

1. Pin a full NHSMM source commit and Nautilus test-owner commit.
2. Execute the full canonical suite at `tests/ml_stack/nhsmm` against that
   exact core source, recording JUnit hashes, results, CUDA skips and limits.
3. Run Ruff and Black where available, recording omitted checks.
4. Run `scripts/smoke_cpu.py` with the qualified source.
5. From a complete tagged checkout, build sdist and wheel in a controlled
   isolated environment with suitable build and twine tooling:

```bash
python -m build
python -m twine check dist/*
```

6. Install the wheel in a clean virtual environment outside the checkout,
   verify its version matches the intended tag, and run the public CPU
   smoke using the installed wheel. Retain artifact checksums.

Do not rely on stale installed metadata or a previously qualified revision.

## Publishing status

Pushing a version tag does not publish packages. A new authorized, reviewed
and independently tested publication mechanism is required. Do not publish
untested artifacts or bypass failed gates. `setuptools_scm` continues to own
version derivation; PyPI files are immutable, so never reuse uploaded versions.

## Historical release rehearsal

The previous workflow and its dated package qualification remain historical
evidence, not a currently executable procedure:

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

A temporary push-trigger workflow was used for this dated rehearsal and was deleted afterward. The former manual release workflow was subsequently removed as well; this record must not be interpreted as a currently available release path.


The former `.github/workflows/smoke.yml` and
`.github/workflows/release.yml` are not active on the current branch.
Existing run records may remain visible on GitHub.
