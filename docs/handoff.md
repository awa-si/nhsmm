# NHSMM handoff

Operational continuation point for the next chat. `docs/state.md` remains the canonical package-status owner.

## Repository

- repository: `awa-si/nhsmm`
- branch: `develop`
- snapshot date: `2026-10-05`
- base head for this handoff update: `c41bc19737d7010f00bcecd936875e9ac6a840dd`

## Current state

### 2026-10-05 active continuation

Current verified core commits before this documentation update:

- `eca455392bbdcde1be7d5f45f2d9ddbd84bdba2e` — distribution initialization contracts: context-free stored base parameters, context applied exactly once, and emission/K-Means reinitialization preserves registered parameter identity.
- `22a298a8ec3a10e146b4afe0a2cc5d3e4354be64` — propagate configured dropout into `DefaultEncoder`.
- `c41bc19737d7010f00bcecd936875e9ac6a840dd` — `workspace.ini` declares `data = nautilus`.

Full local suite after the core fixes:

```text
314 passed
3 skipped  # CUDA unavailable
0 failed
```

AWA dataset rule:

```text
workspace.ini: data = nautilus
runtime mount: /data/nautilus (read-only)
```

Do not transfer/copy the Nautilus dataset between workspaces. A fresh repository import resolves the declared dataset and mounts it automatically.

Current Nautilus 3-vs-4 result (`3 seed bases x 2 folds`, public NHSMM API, `max_iter=8`, `n_init=1`, K-Means init, causal, dropout 0):

| metric | 3 states | 4 states |
| --- | ---: | ---: |
| healthy | **6/6** | 5/6 |
| mean OOS LL / step | **-13.074227** | -13.080162 |
| mean effective states | 2.7878 | **3.8317** |
| min Viterbi states | **2** | 1 |
| min posterior-MAP states | **2** | 1 |

Decision: **retain `n_states=3` as the Nautilus operational baseline.** Four states add posterior capacity but currently reduce robustness and do not improve OOS likelihood.

The 3-state third latent state is typically soft/non-dominant. Do not add a Viterbi-usage penalty merely to force all three states into the hard path. Prior probes rejected transition entropy, posterior entropy, static emission-separation/accessibility penalties, and decoder substitution as fixes.

Detailed evidence: `docs/validation/nautilus-state-count-3v4-2026-10-05.md`.

Research workspace used for the current Nautilus study:

```text
workspace_id: 27089405d2c2400192164b0866536016
repo head: c41bc19737d7010f00bcecd936875e9ac6a840dd
dataset mount: /data/nautilus
```


Package-core mechanism research, canonical context-effect validation API, distribution initialization hardening, deterministic encoder-dropout propagation, full local regression, and the current downstream Nautilus state-count usability study are complete.

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

Full repository pre-CI regression on current `develop` after the latest context-invariant hardening:

```text
314 passed, 3 skipped
```

The three skips are CUDA-only tests on a CPU runtime. No failing local test remains in that verified workspace.

Earlier GitHub smoke CI before the code-scan hardening was green:

```text
run: 36392227850
pytest: 96 passed
runtime benchmark: PASS
```

Do not claim CI coverage of the later code-scan hardening until a new CI run is actually executed.

## AWA workspace

Canonical development resources are defined in `workspace.ini`:

```text
cpu=4
memory_mb=8192
storage_mb=8192
pids=512
tmp_mb=1024
```

The 8 GiB storage profile is accepted by the current AWA runtime. CPU-only development now uses `python scripts/install_cpu_dev.py` inside the virtual environment; it installs the CPU PyTorch wheel first and then the normal editable `.[dev]` environment, preventing unnecessary CUDA dependency resolution.

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
full pytest -> build/verify -> tag-gated PyPI publish
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

The corresponding current local full suite is `314 passed, 3 skipped`.

## Code-scan maintenance status

The previously selected Python/P0 maintenance work is complete:

- Python baseline/tooling/workflows are aligned to 3.12+.
- `SequenceDataset(variable_length=True)` state alignment is fixed and regression-tested.
- learned `attn` / `mha` pooling parameters survive reset/restarts and remain optimizer/state-dict members.
- framework-specific dataframe ingestion was removed from core; `nhsmm/data.py` contains synthetic dataset helpers only.
- main public numeric `ModelConfig` bounds are validated early.
- later context-invariant hardening in `e9169cb` preserves temporal encoder output, validates SequenceSet/ContextRouter shapes, forwards supported masks, and normalizes masks to the active device/bool dtype.

Default-distribution math/performance hardening is also complete:

- initial-state output is normalized log probability rather than raw logits;
- categorical helpers enforce a final category axis and valid probability mass;
- Student-t event dimensions, moments, expansion, and scoring match PyTorch semantics;
- emissions preserve absolute learned locations and are not temperature-scaled;
- context-aware initialization preserves vector-context and dtype/device contracts;
- structural masks reject empty support rows;
- static transition topology is cached;
- batch sequence construction uses the specialized emission scorer instead of constructing full distribution objects.

Measured CPU microbenchmarks showed about 12x lower Gaussian batch-emission scoring latency, about 2x lower Student-t batch-emission latency, and about 8x lower static transition-constraint cost for the tested shapes.

ContextEncoder transient-cache retention is also resolved: `_sequence` and `_context` store detached diagnostic snapshots while returned tensors retain their autograd graph. `reset()` continues to clear only transient caches and preserves learned attention/MHA parameters.

DefaultEncoder audit/hardening is complete:

- non-causal CNN output preserves sequence length for odd and even kernels;
- masks are validated as right-padded sequence masks instead of silently accepting holes incompatible with packed recurrence;
- empty mask rows no longer synthesize a fake valid timestep;
- masked padding is zeroed before convolution so non-causal kernels cannot leak padding values into valid outputs;
- masked output positions are zeroed consistently;
- the private pooled-context cache is detached and no longer retains the latest autograd graph;
- causal streaming parity remains covered by incremental-runtime tests.

Canonical runtime benchmark on a temporary causal artifact (128 steps, 16 warmup, batch 1): p50 11.27 ms, p95 55.04 ms, mean 17.25 ms, bounded runtime state 151 tensor elements.

Models-layer audit/hardening is complete:

- `nhsmm.models` now exposes the single canonical `DistributionSet`/`NHSMM` implementation from `models/base.py`; the redundant `models/training.py` layer was removed;
- distribution restarts preserve injected component factories;
- base/default encoder dimension validation matches the canonical training model;
- restart initialization keeps base distribution parameters independent of observed external context;
- floating observations/context are coerced to the model dtype;
- external context lists must match observation sequence lengths exactly;
- public 2-D context semantics are unified through `align_context_tensor()`;
- transition refinement without explicit context is guarded and regression-tested.

Convergence-monitor audit/hardening is complete:

- plateau convergence now fires after exactly the configured window of observations instead of one iteration late;
- zero tolerances accept exact zero change;
- non-finite scores are rejected and cannot leave stale converged flags behind;
- negative/out-of-range iteration and initialization indices are rejected explicitly;
- constructor invariants validate counts, tolerances, mode, and patience bounds;
- LR-aware patience scaling uses scalar math without transient tensor allocation.

Transition-forecast audit/hardening is complete:

- forecast input now requires a normalized HSMM state posterior explicitly;
- the public forecast result validates its probability identities, not only tensor shapes/ranges;
- boundary mass, next-episode mass, normalized next-state prior, and change-probability bounds are checked coherently;
- state-change probability is computed as boundary mass minus self-transition mass, avoiding a temporary KxK eye mask;
- D=1 deterministic episode termination and dtype/device fail-closed behavior are verified.

Unused public config cleanup is complete:

- `ModelConfig.temperature` and `ModelConfig.debug` were removed after repository and `nhsmm-interfaces` consumer scans found no users;
- model-level `self.debug` was removed with the dead config field;
- distribution/runtime temperature APIs remain intact because they are active consumers and separate from the removed config field;
- artifact payloads no longer persist the removed dead fields.

End-to-end learning/prediction acceptance is implemented:

- canonical harness: scripts/validate_learning_prediction.py;
- 20-fit acceptance run covers strong/moderate/weak/null signal regimes over disjoint train/OOS data;
- every run changed parameters and passed model/data fingerprint plus variable-length prediction contracts;
- median OOS accuracy: strong 0.9978, moderate 0.9656, weak 0.7611, null 0.3589;
- median OOS LL gains: +7.8224, +2.1719, +1.1146, +1.0093 respectively;
- null median ARI is 0.0 and all 5 null runs collapse, demonstrating no spurious state-identity recovery;
- detailed evidence is recorded in docs/validation/learning-prediction-2026-10-04.md.



Structural usefulness validation is now part of the maintained learning/prediction acceptance:

- strong and moderate pass both learning and usefulness gates;
- weak passes a lower-confidence usefulness contract, with exact boundary timing explicitly qualified and ±1-timestep timing reported separately;
- null is retained strictly as a negative control and does not receive a usefulness claim;
- usefulness metrics cover boundary timing, run lengths, transition recovery, occupancy recovery, posterior calibration, and collapse behavior.

Repository-wide file consistency audit completed: 121 tracked files reviewed; 28/28 package modules import; no root-export gaps, import cycles, tracked bytecode, or broken local Markdown links. Safe consistency fixes aligned model imports, artifact/snapshot config serialization, README/agent-domain ownership, and empirical/release workflows. The legacy Freqtrade/Polars loader and dataframe path have now been removed from `nhsmm/data.py`; data helpers are core-only synthetic fixtures. Full regression remains `314 passed, 3 skipped`.

A public `ConfigTuner` layer now sits above the centralized configuration contracts. It supports validated grid/random candidate generation, maximize/minimize objectives, reusable dotted overrides, serializable trial evidence, and deterministic best-config selection. Full regression after this addition: `314 passed, 3 skipped`.

Configuration ownership is consolidated in `nhsmm/config.py`; the previous validation-local config module was removed. Public imports remain unchanged, while model, health, data, scenario, and validation contracts now use the same strict helpers. Full local regression: `314 passed, 3 skipped`.

Configuration/tuning contract is implemented:

- `ModelConfig` has strict serialization and validated override helpers;
- public schema-versioned `ValidationConfig` owns acceptance dataset/model/health/scenario policy;
- validation thresholds and health/collapse policy are tuneable without editing the harness;
- CLI supports JSON config loading, resolved-config dumping, and strict dotted overrides;
- latest full local suite after this change: `314 passed, 3 skipped`.

Production-boundary hardening is complete:

- persisted model state validation covers parameters and persistent buffers;
- artifact build/load and inference loading reject non-finite floating state tensors before use;
- external context and streaming observations reject non-finite values instead of propagating/sanitizing them;
- runtime state requires finite probability mass in every duration/transition row;
- categorical construction/sampling validates finite logits and positive finite temperatures;
- filter-trace length dtype/device contracts are explicit;
- exact-byte validation fingerprints support scalar and `bfloat16` tensor state.

Reproducible validation snapshots are implemented:

- public `evaluate_validation_snapshot(...)`, `compare_validation_snapshots(...)`, `ValidationSnapshot`, `ValidationComparison`, and `model_fingerprint(...)`;
- snapshots fingerprint fitted model state, observations, and explicit context independently;
- sequence lengths, per-sequence/per-timestep likelihood, switch rate, mean run length, and model health are recorded in one JSON-serializable result;
- same-model comparison is enforced by default so independently fitted latent labels are not compared without alignment;
- learned-model train/OOS smoke kept the same model fingerprint, distinct data fingerprints, healthy state usage, and stable switch/run-length behavior.

Model-health diagnostics are implemented and validated:

- public `evaluate_model_health(...)`, `ModelHealthReport`, and `ModelHealthThresholds`;
- state-collapse detection combines effective state count, max occupancy, and decoded state usage;
- duration and transition degeneration are reported separately from state collapse;
- diagnostics use actual aligned sequence context and only valid unpadded timesteps;
- variable-length Viterbi prediction now trims each decoded path to its original sequence length;
- empirical gate over 15 fits: strong 0/5 collapsed, moderate 0/5 collapsed, null 5/5 collapsed; duration/transition degeneration 0/15.

File/code consolidation is complete:

- redundant `nhsmm/models/training.py` was removed; all public model imports now resolve to the single implementation in `models/base.py`;
- external-context alignment and optional transition refinement were folded into the canonical model rather than maintained in a subclass;
- repeated validation correlation logic was centralized in `validation/alignment.py`;
- filtering/runtime/transition dtype-device compatibility checks share one internal helper;
- exact top-level function-body duplicate scan over `nhsmm/` is clean.

Repository-wide consistency cleanup is complete:

- the documented Ruff/Black static gate is clean across `nhsmm`, `tests`, and `scripts`;
- the obsolete distribution-module monkeypatch was removed;
- package logging no longer installs/levels a stream handler at import time and now uses a library-safe `NullHandler`;
- `ModelConfig` rejects non-finite numeric values, invalid Literal values, and non-boolean flag values at construction time;
- artifact construction is confirmed observational by existing regression coverage and the stale finding was removed.

Packaging/static consistency verification is also green:

- `ruff check nhsmm tests scripts`: PASS;
- `black --check nhsmm tests scripts`: PASS;
- local sdist + wheel build: PASS;
- `twine check dist/*`: PASS;
- package license metadata now uses SPDX `Apache-2.0` rather than deprecated setuptools license-table/classifier forms;
- mypy 2.4.0 under the current Python 3.14 workspace aborted with an internal mypy error before emitting repository diagnostics, so no mypy-clean claim is made.

The maintained repository-consistency scan has no unresolved finding.

## 2026-10-05 production-readiness continuation

Current `develop` was profiled on a representative causal `K=3`, `F=18`, `D=12` workload before any further semantic change.

Observed:

- pilot fit `13.22 s`;
- OOS LL/step `-13.0633`;
- healthy model with effective states `2.996` and 3/3 Viterbi states used;
- artifact load/save roundtrip preserved LL and decode exactly;
- streaming CPU p50 `1.98 ms`, p95 `5.47 ms`, mean `2.61 ms`;
- runtime state stayed bounded at `306` tensor elements;
- five-seed health replication: `5/5` healthy.

Repository hygiene drift found during the same pass was formatting/static-only and has been normalized. Final local gate: Ruff PASS, Black PASS, `314 passed, 3 skipped`.

Important integration blocker: `/data/nautilus/regime_model.joblib` currently describes an older 29-feature HMM/scaler/schema, but the canonical `nhsmm-interfaces` Nautilus temporal contract is 18-dimensional. Do not treat that mounted Joblib artifact as current 18-D validation data. The next real-Nautilus production gate requires a current 18-coordinate chronological fixture/replay source.

## 2026-10-06 production-hardening closure

The follow-up code audit found and closed three core defects: stale runtime score-cache reuse after parameter replacement, `decode()` failure for variable-length list input, and silent conversion of positive-infinite likelihood to dtype maximum.

Current verification after the fixes: focused regressions `4 passed`; Ruff PASS; Black PASS; full suite `317 passed, 3 skipped` (CUDA unavailable).

## 2026-10-06 math/Torch correctness closure

The mathematical/PyTorch audit closed variable-length padding leakage, padded-row NaN autograd in causal and non-causal forward recursions, duplicate external-context application, the `max_duration=1` NaN regularizer edge case, and standard-optimizer inclusion of scheduled temperature parameters.

Independent brute-force filtering checks and forward/filter equivalence checks passed. Final local gate: focused regressions `20 passed`; Ruff PASS; Black PASS; full suite `321 passed, 3 skipped` (CUDA unavailable).

## 2026-10-06 models/base audit closure

`nhsmm/models/base.py` was audited independently. Causal and non-causal recursion checks passed, including full small-model enumeration for non-causal likelihood and Viterbi under both transition forms. Boundary hardening now covers uninitialized distributions, malformed/non-finite observations, empty likelihood batches, structurally incompatible optimization configs, and train/eval restoration across restarts.

Final local gate: focused regressions `5 passed`; Ruff PASS; Black PASS; full suite `330 passed, 3 skipped` (CUDA unavailable).

## Next-chat execution order

1. Read `AGENTS.md`, `docs/state.md`, this handoff, and `docs/validation/code-scan-2026-09-28.md`.
2. Resolve current `develop` head before editing.
3. Continue with release/tagging or downstream Nautilus evaluation unless a new concrete core finding is identified.
4. For any new package-core change, run focused tests and the full local pytest suite before CI.
5. Use `scripts/install_cpu_dev.py` in the AWA CPU workspace to avoid the CUDA dependency footprint.
6. Update `docs/state.md` and `docs/handoff.md` after verification.

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

- `AGENTS.md`
- `docs/state.md`
- `docs/handoff.md`
- `docs/validation/code-scan-2026-09-28.md`
- `docs/release.md`
- `docs/testing.md`
- `docs/validation-api.md`
- `docs/validation/public-api-stress-2026-09-28.md`
