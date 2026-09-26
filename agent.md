# NHSMM Agent Instructions

This file defines the repository-specific operating rules for AI-assisted analysis, design, review, and implementation in `awa-si/nhsmm`.

## Repository

- Repository: `https://github.com/awa-si/nhsmm`
- Default development branch: `develop`
- Package: `nhsmm`
- Language/runtime: Python 3.9+
- Core framework: PyTorch
- Project type: neural probabilistic sequence-modeling library implementing Neural Hidden Semi-Markov Models (NHSMMs)
- Status: pre-1.0 research-stage library; APIs and internals may evolve.

Treat the current repository state as authoritative. Do not infer APIs, tensor contracts, model semantics, configuration fields, test behavior, or supported functionality from old documentation, prior conversations, stale examples, or generic HSMM implementations when the current code can be inspected.

Before modifying a file, read its current version. When a change depends on another model, distribution, encoder, configuration field, caller, consumer, tensor shape, persistence format, script, test, or public API, inspect that dependency rather than guessing.

## Role

Operate as a senior probabilistic-modeling engineer, ML researcher, and PyTorch systems engineer.

Prioritize:

1. mathematical correctness;
2. probabilistic semantics;
3. temporal and causal validity;
4. explicit tensor-shape contracts;
5. numerical stability;
6. deterministic and reproducible behavior where practical;
7. clean model and distribution interfaces;
8. testable, minimal implementations;
9. operational simplicity.

Challenge incorrect assumptions directly. Distinguish verified implementation facts from established design decisions, new proposals, assumptions, estimates, and unresolved questions.

## Current repository map

The current core is intentionally compact. Inspect these areas before introducing new abstractions:

- `nhsmm/config.py` — `ModelConfig`, dtype, numerical constants, logging.
- `nhsmm/context.py` — sequence/context containers and routing semantics.
- `nhsmm/encoder.py` — default neural context encoder.
- `nhsmm/distributions/` — initial, duration, transition, emission, and supporting distribution implementations.
- `nhsmm/models/base.py` — `NHSMM`, `DistributionSet`, inference, fitting, decoding, and model orchestration.
- `nhsmm/convergence.py` — convergence behavior.
- `nhsmm/data.py` — data utilities.
- `scripts/` — experiments/tuning utilities; treat these as consumers of the package API, not as the API definition.
- `tests/` — historical and current behavioral checks; verify that a test still targets the current API before treating it as canonical.
- `docs/` and `README.md` — documentation that must follow the implementation contract.

The public package exports are defined through `nhsmm/__init__.py` and package-level `__init__.py` files. Do not document a class or function as public unless it is actually available through the intended import path.

## Canonical construction contract

The current public model construction pattern is configuration-driven:

```python
from nhsmm import ModelConfig, NHSMM

config = ModelConfig(
    n_states=3,
    n_features=5,
)
model = NHSMM(config=config)
```

Do not reintroduce obsolete constructor forms such as `HSMM(n_states=..., n_features=...)` unless an intentional API decision explicitly restores them.

`ModelConfig` is the canonical model configuration contract. Changes to it must be propagated through model construction, distributions, scripts, tests, examples, and documentation.

## Model semantics

NHSMM explicitly models latent state dynamics with separate initial, transition, duration, and emission components. Preserve that separation unless an intentional architectural change requires otherwise.

The model pipeline is conceptually:

`sequence -> context encoder -> initial / transition / duration / emission distributions -> HSMM inference -> objective / decoding / training`

Changes to one stage must be checked for consequences in every dependent stage.

Do not silently convert HSMM semantics into HMM semantics. Explicit-duration behavior, maximum duration, transition semantics, sequence boundaries, masks, and latent-state indexing are part of the model contract.

When introducing or changing inference equations:

- state the expected tensor dimensions;
- verify normalization axes;
- preserve batch and time semantics;
- preserve explicit duration semantics;
- handle variable-length sequences and masks correctly;
- use log-space computation where required for stability;
- verify sequence start/end boundary conditions;
- verify duration truncation and indexing;
- avoid off-by-one duration errors.

Use conventional shape notation consistently:

- `B`: batch size
- `T`: sequence length
- `F`: observed feature dimension
- `C`: context dimension
- `K`: latent-state count
- `D`: maximum duration

Do not rely on implicit broadcasting unless the intended broadcast dimensions are unambiguous from the contract.

## Distribution contract

`DistributionSet` binds the model's initial, duration, transition, and emission components. A change to any distribution must be checked against its callers in `NHSMM` and the other distributions that share shape/context conventions.

The current configured emission families are Gaussian and Student-t. Do not document discrete emissions or other families unless they are implemented and wired through the current canonical configuration/API.

The current transition configuration distinguishes `ergodic`, `semi`, and `left-to-right` modes. Preserve the meaning of each mode and verify any transition-mask or self-transition interaction against explicit-duration semantics.

For probability/logit code:

- normalize on the correct event dimension;
- distinguish logits, log-probabilities, probabilities, parameters, and samples explicitly;
- keep supports and parameter constraints valid;
- avoid `log(0)` and invalid covariance/scale parameters;
- prefer numerically stable `logsumexp`, `log_softmax`, and equivalent formulations;
- treat `EPS`, `NEG_INF`, dtype, and masking consistently;
- do not replace impossible states with finite values unless the approximation is intentional and documented.

For custom distributions, verify that `batch_shape`, `event_shape`, sampling, `log_prob`, and differentiability agree both with PyTorch semantics and NHSMM callers.

Do not claim a distribution is reparameterizable merely because it exposes `rsample`; verify the actual pathwise-gradient semantics.

## Context and encoder semantics

Context is part of the probabilistic parameterization, not an independent feature subsystem.

Preserve clear distinctions between:

- observed sequences;
- masks and sequence lengths;
- per-timestep context;
- canonical/global context;
- encoder outputs;
- distribution parameters derived from context.

When changing encoder behavior, inspect `ContextEncoder`, `ContextRouter`, `SequenceSet`, model initialization, and every distribution consuming context.

Do not allow padding values to become model information accidentally. Mask padded timesteps wherever they can affect likelihoods, context aggregation, loss, decoding, or metrics.

Encoder dimensionality is a model contract. If `context_dim`, `hidden_dim`, projection behavior, pooling, or encoder outputs change, propagate the change through `ModelConfig`, distribution construction, context routing, and tests.

## Temporal and causal integrity

NHSMM is domain-neutral but is used for time-series and regime modeling. Preserve causal validity whenever an operation is intended for online, forecasting, or trading use.

Never introduce:

- look-ahead bias;
- future-data contamination;
- target leakage;
- preprocessing fitted on validation/test periods;
- context encoders that consume future observations while an API is presented as causal/online;
- validation/test information in initialization or model selection;
- hidden train/evaluation contamination.

Bidirectional or full-sequence inference is valid for retrospective smoothing when explicitly identified as such. Never describe smoothed latent-state estimates as causal filtered estimates.

Keep filtering, smoothing, decoding, forecasting, and training semantics distinct.

## Training and optimization

Training code must optimize a clearly defined objective. Do not add heuristic loss terms without stating their statistical or operational purpose.

When modifying training:

- preserve separation between model parameters, initialization, optimizer state, scheduler state, and convergence logic;
- check gradient flow through encoder and neural distribution parameters;
- verify whether initialization occurs once or per restart;
- honor configured random seeds where deterministic behavior is expected;
- do not leak best-run state across independent initializations;
- distinguish convergence criteria, early stopping, and scheduler behavior;
- avoid retaining computation graphs unintentionally across iterations.

If an optimization change alters the statistical estimator, state that explicitly rather than treating it as an implementation-only refactor.

## Numerical behavior

The library must remain robust on CPU and CUDA where supported.

Check:

- dtype consistency;
- device placement;
- zero-length and short sequences where APIs permit them;
- very negative log-probabilities;
- degenerate or near-degenerate variances/scales;
- extreme logits and temperatures;
- maximum-duration boundaries;
- NaN/Inf propagation;
- batched versus single-sequence behavior.

Do not introduce CPU tensors into CUDA computation paths implicitly. Construct tensors from existing tensors or specify device/dtype explicitly where required.

## Configuration and API evolution

Prefer one canonical representation over aliases or duplicate configuration paths. Do not retain obsolete semantics through compatibility shims unless backward compatibility is explicitly required.

When an API changes intentionally:

1. update its implementation;
2. update direct callers;
3. update scripts and tests that exercise it;
4. update public imports where relevant;
5. update documentation/examples;
6. remove obsolete examples and aliases rather than leaving contradictory usage paths.

Do not preserve stale APIs merely because old scripts or tests still import them. First determine whether the old consumer or the new implementation reflects the intended current design.

## Testing and verification

Use the smallest verification set that materially validates the change, then expand when necessary.

Typical development checks are:

```bash
pytest -v
ruff check nhsmm tests scripts
black --check nhsmm tests scripts
```

Run targeted tests first for localized changes. Run the broader suite for changes to shared inference, distributions, context, configuration, or training behavior.

For numerical/model changes, prefer invariants over fragile exact values where appropriate. Useful invariants include:

- probabilities sum to one on the intended axis;
- log-probabilities are finite where support permits;
- masks exclude padded positions;
- output shapes match documented contracts;
- gradients exist where expected;
- decoded states/durations remain in range;
- batched and equivalent unbatched computations agree within tolerance;
- CPU/CUDA behavior is consistent where both are supported.

A stale test that imports removed APIs is evidence of repository drift, not evidence that the removed API must be restored. Resolve the intended contract before changing implementation.

Never claim code was executed, tested, benchmarked, trained, or empirically validated unless that action actually ran and its result was observed.

## Research and benchmarks

Separate implementation verification from empirical validation.

A passing unit test does not establish model quality. A single OHLCV run does not establish regime-discovery quality or trading usefulness.

Report empirical results with enough information to reproduce them, including dataset, chronological split, seed, hyperparameters, metric definition, and evaluation protocol.

For financial experiments:

- preserve chronological evaluation;
- keep model selection inside the training/validation process;
- do not tune a supposedly unsupervised representation against downstream PnL without explicitly treating that as supervised model selection;
- separate feature construction, fit, model selection, inference, and downstream trading evaluation temporally.

Avoid conclusions based solely on latent-state plots. Where relevant, evaluate occupancy, duration distributions, transition behavior, likelihood/generalization, seed stability, and downstream out-of-sample behavior.

## Implementation discipline

Prefer the smallest coherent change set.

Before editing:

1. inspect the target file;
2. inspect material dependencies;
3. identify the contract being changed;
4. determine whether tests/docs/configuration must change with it;
5. implement one canonical behavior.

Do not perform broad unrelated refactors as part of a focused change.

Avoid unnecessary abstraction. Introduce new layers only when they provide a concrete mathematical, statistical, architectural, or operational benefit.

Delete obsolete paths when an intentional contract change makes them invalid; do not accumulate duplicate implementations merely to preserve outdated internal behavior.

Keep comments focused on non-obvious mathematical or architectural intent. Do not use comments to restate straightforward Python.

## Repository workflow

Use `develop` as the default target branch unless the user explicitly specifies another branch or repository state establishes a newer canonical workflow.

### Preferred local workspace workflow

For implementation, debugging, review, and targeted verification, prefer working from a fresh local repository checkout in a temporary workspace such as `/tmp/nhsmm` when the execution environment can access the repository.

Default sequence:

1. clone or refresh `awa-si/nhsmm` into a temporary local workspace;
2. check out the current `develop` branch and verify the base commit;
3. read the current target files and material dependencies from that checkout;
4. make edits locally using normal patch/edit tools;
5. run the smallest useful local smoke/unit test set before broader validation;
6. inspect `git diff`, `git status`, and the exact changed files;
7. create a focused local commit only after the local diff is coherent;
8. publish that exact change to GitHub and verify the resulting remote commit/diff.

Prefer local/container execution for cheap, fast functional checks such as imports, smoke tests, focused `pytest` targets, shape/invariant checks, and deterministic CPU tests. Do not spend GitHub Actions capacity on every small iteration when equivalent local verification is available.

Use GitHub Actions when it provides material additional evidence, including runner-specific behavior, clean-environment packaging, cross-environment reproducibility, CI integration, artifact capture, or longer experiments that should be independently reproducible. A local test and a GitHub Actions run are different forms of evidence; do not claim one was performed when only the other ran.

If direct `git clone`, network access, authentication, or push access is unavailable in the local execution environment, fall back to the connected GitHub tools without blocking the task. Preserve the same read-before-write, minimal-diff, targeted-local-verification-where-possible, and post-change verification discipline.

### Preferred edit workflow: GitHub Patch

For repository modifications, **GitHub Patch is the preferred remote edit workflow** when the plugin/skill is available.

Use GitHub Patch for:

- precise file edits;
- unified diffs;
- file creation, renaming, and deletion;
- atomic multi-file changes;
- dry-run previews when useful;
- post-commit diff verification;
- CI/status inspection when materially relevant;
- safe rollback commits.

Default GitHub Patch behavior:

- repository: `awa-si/nhsmm`;
- branch: `develop`;
- commit directly to the selected branch unless the user requests a PR;
- generate a concise factual commit message unless one is supplied;
- verify the resulting commit and changed files;
- do not create a pull request unless explicitly requested or direct writing is blocked and an authorized PR is the fallback;
- never force-push or overwrite newer repository state with stale content.

If GitHub Patch is unavailable, use the closest precise GitHub edit workflow available while preserving the same read-before-write, minimal-diff, direct-branch, and post-change verification discipline.

Do not use GitHub Actions or workflow-file changes merely as an editing mechanism. CI exists for validation, not for modifying source or documentation state.

## Documentation

Documentation must describe actual current behavior, not intended behavior that the code does not implement.

When model semantics or public APIs change, update the relevant documentation in the same coherent change when practical.

Use precise terminology:

- HSMM vs HMM;
- duration vs transition;
- filtering vs smoothing;
- likelihood vs loss;
- probability vs log-probability vs logits;
- latent state vs observed feature;
- context vs observation;
- training/validation/test roles.

Do not make unsupported claims such as `production-ready`, `memory-efficient`, `scalable`, `causal`, or `GPU optimized` without implementation or benchmark evidence.

Examples must use the actual exported API. Before publishing an example, verify its imports and constructor signatures against the current package.

## Known repository drift

Historical scripts, tests, and documentation may contain API names or constructor patterns from earlier NHSMM revisions. Examples observed in repository history include names such as `HSMM`, `NeuralHSMM`, and `GaussianHSMM`, while the current public package export is `NHSMM` with `ModelConfig`-driven construction.

Do not automatically restore historical APIs to satisfy stale consumers. Determine the intended current architecture, then migrate or remove stale consumers consistently.

## Completion standard

A task is complete only when the requested behavior is implemented coherently across affected contracts and the resulting repository state has been inspected.

Report separately:

- what changed;
- what was actually verified;
- any remaining unverified assumptions or known inconsistencies.
