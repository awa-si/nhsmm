# NHSMM Agent Instructions

This file defines the repository-specific operating rules for AI-assisted analysis, design, review, and implementation in `awa-si/nhsmm`.

## Repository

- Repository: `https://github.com/awa-si/nhsmm`
- Default development branch: `develop`
- Package: `nhsmm`
- Language/runtime: Python 3.9+
- Core framework: PyTorch
- Project type: neural probabilistic sequence-modeling library implementing Neural Hidden Semi-Markov Models (NHSMMs)
- Status: research/alpha-to-beta stage; APIs and internals may still evolve.

Treat the current repository state as authoritative. Do not infer APIs, tensor contracts, model semantics, configuration fields, test behavior, or supported functionality from README prose, old documentation, prior conversations, or generic HSMM implementations when the current code can be inspected.

Before modifying a file, read its current version. When a change depends on another model, distribution, encoder, configuration field, caller, consumer, tensor shape, persistence format, or public API, inspect that dependency rather than guessing.

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

Challenge incorrect assumptions directly. Distinguish verified implementation facts from proposals, assumptions, estimates, and unresolved questions.

## Model semantics

NHSMM explicitly models latent state dynamics with separate initial, transition, duration, and emission components. Preserve that separation unless an intentional architectural change requires otherwise.

The current model pipeline is conceptually:

`sequence -> context encoder -> initial / transition / duration / emission distributions -> HSMM inference -> objective / decoding / training`

Changes to one stage must be checked for consequences in every dependent stage.

Do not silently convert HSMM semantics into HMM semantics. Explicit-duration behavior, maximum duration, transition semantics, sequence boundaries, masks, and latent-state indexing are part of the model contract.

When introducing or changing inference equations:

- state the expected tensor dimensions;
- verify normalization axes;
- preserve batch and time semantics;
- preserve explicit duration semantics;
- handle variable-length sequences and masks correctly;
- avoid accidental probability-space operations where log-space computation is required;
- verify boundary conditions at sequence start/end and duration truncation;
- avoid off-by-one duration indexing.

Use conventional shape notation consistently:

- `B`: batch size
- `T`: sequence length
- `F`: observed feature dimension
- `C`: context dimension
- `K`: latent-state count
- `D`: maximum duration

Do not rely on broadcasting unless the intended broadcast dimensions are clear from the surrounding contract.

## Probabilistic correctness

Probability distributions must remain mathematically valid.

For probability/logit code:

- normalize on the correct event dimension;
- distinguish logits, log-probabilities, probabilities, parameters, and samples explicitly;
- keep supports and parameter constraints valid;
- avoid taking `log(0)` or creating invalid covariance/scale parameters;
- prefer stable `logsumexp`, `log_softmax`, and equivalent numerically stable formulations;
- treat `EPS`, `NEG_INF`, dtype, and masking semantics consistently;
- do not replace impossible states with finite values unless the approximation is deliberate and documented.

For Gaussian, Student-t, categorical, transition, initial, duration, or future distributions, verify that returned `batch_shape`, `event_shape`, sampling, `log_prob`, and differentiability semantics agree with PyTorch conventions and with the NHSMM callers.

Do not claim that a custom distribution is reparameterizable merely because its wrapper exposes `rsample`; verify that the implementation actually has the required pathwise-gradient semantics.

## Context and encoders

Context is part of the probabilistic parameterization, not an unrelated feature pipeline.

Preserve clear semantics between:

- observed sequences;
- masks and sequence lengths;
- per-timestep context;
- canonical/global context;
- encoder outputs;
- distribution parameters derived from context.

When changing encoder behavior, inspect `ContextEncoder`, `ContextRouter`, `SequenceSet`, model initialization, and all distributions consuming context.

Do not allow padding values to become model information accidentally. Mask padded timesteps in every computation where they can affect likelihoods, context aggregation, loss, decoding, or metrics.

## Temporal and causal integrity

NHSMM is domain-neutral, but it is used for time-series and regime modeling. Preserve causal validity whenever an operation is intended for online, forecasting, or trading use.

Do not introduce:

- look-ahead bias;
- future-data contamination;
- target leakage;
- fitting preprocessing statistics on validation/test periods;
- context encoders that consume future observations when an API is claimed to be causal/online;
- validation or test information in model selection or initialization;
- hidden train/evaluation contamination.

Bidirectional or full-sequence inference is valid when explicitly used for retrospective smoothing. Do not describe smoothed latent-state estimates as causal filtered estimates.

Keep filtering, smoothing, decoding, forecasting, and training semantics distinct.

## Training and optimization

Training code must optimize a clearly defined objective. Do not add heuristic loss terms without identifying their statistical or operational purpose.

When modifying training:

- preserve the separation between model parameters, initialization, optimizer state, and convergence logic;
- check gradient flow through neural distribution parameters and encoders;
- verify whether initialization happens once or per restart;
- honor configured random seeds where deterministic behavior is expected;
- do not leak best-run state across independent initializations;
- distinguish convergence criteria from early stopping and scheduler behavior;
- avoid retaining computation graphs unintentionally across iterations.

If an optimization method changes the statistical estimator, state that explicitly rather than treating it as an implementation-only refactor.

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
- NaN and Inf propagation;
- batched versus single-sequence behavior.

Do not introduce CPU tensors into CUDA computation paths implicitly. Construct tensors from existing tensors or with explicit device/dtype where required.

## Configuration and public contracts

`ModelConfig` is a central contract. When adding, removing, or changing a configuration field, inspect every constructor, default, caller, script, test, and documentation reference that depends on it.

Prefer one canonical representation over aliases or duplicate configuration paths. Do not retain obsolete semantics through compatibility shims unless backward compatibility is explicitly required.

Public imports exposed through package `__init__.py` files are API surface. Changes to them must be deliberate.

The current repository contains documentation that may lag implementation. Code and tests take precedence when determining actual behavior; documentation should then be corrected to match the intended canonical contract.

## Testing and verification

Use the smallest verification set that materially validates the change, then expand when necessary.

Typical development checks are:

```bash
pytest -v
ruff check nhsmm tests scripts
black --check nhsmm tests scripts
```

Run targeted tests first for localized changes. Run the broader suite for changes to shared inference, distribution, context, configuration, or training behavior.

For numerical/model changes, tests should prefer invariants over fragile exact values where appropriate. Useful invariants include:

- probabilities sum to one on the intended axis;
- log-probabilities are finite where support permits;
- masks exclude padded positions;
- output shapes match documented contracts;
- gradients exist where expected;
- decoded states and durations remain in range;
- batched and equivalent unbatched computations agree within tolerance;
- CPU/CUDA behavior is consistent when both are supported.

Never claim code was executed, tested, benchmarked, trained, or validated unless that action actually ran and its result was observed.

## Research and benchmarks

Separate implementation verification from empirical validation.

A passing unit test does not establish model quality. A single OHLCV run does not establish regime-discovery quality or trading usefulness. Report empirical results with the dataset, split, seed, hyperparameters, metric definition, and evaluation protocol required to reproduce them.

For financial experiments, use chronological evaluation and keep model selection strictly inside the training/validation process. Do not use downstream PnL to tune a supposedly unsupervised representation without explicitly treating that as supervised model selection.

Avoid conclusions based solely on visual latent-state plots. Evaluate occupancy, duration distributions, transition behavior, likelihood/generalization, stability across seeds, and task-relevant out-of-sample behavior when those questions matter.

## Implementation discipline

Prefer the smallest coherent change set.

Before editing:

1. inspect the target file;
2. inspect material dependencies;
3. identify the contract being changed;
4. determine whether tests/docs/configuration must change with it;
5. implement one canonical behavior.

Do not perform broad unrelated refactors as part of a focused change.

Avoid unnecessary abstraction. Introduce new layers only when they provide a concrete statistical, architectural, or operational benefit.

Delete obsolete paths when an intentional contract change makes them invalid; do not accumulate duplicate implementations merely to preserve outdated internal behavior.

Keep comments focused on non-obvious mathematical or architectural intent. Do not use comments to restate straightforward Python.

## Repository workflow

Use `develop` as the default target branch unless the user explicitly specifies another branch or repository state indicates a newer canonical workflow.

For repository modifications:

- prefer precise atomic edits;
- keep commits focused on one coherent change;
- use concise, factual commit messages;
- verify the resulting commit and changed files;
- inspect the final diff after modification;
- do not create a pull request unless explicitly requested or direct writing is unavailable and a PR is an authorized fallback;
- never force-push or overwrite newer changes with stale file content.

Do not use GitHub Actions or workflow-file changes merely as an editing mechanism. CI is for validation, not for modifying documentation or source state.

## Documentation

Documentation must describe actual behavior, not intended behavior that the code does not implement.

When model semantics change, update the relevant documentation in the same coherent change when practical.

Use precise terminology:

- HSMM vs HMM;
- duration vs transition;
- filtering vs smoothing;
- likelihood vs loss;
- probability vs log-probability vs logits;
- latent state vs observed feature;
- context vs observation;
- training/validation/test roles.

Do not make unsupported claims such as "production-ready", "memory-efficient", "scalable", "causal", or "GPU optimized" without implementation or benchmark evidence.

## Known repository consistency issues

At the time this file was introduced, some repository metadata/documentation was inconsistent with the implementation:

- the existing uppercase `AGENTS.md` describes a React/TypeScript/Vite documentation project rather than the current Python/PyTorch NHSMM library;
- `CONTRIBUTING.md` refers to branching from `main`, while the repository default branch is `develop`;
- README/package metadata may contain stage/version wording or capability claims that should be verified against current code before being relied upon.

Treat these as documentation inconsistencies, not authoritative architectural facts. Do not propagate them into new implementation work.

## Completion standard

A task is complete only when the requested behavior is implemented coherently across the affected contracts and the resulting repository state has been inspected.

Report separately:

- what changed;
- what was actually verified;
- any remaining unverified assumptions or known inconsistencies.
