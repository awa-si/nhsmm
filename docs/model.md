# NHSMM model and runtime contract

This document describes the core model/runtime boundary of `nhsmm`.

## Model

The NHSMM separates four probabilistic components:

1. initial-state distribution;
2. transition distribution;
3. duration distribution;
4. emission distribution.

An optional, explicitly enabled context encoder can condition these components on sequence context. The default model has no internal encoder.

```text
observations
    |
optional context encoder
    |
context
    |
initial / transition / duration / emission
    |
HSMM inference
```

## Configuration

`ModelConfig` is the main configuration contract.

Typical construction:

```python
from nhsmm import ModelConfig, NHSMM

config = ModelConfig(
    n_states=3,
    n_features=5,
    max_duration=35,
    causal=True,
)

model = NHSMM(config=config)
model.initialize_distributions()

# use_context_encoder=False by default
```

`NHSMM` owns a resolved copy of the supplied `ModelConfig`; constructing a model, including construction with an explicitly injected encoder, does not mutate the caller's reusable config object.

A model intended for streaming use must be causal.

## Training ownership

NHSMM.fit(X, context=None, cfg=None) initializes distributions when needed and
delegates to the public optimize method; both return the same model. Qlib's
NHSMMModel calls fit. Optimization orchestration lives in nhsmm/training.py:
optimizer/scheduler construction, independent restarts, convergence, score/history
ordering and selection of the final best restart. Model-specific initialization,
objective evaluation, chunking, parameter snapshots and optional transition
refinement remain model helpers. Configuration and actual update telemetry remain
public through ModelConfig and optimization_summary.

Convergence owns scalar score/delta bookkeeping, stopping predicates and optional
LR-aware patience; the training loop owns the actual convergence_stop decision.
The unused second stop_training flag, callback execution/locks and JSON export
are removed. Research persistence belongs to the host recorder/MLflow boundary.

The move preserves the existing update protocol: history entries contain
pre-update scores, completed_updates counts the preceding main Adam updates,
and each entry is followed by one clipped/projected update. Final restart scoring
happens after that run's last update; selection restores the winning snapshot
when multiple restarts run. Optional transition refinement follows selection.
Existing histories/optimizer diagnostics, subclass hooks, caller train/eval mode,
config defaults and artifact/state_dict keys retain their contracts. Exceptions
restore mode but do not roll back completed parameter updates.

### Completed optimization telemetry

`model.optimization_summary()` returns an independent dictionary after a
successful optimize call. It reports observed main Adam updates per restart and
in total, the selected restart, fixed restart-score temperature, post-main-update
restart scores and completed optional transition-refinement updates. A new
optimization attempt invalidates the previous summary before changing training
state; a partial/failed fit publishes no completed summary. Main-update counters
increment after optimizer.step succeeds. Refinement counts are published only
after the refinement completes. The summary is not part of inference artifacts
or state_dict, and changing the returned dictionary cannot mutate model telemetry.

## Duration-tail training

The optional D+ tail stores its per-state end hazard directly as a probability. Joint optimization projects this parameter into [EPS, 1] after each optimizer update, before restart scoring and snapshots. This prevents an update beyond the feasible domain from entering the zero-gradient region of the inference clamp. The exact h_tail=1 finite-support boundary and artifact parameter representation are preserved. The inference clamp remains a guard for externally supplied parameters.

## Public runtime API

The supported streaming types are exported from the package root:

```python
from nhsmm import (
    HSMMFilterRuntime,
    HSMMFilterState,
    HSMMRuntimeState,
)
```

Avoid importing runtime/filter internals when the package-root contract is sufficient.

## HSMMFilterState

`HSMMFilterState.log_posterior` has shape:

```text
[B, K, D]
```

where:

- `B` = batch size;
- `K` = latent states;
- `D` = episode-age/duration support.

Semantics:

```text
log_posterior[b, k, a]
    = log P(z_t = k, age_t = a + 1 | x_0:t)
```

Convenience properties:

- `state_log_posterior` -> `[B,K]`;
- `state_posterior` -> `[B,K]`;
- `age_posterior` -> `[B,D]`.

The age posterior is not a duration forecast.

## HSMMFilterRuntime

`HSMMFilterRuntime` provides stateful causal filtering.

```python
runtime = HSMMFilterRuntime(model)

state = runtime.step(
    observation,
    context=None,
    timestamp=None,
)
```

`step()` returns an `HSMMFilterState`.

Accepted one-step observation shapes are:

```text
[F]
[B,F]
[B,1,F]
```

The feature dimension must equal `model.config.n_features`.

## Runtime invariants

### Causal model

The runtime requires `ModelConfig.causal=True`.

### Eval mode

The model must be in eval mode before causal filtering.

### Initialized distributions

Model distributions must already be initialized/loaded.

### Fixed batch size

After the first step, streaming batch size cannot change without `runtime.reset()`.

### Timestamp mode

A runtime session is either timestamped or untimestamped.

If the first step includes a timestamp:

- all later steps must include timestamps;
- timestamps must be strictly increasing.

If the first step omits a timestamp, timestamps cannot be enabled later without reset.

### Context mode

A runtime session uses either:

- the model's no-external-context path, which may be encoderless or use an opted-in internal encoder; or
- external caller-provided context.

The external-vs-non-external mode is fixed on the first successful step and cannot change until `runtime.reset()`. A rejected initialization step leaves the mode unset.

## Causal Float32 recursion

The causal batch forward recursion propagates a normalized `(state, age)` frontier at each timestep and carries cumulative log evidence separately. This preserves the public unnormalized `alpha`/log-likelihood semantics while avoiding repeated propagation of a growing sequence-level log offset in Float32. Batch forward, model-bound filtering and runtime share one internal initialization and state-age update, returning a normalized log posterior plus incremental observation log evidence. The batch path keeps its precomputed duration hazards and active-length masks; streaming supplies the previously available boundary scores. The public `filter_step()` still validates and normalizes arbitrary caller-provided scores. Internal evidence assumes canonical normalized priors and boundary distributions; the public initializer retains its score-input contract and does not expose the internal normalizer. The shared implementation is checked against independent path enumeration, not merely by comparing its callers.

## No internal context — default

With `use_context_encoder=False` and no external context, observations are scored directly by the emission model and initial/duration/transition use their context-free parameters. No `ContextEncoder` is constructed. The canonical causal runtime remains bounded and retains only the latest accepted observation plus HSMM filter state.

## Internal context — opt-in

Set `use_context_encoder=True` in `ModelConfig` when learned observation-derived context is required.

With no external context:

```python
state = runtime.step(observation)
```

the causal model derives context through its configured encoder/runtime path. The initial prior uses zero context; emissions use the previous encoder row. Boundary duration/transition propagation preserves its existing previous-row timing. Batch and incremental streaming use the same information contract.

When the configured encoder exposes incremental streaming state, the runtime keeps bounded encoder state. Custom causal encoders without that contract may use the correctness-first retained-prefix fallback.

## Predictive neural likelihood

For causal internal context, c_t summarizes observations through t. The initial-state prior uses c_-1 = 0, independent of the first observation. Emission parameters for x_t use c_(t-1), excluding x_t. Boundary propagation from t-1 to t also uses c_(t-1); duration/transition contexts are not shifted again. Runtime stores c_t after accepting x_t for the next step. Custom causal encoder prefix fallback follows the same batch timing.

This removes emission self-conditioning. The former normalization counterexample now integrates to 1.0 on both maintained quadrature grids. This is bounded regression evidence, not proof for arbitrary custom distributions. External context is conditional on covariates held fixed while observations vary; their information content is the caller's responsibility.

Noncausal internal encoders retain retrospective forward/decode scores. Public generative log_likelihood and optimize reject them when no external context is supplied: lagging a bidirectional encoder does not remove future dependence. Encoderless noncausal models remain supported.

Artifacts use version 3 and declare both internal_context_timing and likelihood_semantics (causal_prefix or conditional_completed). Legacy v2 causal artifacts and v1 encoderless causal artifacts remain loadable. Legacy noncausal artifacts are rejected with retraining guidance because their unnormalized objective cannot be silently reinterpreted; v1 encoder artifacts also retain the timing rejection. Raw state_dict files carry no timing version: callers must establish provenance and retrain old encoder weights before predictive use. See [repair evidence](https://github.com/awa-si/nautilus/tree/main/docs/ml-stack/nhsmm/evaluation.md#predictive-neural-repair--2026-10-12).

## External context

External context is supplied explicitly:

```python
state = runtime.step(
    observation,
    context=context,
)
```

External context requires `ModelConfig.context_dim`; its dimensionality must equal `model.context_dim`. An internal encoder is not required. Variable-length external-context lists are validated against `context_dim` independently of the observation feature width.

External context accepts [H], shared [T,H], batch-static [B,1,H],
[B,T,H], or one [T_i,H] tensor per independent observation sequence. Ragged lists
are checked against the observation mask's original lengths, then padded with
zero context. Model construction prepares observations once and passes the same
validated mask to context alignment. Padding contributes no likelihood or context
gradient. Empty external-context sequences retain canonical shape [B,1,H] with
zero canonical context and neutral empty likelihood; this does not change positive-
length context timing.

External context remains the caller's responsibility for every step in that runtime session.

## Runtime reset

```python
runtime.reset()
```

Reset clears:

- filter/runtime state;
- context-mode lock;
- timestamp/session history.

It does not reinitialize model parameters.

Use reset when intentionally starting a new independent stream or changing runtime mode.

## Forecasts

After at least one observation has been processed:

```python
survival = runtime.forecast_survival(horizons)
transition = runtime.forecast_transition()
```

These forecasts use only the runtime's current information set.

They are separate from `HSMMFilterState.age_posterior`.

## Causal semantics

The streaming runtime is a filtering path, not a retrospective smoother.

Future observations are not consumed.

The filter tracks the posterior over current latent state and current episode age. State propagation follows explicit-duration HSMM boundary semantics; the current observation enters through the causal emission/context path for the current step.

Retrospective/non-causal model methods are distinct APIs and should not be interpreted as live estimates.

## Artifacts

Artifact helpers are exported publicly:

```python
from nhsmm import (
    build_artifact,
    load_artifact,
    save_artifact,
)
```

Inference helpers:

```python
from nhsmm import (
    load_inference_model,
    prepare_inference,
)
```

A loaded model intended for streaming must satisfy the same causal/runtime invariants.

`build_artifact()` returns an independent CPU tensor snapshot. Subsequent model parameter updates do not mutate the built artifact.

## Integration boundary

The NHSMM core intentionally stops at model/runtime outputs.

Framework/domain mapping belongs in the maintained host repository.

```text
external event
    |
host adapter
    |
nhsmm public runtime API
    |
HSMMFilterRuntime / HSMMFilterState
    |
downstream representation
```

The dependency direction is one-way: host adapters may depend on the public `nhsmm` package API; `nhsmm` must not depend on host/framework packages. For NautilusTrader, the active adapter owner is `awa-si/nautilus@main/adapter/nhsmm`.

Microsoft Qlib research mapping is implemented in the maintained host's
[`adapter/qlib`](https://github.com/awa-si/nautilus/tree/main/adapter/qlib)
main branch. It consumes trusted causal artifacts and the public streaming
runtime, isolates instrument state, and builds real Qlib datasets with explicit
label-maturity purging. Fit-cutoff/observation-schema provenance and walk-forward
state alignment remain caller obligations. Qlib is an optional host dependency;
it does not replace NHSMM inference or become a model-core dependency.

The core repository should not absorb:

- Nautilus/Freqtrade strategy semantics;
- trading execution/risk policy;
- research workflow/business policy;
- medical decisions;
- raw-document ingestion/OCR pipelines.

## Validation boundary

Context-effect validation now lives in Nautilus adapter.qlib.nhsmm_validation, separate from core training and runtime integration.

See [validation.md](https://github.com/awa-si/nautilus/tree/main/docs/ml-stack/nhsmm/evaluation.md).

## Related documentation

## Distribution numerical boundaries

Continuous Gaussian/Student-t locations are finite real values, not categorical logits; no categorical score cap is applied to them. Independent Student-t parameters are broadcast before statistics and event reductions. Categorical labels broadcast across sample/batch dimensions; exact zero probabilities retain finite gradients for observed positive-probability classes.

Initial, duration and transition hard masks use exact negative infinity. Categorical logit construction accepts partial impossible support but rejects NaN, positive infinity and rows with no finite mass. Safe survival and sum reductions preserve impossible mass without undefined all-impossible backward paths. An impossible observed event can still have negative-infinite likelihood; numerical safety does not make it possible.




## Empty inputs, support errors and model dtype

For supported generative likelihood modes, an empty observation sequence has
probability one and log-likelihood zero. Empty entries are neutral in summed
training scores; a completely empty training dataset is rejected before any
restart initialization or optimizer update. An empty batch returns an empty
likelihood tensor, with reduced sum zero. Full forward returns [B,0,K,D] for
zero timesteps. Viterbi prediction returns one empty path per sequence;
predict(mode="log_likelihood") always returns the likelihood tensor. Invalid
prediction modes and unsupported causal timestep arguments are still rejected.

Causal forward checks finite, positive evidence before normalization and raises
ValueError when the active observations have no possible path. The public
likelihood wrapper raises on NaN instead of hiding it behind a finite sentinel.
Noncausal raw forward may be negative infinity when no complete path exists. Public likelihood raises ValueError when the context/length partition is zero; an impossible observation with a positive partition retains negative-infinite likelihood. Viterbi raises ValueError when no finite terminal MAP path exists. These checks do not assign mass to impossible events.

Observation conversion, distribution initialization and training restarts use
the live model parameter dtype/device. This includes Float64 models with an
internal encoder. Continuous segment emission sums retain their actual density
values and gradients instead of categorical score caps. Segment sums count impossible emission factors separately so earlier -inf values cannot poison later valid segments (`doc:nhsmm.segment-emissions`). Artifact loading preserves the homogeneous floating model dtype and rejects mixed floating state rather than silently casting.

Transition refinement requires explicit external context. The optimizer checks
that requirement before it initializes a restart or changes any weights.

## Filtering validation and convergence bookkeeping

Public HSMMFilterState construction requires normalized mass per batch item.
The absolute tolerance in log mass is max(1e-12, 8 * finfo(dtype).eps), with no
relative tolerance. HSMMFilterTrace validates valid rows by the same rule and
requires padding to contain only negative infinity. Empty batch/time dimensions
remain valid for traces; state and age dimensions must be positive.

A filter update with zero or non-finite observation evidence raises ValueError
before producing a state. Large finite emission offsets are normalized using
PyTorch log_softmax to avoid a residual caused by subtracting a large Float32 log
normalizer. These checks preserve impossible support; no replacement posterior is
invented. The public state container is frozen, but its Tensor storage remains
mutable: forecast APIs retain their defensive validation.

Convergence histories, score deltas and best scores use Float64 bookkeeping,
independent of model parameter dtype. Export produces strict JSON and represents
unobserved best scores and score-history entries as null. Plateau still uses its
exact rolling window; automatic patience applies to delta mode. The stop_training
request remains latched until reset.

ModelConfig.seed follows the PyTorch integer range [-2**63, 2**64 - 1], including
valid negative seeds. ValidationConfig seeds follow the validation runner's
legacy NumPy seed range [0, 2**32 - 1]; derived evaluation seeds must be
non-negative and may exceed that legacy range because evaluation generation uses
default_rng. Validation schema_version must be an integer excluding bool.
Scenario names must be non-empty strings; from_dict does not coerce or merge
different identifiers.

### Duration normalization gate

Duration soft_dmax is applied consistently by log_matrix, expected_probs, forward (scores or return_dist=True) and log_prob when supplied with the same arguments. The [D] or [K,D] gate sigmoid(soft_dmax).clamp_min(EPS) is applied after temperature and hard masks, before normalization. Helpers preserve their native batch/context shapes; log_matrix retains its padded shape and T expansion. The D+ tail hazard remains separate.


## Normalized noncausal likelihood and implementation boundaries

Noncausal public likelihood and training divide the completed-segmentation score by its context/length partition. Raw forward and MAP retain their segment-potential meaning. See [the mathematical owner](hsmm-mathematical-contract.md#normalized-observation-likelihood) for the generative definition; fixed external context is a conditioning variable.

The model class delegates recursions to `models/forward.py`, decoding to `models/decoding.py`, support-safe segment sums to `models/segments.py`, and observation objectives to `models/objective.py`. `models/boundary.py` owns the analytic first-order tail boundary VJP. No model or state-dict parameter paths change during this split. Higher-order differentiation at h_tail=1 is explicitly rejected; use interior parameters for higher-order work. Runtime and ordinary finite-support causal scoring keep their existing kernels.

See [correctness repair](correctness-repair.md) for bounded evidence and artifact migration.

## Explicit external component context

Use CausalContext(emission=E, boundary=U) for distinct pre-observation and outgoing-boundary information sets. Initial probabilities use E[0]. Plain Tensor timing remains compatible. See [the timing contract](causal-external-context.md).


## Explicit base-score context

BASE_CONTEXT bypasses context heads and the internal encoder for batch
likelihood/filtering or an entire streaming step. Within an external stream it
preserves the posterior and age; the prior boundary still propagates into the
current observation. None retains internal-mode semantics. Base scores use the
current parameters. See [timing and fallback](causal-external-context.md).

## Mixed external-context availability

CausalContext(emission=E, boundary=U, available=M) carries strict boolean row
availability through batch scoring, supervised/latent fitting and streaming.
Missing rows bypass neural corrections while observations and filtering history
remain intact. See [shapes, timing and limits](causal-external-context.md#mixed-availability-within-a-batch).
