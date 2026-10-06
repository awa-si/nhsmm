# Configuration contract

All public configuration types are defined in `nhsmm/config.py`. Validation modules may re-export them, but configuration ownership is centralized.

NHSMM separates model/training configuration from validation policy.

## ModelConfig

`ModelConfig` is the runtime/training contract used by `NHSMM`.

It remains the single source of truth for model architecture, optional context-encoder use, context dimensions and causality, distribution and initialization choices, duration/transition capacity, optimizer/convergence policy, seed, and training budget.

The contract is strict:

```python
from nhsmm import ModelConfig

config = ModelConfig.from_dict({
    "n_states": 3,
    "n_features": 8,
    "max_duration": 40,
    "lr": 0.005,
    "max_iter": 80,
})

tuned = config.with_overrides(
    lr=0.002,
    max_iter=120,
)

payload = tuned.to_dict()
```

Unknown keys are rejected. Overrides return a newly validated config and never silently ignore misspelled tuning parameters.

`use_context_encoder=False` is the default. In that mode `NHSMM` does not construct a `ContextEncoder`. Set `use_context_encoder=True` to opt into learned observation-derived context. External context is independent of this switch: configure `context_dim` and pass context explicitly when needed.

## ValidationConfig

`ValidationConfig` is a separate public contract for controlled learning/prediction validation. Validation policy is intentionally not stored in `ModelConfig`.

The contract is versioned with `schema_version=1` and contains:

- `model`: a complete `ModelConfig` template;
- `data`: synthetic acceptance dataset dimensions and duration-generation policy;
- `seeds`: reproducibility seeds;
- `boundary_tolerance`: tolerated boundary timing error in timesteps;
- `health`: model-health/collapse thresholds used by direct diagnostics and validation snapshots;
- `scenarios`: named signal regimes and their learning/usefulness acceptance thresholds.

Example:

```python
from nhsmm import ValidationConfig

config = ValidationConfig().with_overrides(
    model={
        "max_iter": 60,
        "lr": 0.005,
    },
    data={
        "steps": 240,
        "n_train": 12,
    },
    boundary_tolerance=2,
    health={
        "max_state_occupancy": 0.85,
    },
    scenarios={
        "weak": {
            "min_median_boundary_f1_tolerance": 0.50,
            "max_collapse_fraction": 0.25,
        },
    },
)

config.to_json("validation.json")
```

A saved JSON contract can be loaded with:

```python
config = ValidationConfig.from_json("validation.json")
```

Unknown fields, invalid value ranges, incompatible dimensions, unsupported schema versions, and contradictory scenario policies fail closed.

## Scenario policy

Each `ValidationScenarioConfig` defines signal separation plus explicit acceptance policy. No scenario name has hidden semantics.

Learning thresholds can include minimum OOS state accuracy, accuracy gain, held-out likelihood gain, maximum null OOS accuracy, and maximum absolute null ARI.

Usefulness thresholds can include exact boundary F1, tolerated-boundary F1, run-length relative error, transition-matrix MAE, occupancy L1 error, posterior ECE, and collapse policy.

`usefulness_required=False` explicitly marks a negative control such as the null-signal scenario.

## CLI tuning

The maintained acceptance harness accepts either a complete JSON config or strict dotted overrides:

```bash
python scripts/validate_learning_prediction.py \
  --config validation.json \
  --scenario moderate \
  --set model.max_iter=80 \
  --set model.lr=0.003 \
  --set data.steps=240 \
  --set health.max_state_occupancy=0.85 \
  --set boundary_tolerance=2 \
  --set scenario.moderate.max_median_ece=0.15 \
  --output /tmp/validation.json
```

Values after `=` are parsed as JSON when possible, so numbers, booleans, `null`, arrays, and quoted strings retain their types.

To materialize the fully resolved contract:

```bash
python scripts/validate_learning_prediction.py \
  --set model.max_iter=80 \
  --dump-config /tmp/resolved-validation-config.json \
  --scenario moderate
```

The output report embeds the full resolved `validation_config`, making each acceptance result reproducible from its exact configuration.


## Configuration tuner

NHSMM exposes a domain-neutral tuning interface around the validated configuration contracts.

The tuner does not own training or validation. It materializes validated candidate configs and delegates evaluation to a callback. This keeps search logic separate from NHSMM fitting, walk-forward validation, and downstream trading logic.

### Public API

```python
from nhsmm import (
    Choice,
    ConfigTuner,
    FloatRange,
    IntRange,
    ModelConfig,
    TuneEvaluation,
)
```

Supported domains:

- `Choice([...])`
- `IntRange(low, high, step=...)`
- `FloatRange(low, high, steps=..., log=False)`

Supported search strategies:

- deterministic Cartesian `grid`
- reproducible seeded `random`

Supported objectives:

- `direction="maximize"`
- `direction="minimize"`

### ModelConfig example

```python
from nhsmm import ConfigTuner, FloatRange, IntRange, ModelConfig

base = ModelConfig(
    n_states=3,
    n_features=8,
)

tuner = ConfigTuner(
    base,
    {
        "lr": FloatRange(1e-4, 1e-2, steps=5, log=True),
        "max_iter": IntRange(40, 120, step=40),
    },
    direction="maximize",
    seed=17,
)

def evaluate(config):
    # Fit/evaluate using your own train/OOS pipeline.
    return held_out_score(config)

report = tuner.run(evaluate, strategy="grid")
best_config = report.best.config
```

For `ModelConfig`, paths may be written as either `lr` or `model.lr`.

### ValidationConfig example

```python
from nhsmm import Choice, ConfigTuner, FloatRange, ValidationConfig

base = ValidationConfig()

tuner = ConfigTuner(
    base,
    {
        "model.lr": FloatRange(1e-4, 1e-2, steps=5, log=True),
        "model.max_iter": Choice([40, 60, 80]),
        "health.max_state_occupancy": Choice([0.75, 0.80, 0.85]),
        "boundary_tolerance": Choice([1, 2]),
        "scenario.moderate.max_median_ece": Choice([0.10, 0.15, 0.20]),
    },
    direction="maximize",
)

report = tuner.run(
    evaluate_validation_config,
    strategy="random",
    trials=20,
)
```

Supported `ValidationConfig` paths are:

- `model.<field>`
- `data.<field>`
- `health.<field>`
- `scenario.<name>.<field>`
- `boundary_tolerance`
- `seeds`

All overrides pass through the same strict configuration validators. Invalid or misspelled fields fail closed.

### Evaluation result

An evaluation callback can return a scalar:

```python
def evaluate(config):
    return 0.83
```

or a score plus diagnostics:

```python
from nhsmm import TuneEvaluation

def evaluate(config):
    return TuneEvaluation(
        score=0.83,
        metrics={
            "oos_accuracy": 0.91,
            "boundary_f1": 0.78,
            "ece": 0.04,
        },
    )
```

The score determines ranking. Metrics are retained as evidence and do not implicitly affect ranking.

### Reproducibility

Random tuning is deterministic for the same:

- base config;
- search space;
- tuner seed;
- trial count.

Every `TuneTrial` stores:

- trial index;
- exact overrides;
- fully resolved validated config;
- objective score;
- diagnostic metrics.

A complete report can be persisted:

```python
report.to_json("/tmp/nhsmm-tuning.json")
```

The report contains the best trial plus every evaluated candidate, so a selected configuration can be audited and reproduced.

### Design boundary

`ConfigTuner` deliberately does not:

- train an NHSMM by itself;
- choose a trading objective;
- perform walk-forward splitting;
- perform early stopping across external trials;
- depend on Optuna, Ray, or another tuning framework.

Those concerns can be layered above the interface. The tuner owns only validated configuration search and reproducible result selection.
