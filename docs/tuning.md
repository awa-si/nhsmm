# Configuration tuner

NHSMM exposes a domain-neutral tuning interface around the validated configuration contracts.

The tuner does not own training or validation. It materializes validated candidate configs and delegates evaluation to a callback. This keeps search logic separate from NHSMM fitting, walk-forward validation, and downstream trading logic.

## Public API

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

## ModelConfig example

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

## ValidationConfig example

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

## Evaluation result

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

## Reproducibility

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

## Design boundary

`ConfigTuner` deliberately does not:

- train an NHSMM by itself;
- choose a trading objective;
- perform walk-forward splitting;
- perform early stopping across external trials;
- depend on Optuna, Ray, or another tuning framework.

Those concerns can be layered above the interface. The tuner owns only validated configuration search and reproducible result selection.
