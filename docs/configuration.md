# Configuration contract

NHSMM separates model/training configuration from validation policy.

## ModelConfig

`ModelConfig` is the runtime/training contract used by `NHSMM`.

It remains the single source of truth for model architecture, encoder dimensions and causality, distribution and initialization choices, duration/transition capacity, optimizer/convergence policy, seed, and training budget.

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
