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

### Optimization budgets and structural overrides

`optimizer_batch_size=None` retains a full sequence batch. A positive integer
enables gradient accumulation over whole independent sequences. Sequence
likelihood gradients are summed without averaging; the global loss-bias penalty
is applied once. Gradient clipping and Adam stepping occur once per complete
training batch. Fixed-target monitoring/restart scoring uses the same chunk
bound. Uneven final chunks, variable lengths and aligned external context are
preserved. Chunking does not truncate episodes or change censoring.
Loss/gradient/update equivalence is tested in deterministic float64 plain models
and float32 external/neural-context models. Floating-point reduction ordering
may differ; dropout draws can differ with chunking, so stochastic trajectories
are not promised to be bit-identical. Defaults remain unchunked.

`optimizer_temperature_mode="annealed"` retains the update-temperature schedule
from target to0.3 times target. `"constant"` uses the target for every update.
`optimizer_target_temperature=1.0` is the default; values must be finite and
strictly positive. Temperature changes distribution probabilities, not only
optimization step sizes.

Convergence/scheduler monitoring and restart selection now score the fixed
target objective in eval mode, independently of update annealing/dropout.
`_optimization_history` separates pre-update scheduled and target likelihoods;
`_restart_scores` score final post-update snapshots at the recorded
`_restart_score_temperature`. At target1 these match public eval likelihood.
The fixed-target monitoring adds one no-gradient forward pass per update.
This corrects scoring semantics; it can change stopping and restart selection.

`optimizer_initialization="restart"` retains independent fresh initialization.
Opt-in `"current"` requires `n_init=1` and preserves prepared distributions,
parameter values and trainability flags. It supports controlled frozen-nuisance
tests without claiming production/statistical acceptance. Configs are validated
again before optimization, including mutated or override objects.

`max_iter` is the maximum number of updates per restart. With `convergence_stop=True`, a detected plateau can end a run early; with `convergence_stop=False`, a successful run performs the full configured budget. Convergence monitoring remains active independently of stopping.

An `optimize(..., cfg=...)` override must remain structurally compatible with the constructed model, including `duration_tail`. Changing finite support to D+ or back requires constructing the corresponding model.

### Gaussian emission initialization

`emission_init_mode="kmeans_cluster_variance"` initializes Gaussian means from training-observation kmeans and actual variances from within-cluster population variances. Targets are clipped to [0.05, 5] in training-observation units; clusters with fewer than two members use the global population variance. Gaussian variance is `softplus(log_var)`, so this initializer stores the stable inverse-softplus of each target. It requires Gaussian emissions. The package default remains `spread`; the existing means-only `kmeans` mode retains its scale initialization.

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


## Configuration updates and experiment ownership

V3 removes the package's custom ConfigTuner, search domains and Tune* reports.
The removed exports are Choice, IntRange, FloatRange, ConfigTuner,
TuneEvaluation, TuneTrial, TuneReport and apply_config_overrides. There is no
compatibility shim or Qlib dependency in the model package.

Use the existing strict configuration methods:

```python
from nhsmm import ModelConfig, ValidationConfig

base = ModelConfig(n_states=3, n_features=2)
candidate = base.with_overrides(lr=0.003, max_iter=80)

policy = ValidationConfig().with_overrides(
    model={"lr": 0.003, "max_iter": 80},
    data={"steps": 240},
    health={"max_state_occupancy": 0.85},
    scenarios={"moderate": {"max_median_ece": 0.15}},
    boundary_tolerance=2,
    seeds=(1, 2, 3),
)
```

Both methods return validated copies and reject unknown fields. The synthetic
validation CLI retains its documented --set paths; parsing those command-line
paths is local to the script and does not require a core search framework.

Integrated experiment orchestration is owned by
[awa-si/nautilus adapter/qlib](https://github.com/awa-si/nautilus/tree/main/adapter/qlib):
declare Qlib model/dataset/record tasks, execute them with native TrainerR,
and inspect the resulting Recorder artifacts/metrics. Explicit candidate lists
may use standard-library itertools.product or a separately seeded random
generator. No custom tuner or automatic winner selector is added to the host.

NHSMM still owns optimize, configuration validation, artifact IO and general
model-health/validation snapshots. Standalone synthetic mathematical acceptance
tests remain here. Qlib owns task execution and tracking; chronological folds,
train-only preprocessing, trusted source/data identities and the validation-only
selection policy remain caller responsibilities. Reserve test for the
preregistered final evaluation.

See [slimming evidence](validation/v3-qlib-slimming-20261015.md) for the migration.
