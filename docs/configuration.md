# Configuration contract

ModelConfig is defined in nhsmm/config.py. ModelHealthThresholds and
ContextEvidenceConfig belong to Nautilus adapter.qlib.nhsmm_validation.

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

## Health and context evidence

Health thresholds and context-evidence policy now live in Nautilus
adapter.qlib.nhsmm_validation; they are no longer exported by NHSMM.
