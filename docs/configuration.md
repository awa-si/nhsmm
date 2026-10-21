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
from target toward temperature_min_ratio times target (default 0.3). `"constant"` uses the target for every update.
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

`emission_init_mode="kmeans_cluster_variance"` initializes Gaussian means from training-observation kmeans and actual variances from within-cluster population variances. Targets are clipped to configured cluster_variance_min/max (defaults [0.05, 5]) in training-observation units; clusters with fewer than two members use the global population variance. Gaussian variance is `softplus(log_var)`, so this initializer stores the stable inverse-softplus of each target. It requires Gaussian emissions. The package default remains `spread`; the existing means-only `kmeans` mode retains its scale initialization.

## Health and context evidence

Health thresholds and context-evidence policy now live in Nautilus
adapter.qlib.nhsmm_validation; they are no longer exported by NHSMM.


## Centralized tunables and search

ModelConfig in nhsmm/config.py owns validated model and training settings.
Use with_overrides for a new validated copy and to_dict/from_dict for complete
serialization. Unknown fields, booleans used as numbers, invalid ranges and
nonfinite values fail before fitting.

NHSMM.fit_supervised(..., steps=None, lr=None) resolves omitted arguments from
supervised_steps and supervised_lr; explicit arguments take precedence. Adam
betas/epsilon apply to supervised, latent and transition refinement optimization.
grad_clip_norm controls latent/refinement updates. supervised_grad_clip_norm=None
preserves the previous unclipped supervised path.

DistributionSet passes config to all four component factories and fresh restarts.
Injected factories must accept config (ordinary subclasses inherit it). Context
projection, layer-normalization epsilon, output activation, Xavier gain and
initial delta scale reach every head. transition_context_max_delta retains the
transition-specific cap; context_max_delta controls the other heads.
context_hidden_min is a minimum; explicit hidden/context dimensions also
contribute to head width.

Initialization values affect construction/reinitialization, preserving existing
parameter identities. They do not overwrite fitted state_dict tensors at inference.
emission_raw_log_var and studentt_raw_* describe unconstrained raw parameters.
emission_variance_proxy preserves the existing Gaussian initializer:
log_var=log(proxy), hence effective variance softplus(log(proxy))=log(1+proxy).
The K-Means variance initializer instead clips empirical physical variances to
cluster_variance_min/max and applies exact inverse softplus.

Artifacts persist the full configuration. Older v3 payloads without these fields
resolve package defaults; current legacy-prediction equivalence is tested.
Historical host warm-start manifests remain strict and are not silently rewritten
or rehashed. Their original pinned source remains available for exact continuation.

Standalone Duration now shares ModelConfig.max_duration=35 (previously 30);
pass max_duration explicitly for a fixed direct-distribution shape. Standalone
DefaultEncoder channels and Convergence delta-mode API defaults are named
separately in config.py to preserve their existing direct-call behavior.

## New tunables

Defaults below describe this source revision. Pre-existing fields remain available.

| Field | Default | Consumer |
|---|---|---|
| adam_beta1 | 0.9 | nhsmm/models/base.py, nhsmm/supervised.py, nhsmm/training.py |
| adam_beta2 | 0.999 | nhsmm/models/base.py, nhsmm/supervised.py, nhsmm/training.py |
| adam_eps | 1e-08 | nhsmm/models/base.py, nhsmm/supervised.py, nhsmm/training.py |
| cluster_variance_max | 5.0 | nhsmm/models/base.py |
| cluster_variance_min | 0.05 | nhsmm/models/base.py |
| context_allow_projection | True | nhsmm/distributions/default.py |
| context_attention_init_std | 0.1 | nhsmm/models/base.py |
| context_delta_scale | 0.1 | nhsmm/distributions/default.py |
| context_final_activation | 'tanh' | nhsmm/distributions/default.py |
| context_hidden_min | 16 | nhsmm/distributions/default.py, nhsmm/models/base.py |
| context_layer_norm_eps | 1e-05 | nhsmm/distributions/default.py, nhsmm/models/base.py |
| context_max_delta | 0.5 | nhsmm/distributions/default.py |
| context_pool_layer_norm | True | nhsmm/models/base.py |
| context_pool_scale | 1.0 | nhsmm/models/base.py |
| context_xavier_gain | 1.0 | nhsmm/distributions/default.py |
| convergence_auto_patience | True | nhsmm/training.py |
| convergence_patience | None | nhsmm/training.py |
| convergence_patience_max | None | nhsmm/training.py |
| convergence_patience_scale | 1.0 | nhsmm/training.py |
| duration_bias_end | 0.3 | nhsmm/distributions/default.py |
| duration_bias_start | 0.7 | nhsmm/distributions/default.py |
| duration_init_decay | 0.05 | nhsmm/distributions/default.py |
| emission_init_std | 0.1 | nhsmm/distributions/default.py |
| emission_raw_log_var | -1.0 | nhsmm/distributions/default.py |
| emission_spread_extent | 1.0 | nhsmm/distributions/default.py |
| emission_spread_noise | 0.05 | nhsmm/distributions/default.py |
| emission_variance_proxy | 1.0 | nhsmm/distributions/default.py |
| encoder_hidden_max | 64 | nhsmm/models/base.py |
| encoder_hidden_min | 32 | nhsmm/models/base.py |
| encoder_hidden_multiplier | 2 | nhsmm/models/base.py |
| grad_clip_norm | 5.0 | nhsmm/models/base.py, nhsmm/training.py |
| init_jitter | 1e-05 | nhsmm/distributions/default.py |
| initial_bias_end | 0.2 | nhsmm/distributions/default.py |
| initial_bias_start | 0.8 | nhsmm/distributions/default.py |
| kmeans_atol | 1e-05 | nhsmm/models/base.py |
| kmeans_max_iter | 20 | nhsmm/models/base.py |
| kmeans_rtol | 1e-05 | nhsmm/models/base.py |
| leaky_relu_slope | 0.01 | nhsmm/distributions/default.py |
| logits_init_std | 0.1 | nhsmm/distributions/default.py |
| scheduler_factor | 0.5 | nhsmm/training.py |
| scheduler_min_patience | 2 | nhsmm/training.py |
| studentt_kmeans_raw_scale | 1.0 | nhsmm/models/base.py |
| studentt_raw_dof | 5.0 | nhsmm/distributions/default.py |
| studentt_raw_dof_floor | 2.1 | nhsmm/distributions/default.py, nhsmm/models/base.py |
| studentt_raw_scale | 0.1 | nhsmm/distributions/default.py |
| supervised_grad_clip_norm | None | nhsmm/supervised.py |
| supervised_lr | 0.03 | nhsmm/supervised.py |
| supervised_steps | 200 | nhsmm/supervised.py |
| temperature_min_ratio | 0.3 | nhsmm/training.py |
| temperature_steepness | 10.0 | nhsmm/models/base.py, nhsmm/training.py |
| transition_bias_diagonal | 0.7 | nhsmm/distributions/default.py |
| transition_bias_off_diagonal | 0.1 | nhsmm/distributions/default.py |

## Fixed constants and audit scope

Numerical policy constants are centralized outside the tunable dataclass:
EPS, dtype/logit limits, filter normalization tolerance/ULPs, probability
validation tolerances, boundary ULP allowances, LOG_HALF, attention mask sentinel
and the convergence learning-rate ratio floor. They preserve numerical/support
contracts and are excluded from tuning.

The transition row offset cancels over destination states under softmax; it is
not a search dimension. The standalone automatic head-width ratio is separate
from configured model head width. Mathematical identities, tensor dimensions,
indices, support endpoints and recurrence counters remain at their use sites.

The complete source-wide AST inventory, including every numeric literal and
its source line, is retained with source hashes in the external reference.
A numeral is not automatically a meaningful hyperparameter.

Search orchestration and winner selection belong to the Nautilus host:
adapter.qlib.nhsmm_research.config_tuning uses scikit-learn ParameterGrid,
native RD-Agent, Qlib and MLflow. The host docs/nhsmm-config-tuning report
records the frozen protocol and observed result. Synthetic calibration does
not establish an optimal trading configuration.

### Per-call optimization configuration

An alternate `optimize(cfg=...)` may change latent optimization budgets,
learning rate, Adam/clipping, temperature schedule, convergence/scheduler
settings, transition refinement and emission initialization mode.
The actual annealed objective and its history use the same supplied steepness;
transition refinement uses the supplied Adam and clipping settings too.

All remaining fields must match the model configuration. In particular, encoder,
distribution, initializer magnitude/KMeans settings, seed, padding and
supervised-only settings cannot be changed through this latent optimizer call.
Construct a model with those settings instead. Unsupported changes fail before
updates; they are never silently ignored. Per-call training overrides do not
replace the model construction configuration saved in inference artifacts.

### Numerical temperature limits

The latent objective and history share one logistic schedule calculation.
Constant temperatures bypass the exponential. Large positive exponents use the
equivalent reciprocal-exponential expression, avoiding overflow without changing
ordinary-range arithmetic. No temperature floor is silently added.

The final (smallest) update temperature is checked before model state changes.
For example, the product of a positive subnormal minimum ratio and a small
target can underflow to zero; with extreme steepness,
that unrepresentable schedule is rejected before initialization or updates.
This is a numerical validity check, not a change to the default schedule.

Formula reference: [SciPy expit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.expit.html).
The scalar implementation uses the standard library and adds no dependency.

### Nonfinite training failures

Latent training rejects nonfinite chunk and accumulated objectives, monitoring
scores and final restart scores. Transition refinement checks its objectives
and final score too. Whenever clipping is enabled, the native PyTorch
[error_if_nonfinite](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)
guard rejects nonfinite total gradient norms before Adam steps, including norm
overflow from individually finite gradients. Supervised elementwise gradient
checks remain active even with clipping disabled.

These failures raise immediately. Training/eval mode is restored; refinement
and supervised fitting also restore trainability flags. Failed latent attempts
publish no completed optimization summary. A failure does not automatically
roll back earlier successful updates or restart initialization. Final-score
failures may occur after updates; callers must discard failed fit results.
No retry, replacement loss or numerical threshold tuning is introduced.

### Supervised setup failures

The supervised cleanup boundary covers context-only trainability restrictions,
selection of trainable parameters, Adam construction and switching into training
mode. If any of those setup operations fails, original trainability flags and
the caller training/eval mode are restored. The no-trainable-parameters case
uses the same cleanup path. Numerical checks and the supervised objective,
updates, learning rate and returned summary remain unchanged.
