# Central model configuration

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
