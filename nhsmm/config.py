from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, fields, replace
from numbers import Real
from typing import Any, Literal, Mapping

import torch

EPS: float = 1e-12
DTYPE = torch.float32
NEG_INF = torch.finfo(DTYPE).min
MIN_LOGITS: float = NEG_INF
MAX_LOGITS: float = 1e5

# Numerical/standalone API constants: inspectable, not search hyperparameters.
FILTER_NORMALIZATION_ATOL = 1e-12
FILTER_NORMALIZATION_ULPS = 8
PROBABILITY_ATOL = 1e-6
BOUNDARY_RANGE_ULPS = 32
BOUNDARY_SUM_ULPS = 64
LOG_HALF = -math.log(2.0)
ATTENTION_MASK_LOGIT = -1e9
CONVERGENCE_LR_RATIO_FLOOR = 1e-8
STANDALONE_ENCODER_CHANNELS = 16
CONTEXT_ENCODER_DROPOUT = 0.0
STANDALONE_CONTEXT_HIDDEN_RATIO = 0.5
TRANSITION_INITIAL_ROW_OFFSET = 0.05
CONVERGENCE_DEFAULT_PATIENCE = 3
CONVERGENCE_DEFAULT_TOL = 1e-5
CONVERGENCE_DEFAULT_WINDOW = 5
CONVERGENCE_DEFAULT_MODE = "delta"


SUPERVISED_BOUNDARY_FLOAT32_ATOL = 2e-5
SUPERVISED_BOUNDARY_FLOAT64_ATOL = 2e-12

logger = logging.getLogger("NHSMM")
logger.addHandler(logging.NullHandler())


def _require_int(name: str, value: Any, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _require_optional_int(name: str, value: Any, *, minimum: int) -> None:
    if value is not None:
        _require_int(name, value, minimum=minimum)


def _require_finite(
    name: str,
    value: Any,
    *,
    minimum: float | None = None,
    minimum_inclusive: bool = True,
) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    numeric = float(value)
    if minimum is None:
        return
    valid = numeric >= minimum if minimum_inclusive else numeric > minimum
    if not valid:
        operator = ">=" if minimum_inclusive else ">"
        raise ValueError(f"{name} must be finite and {operator} {minimum}")


def _require_bool(name: str, value: Any) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")


def _reject_unknown(cls: type[Any], payload: Mapping[str, Any]) -> None:
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(payload) - allowed, key=repr)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {unknown}")


def _strict_dataclass_from_dict(
    cls: type[Any], payload: Mapping[str, Any], *, strict: bool = True
) -> Any:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{cls.__name__} payload must be a mapping")
    if strict:
        _reject_unknown(cls, payload)
        values = dict(payload)
    else:
        allowed = {item.name for item in fields(cls)}
        values = {key: value for key, value in payload.items() if key in allowed}
    return cls(**values)


@dataclass
class ModelConfig:
    """Validated runtime/training contract for NHSMM."""

    # Model architecture
    n_states: int
    n_features: int
    pad_value: float = 0.0

    # Encoder / context
    use_context_encoder: bool = False
    n_heads: int = 4
    cnn_kernel: int = 3
    cnn_channels: int = 5
    dropout: float = 0.05
    hidden_dim: int | None = None
    context_dim: int | None = None
    pool: Literal["mean", "last", "max", "attn", "mha"] = "mean"
    causal: bool = False

    # HMM / distributions
    max_duration: int = 35
    duration_tail: bool = False
    duration_tail_init_probability: float = 0.5
    supervised_duration_end_prior: float = 0.5
    supervised_duration_continue_prior: float = 0.5
    supervised_transition_prior: float = 0.5
    supervised_transition_pool_strength: float = 4.0
    min_covar: float = 1e-6
    emission_init_mode: Literal["random", "spread", "kmeans", "kmeans_cluster_variance"] = "spread"
    initial_init_mode: Literal["normal", "biased", "uniform"] = "normal"
    duration_init_mode: Literal["normal", "biased", "uniform"] = "normal"
    transition_init_mode: Literal["normal", "biased", "uniform"] = "normal"
    transition_context_max_delta: float = 0.5
    allow_self_transitions: bool = True
    transition_type: Literal["ergodic", "semi", "left-to-right"] = "ergodic"
    activation: Literal[
        "leaky_relu",
        "identity",
        "softplus",
        "gelu",
        "relu",
        "tanh",
    ] = "tanh"
    emission_type: Literal["gaussian", "studentt"] = "gaussian"

    # General
    seed: int | None = None

    # Optimization / training
    n_init: int = 3
    lr: float = 1e-2
    tol: float = 1e-4
    max_iter: int = 40
    optimizer_temperature_mode: Literal["annealed", "constant"] = "annealed"
    optimizer_target_temperature: float = 1.0
    optimizer_initialization: Literal["restart", "current"] = "restart"
    optimizer_batch_size: int | None = None
    transition_refine_steps: int = 0
    transition_refine_lr: float = 3e-2
    loss_bias: float = 1e-3
    plateau_window: int = 6
    plateau_tol: float = 1e-4
    use_scheduler: bool = True
    convergence_stop: bool = True
    convergence_mode: Literal["delta", "plateau"] = "plateau"
    verbose: bool = True

    # Central tunables: explicit overrides, artifacts and host-owned searches.
    context_pool_scale: float = 1.0
    context_pool_layer_norm: bool = True
    context_attention_init_std: float = 0.1
    context_hidden_min: int = 16
    encoder_hidden_min: int = 32
    encoder_hidden_max: int = 64
    encoder_hidden_multiplier: int = 2
    context_delta_scale: float = 0.1
    context_max_delta: float = 0.5
    context_layer_norm_eps: float = 1e-5
    context_allow_projection: bool = True
    context_final_activation: str = "tanh"
    context_xavier_gain: float = 1.0
    leaky_relu_slope: float = 0.01
    init_jitter: float = 1e-5
    logits_init_std: float = 0.1
    initial_bias_start: float = 0.8
    initial_bias_end: float = 0.2
    duration_bias_start: float = 0.7
    duration_bias_end: float = 0.3
    duration_init_decay: float = 0.05
    transition_bias_diagonal: float = 0.7
    transition_bias_off_diagonal: float = 0.1
    emission_init_std: float = 0.1
    emission_spread_extent: float = 1.0
    emission_spread_noise: float = 0.05
    emission_raw_log_var: float = -1.0
    emission_variance_proxy: float = 1.0
    studentt_raw_dof: float = 5.0
    studentt_raw_scale: float = 0.1
    studentt_kmeans_raw_scale: float = 1.0
    studentt_raw_dof_floor: float = 2.1
    kmeans_max_iter: int = 20
    kmeans_atol: float = 1e-5
    kmeans_rtol: float = 1e-5
    cluster_variance_min: float = 0.05
    cluster_variance_max: float = 5.0
    supervised_steps: int = 200
    supervised_lr: float = 0.03
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    grad_clip_norm: float = 5.0
    supervised_grad_clip_norm: float | None = None
    scheduler_factor: float = 0.5
    scheduler_min_patience: int = 2
    temperature_min_ratio: float = 0.3
    temperature_steepness: float = 10.0
    convergence_patience: int | None = None
    convergence_patience_scale: float = 1.0
    convergence_auto_patience: bool = True
    convergence_patience_max: int | None = None

    def __post_init__(self) -> None:
        for name, minimum in (
            ("n_states", 1),
            ("n_features", 1),
            ("n_heads", 1),
            ("cnn_kernel", 1),
            ("cnn_channels", 1),
            ("max_duration", 1),
            ("n_init", 1),
            ("max_iter", 1),
            ("plateau_window", 1),
            ("transition_refine_steps", 0),
        ):
            _require_int(name, getattr(self, name), minimum=minimum)

        _require_finite("context_pool_scale", self.context_pool_scale, minimum=0.0)
        _require_bool("context_pool_layer_norm", self.context_pool_layer_norm)
        _require_finite("context_attention_init_std", self.context_attention_init_std, minimum=0.0)
        _require_int("context_hidden_min", self.context_hidden_min, minimum=1)
        _require_int("encoder_hidden_min", self.encoder_hidden_min, minimum=1)
        _require_int("encoder_hidden_max", self.encoder_hidden_max, minimum=1)
        _require_int("encoder_hidden_multiplier", self.encoder_hidden_multiplier, minimum=1)
        _require_finite("context_delta_scale", self.context_delta_scale, minimum=0.0)
        _require_finite("context_max_delta", self.context_max_delta, minimum=0.0)
        _require_finite(
            "context_layer_norm_eps",
            self.context_layer_norm_eps,
            minimum=0.0,
            minimum_inclusive=False,
        )
        _require_bool("context_allow_projection", self.context_allow_projection)
        _require_finite(
            "context_xavier_gain", self.context_xavier_gain, minimum=0.0, minimum_inclusive=False
        )
        _require_finite("leaky_relu_slope", self.leaky_relu_slope, minimum=0.0)
        _require_finite("init_jitter", self.init_jitter, minimum=0.0)
        _require_finite("logits_init_std", self.logits_init_std, minimum=0.0)
        _require_finite(
            "initial_bias_start", self.initial_bias_start, minimum=0.0, minimum_inclusive=False
        )
        _require_finite(
            "initial_bias_end", self.initial_bias_end, minimum=0.0, minimum_inclusive=False
        )
        _require_finite(
            "duration_bias_start", self.duration_bias_start, minimum=0.0, minimum_inclusive=False
        )
        _require_finite(
            "duration_bias_end", self.duration_bias_end, minimum=0.0, minimum_inclusive=False
        )
        _require_finite("duration_init_decay", self.duration_init_decay, minimum=0.0)
        _require_finite(
            "transition_bias_diagonal",
            self.transition_bias_diagonal,
            minimum=0.0,
            minimum_inclusive=False,
        )
        _require_finite(
            "transition_bias_off_diagonal",
            self.transition_bias_off_diagonal,
            minimum=0.0,
            minimum_inclusive=False,
        )
        _require_finite("emission_init_std", self.emission_init_std, minimum=0.0)
        _require_finite("emission_spread_extent", self.emission_spread_extent, minimum=0.0)
        _require_finite("emission_spread_noise", self.emission_spread_noise, minimum=0.0)
        _require_finite("emission_raw_log_var", self.emission_raw_log_var)
        _require_finite(
            "emission_variance_proxy",
            self.emission_variance_proxy,
            minimum=0.0,
            minimum_inclusive=False,
        )
        _require_finite("studentt_raw_dof", self.studentt_raw_dof)
        _require_finite("studentt_raw_scale", self.studentt_raw_scale)
        _require_finite("studentt_kmeans_raw_scale", self.studentt_kmeans_raw_scale)
        _require_finite("studentt_raw_dof_floor", self.studentt_raw_dof_floor)
        _require_int("kmeans_max_iter", self.kmeans_max_iter, minimum=1)
        _require_finite("kmeans_atol", self.kmeans_atol, minimum=0.0)
        _require_finite("kmeans_rtol", self.kmeans_rtol, minimum=0.0)
        _require_finite(
            "cluster_variance_min", self.cluster_variance_min, minimum=0.0, minimum_inclusive=False
        )
        _require_finite(
            "cluster_variance_max", self.cluster_variance_max, minimum=0.0, minimum_inclusive=False
        )
        _require_int("supervised_steps", self.supervised_steps, minimum=1)
        _require_finite("supervised_lr", self.supervised_lr, minimum=0.0, minimum_inclusive=False)
        _require_finite("adam_beta1", self.adam_beta1, minimum=0.0)
        _require_finite("adam_beta2", self.adam_beta2, minimum=0.0)
        _require_finite("adam_eps", self.adam_eps, minimum=0.0, minimum_inclusive=False)
        _require_finite("grad_clip_norm", self.grad_clip_norm, minimum=0.0, minimum_inclusive=False)
        if self.supervised_grad_clip_norm is not None:
            _require_finite(
                "supervised_grad_clip_norm",
                self.supervised_grad_clip_norm,
                minimum=0.0,
                minimum_inclusive=False,
            )
        _require_finite("scheduler_factor", self.scheduler_factor, minimum=0.0)
        _require_int("scheduler_min_patience", self.scheduler_min_patience, minimum=1)
        _require_finite("temperature_min_ratio", self.temperature_min_ratio, minimum=0.0)
        _require_finite(
            "temperature_steepness",
            self.temperature_steepness,
            minimum=0.0,
            minimum_inclusive=False,
        )
        _require_optional_int("convergence_patience", self.convergence_patience, minimum=1)
        _require_finite("convergence_patience_scale", self.convergence_patience_scale, minimum=0.0)
        _require_bool("convergence_auto_patience", self.convergence_auto_patience)
        _require_optional_int("convergence_patience_max", self.convergence_patience_max, minimum=1)
        if self.encoder_hidden_min > self.encoder_hidden_max:
            raise ValueError("encoder_hidden_min must not exceed encoder_hidden_max")
        if self.cluster_variance_min > self.cluster_variance_max:
            raise ValueError("cluster_variance_min must not exceed cluster_variance_max")
        for name in ("adam_beta1", "adam_beta2"):
            if not 0.0 <= getattr(self, name) < 1.0:
                raise ValueError(f"{name} must be in [0,1)")
        for name in ("scheduler_factor", "temperature_min_ratio"):
            if not 0.0 < getattr(self, name) < 1.0:
                raise ValueError(f"{name} must be in (0,1)")
        if self.convergence_patience_max is not None and self.convergence_patience_max < (
            self.convergence_patience or max(1, self.plateau_window // 2)
        ):
            raise ValueError("convergence_patience_max is below effective patience")
        if not isinstance(
            self.context_final_activation, str
        ) or self.context_final_activation not in {
            "leaky_relu",
            "identity",
            "softplus",
            "gelu",
            "relu",
            "tanh",
        }:
            raise ValueError("unsupported context_final_activation")

        _require_optional_int("hidden_dim", self.hidden_dim, minimum=1)
        _require_optional_int("context_dim", self.context_dim, minimum=1)
        _require_optional_int("optimizer_batch_size", self.optimizer_batch_size, minimum=1)
        if self.seed is not None and (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not -(2**63) <= self.seed < 2**64
        ):
            raise ValueError("seed must be None or an integer in [-2**63, 2**64 - 1]")

        _require_finite("pad_value", self.pad_value)
        _require_finite("dropout", self.dropout)
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")

        for name in (
            "supervised_duration_end_prior",
            "supervised_duration_continue_prior",
            "supervised_transition_prior",
        ):
            _require_finite(name, getattr(self, name), minimum=0.0, minimum_inclusive=False)
        _require_finite(
            "supervised_transition_pool_strength",
            self.supervised_transition_pool_strength,
            minimum=0.0,
        )

        for name in ("min_covar", "lr", "transition_refine_lr", "optimizer_target_temperature"):
            _require_finite(
                name,
                getattr(self, name),
                minimum=0.0,
                minimum_inclusive=False,
            )
        _require_finite(
            "duration_tail_init_probability",
            self.duration_tail_init_probability,
            minimum=0.0,
            minimum_inclusive=False,
        )
        if self.duration_tail_init_probability > 1.0:
            raise ValueError("duration_tail_init_probability must be in (0,1]")

        for name in (
            "transition_context_max_delta",
            "tol",
            "loss_bias",
            "plateau_tol",
        ):
            _require_finite(name, getattr(self, name), minimum=0.0)

        for name in (
            "allow_self_transitions",
            "use_context_encoder",
            "duration_tail",
            "causal",
            "use_scheduler",
            "convergence_stop",
            "verbose",
        ):
            _require_bool(name, getattr(self, name))

        choices = {
            "pool": {"mean", "last", "max", "attn", "mha"},
            "emission_init_mode": {"random", "spread", "kmeans", "kmeans_cluster_variance"},
            "initial_init_mode": {"normal", "biased", "uniform"},
            "duration_init_mode": {"normal", "biased", "uniform"},
            "transition_init_mode": {"normal", "biased", "uniform"},
            "transition_type": {"ergodic", "semi", "left-to-right"},
            "activation": {
                "leaky_relu",
                "identity",
                "softplus",
                "gelu",
                "relu",
                "tanh",
            },
            "emission_type": {"gaussian", "studentt"},
            "convergence_mode": {"delta", "plateau"},
            "optimizer_temperature_mode": {"annealed", "constant"},
            "optimizer_initialization": {"restart", "current"},
        }
        for name, allowed in choices.items():
            value = getattr(self, name)
            if not isinstance(value, str) or value not in allowed:
                raise ValueError(f"{name} must be one of {sorted(allowed)}, got {value!r}")
        if not self.allow_self_transitions and (
            self.n_states < 2 or self.transition_type != "ergodic"
        ):
            raise ValueError("excluding self transitions requires ergodic K>=2")
        if self.optimizer_initialization == "current" and self.n_init != 1:
            raise ValueError("optimizer_initialization=current requires n_init=1")

    @property
    def model_mode(self) -> Literal["hmm", "hsmm", "nhsmm"]:
        """Resolved model family, derived from the serialized architecture."""
        if self.use_context_encoder or self.context_dim is not None:
            return "nhsmm"
        if self.max_duration == 1 and not self.duration_tail:
            return "hmm"
        return "hsmm"

    @classmethod
    def for_mode(
        cls,
        mode: Literal["hmm", "hsmm", "nhsmm"],
        *,
        n_states: int,
        n_features: int,
        **overrides: Any,
    ) -> "ModelConfig":
        """Build a causal, independent HMM -> HSMM -> NHSMM profile.

        doc:nhsmm.model-modes -> docs/model-modes.md
        Lower-level ModelConfig construction remains available for custom laws.
        Only resolved existing fields are serialized; artifact v3 is unchanged.
        """
        if not isinstance(mode, str) or mode not in {"hmm", "hsmm", "nhsmm"}:
            raise ValueError("mode must be hmm, hsmm or nhsmm")
        _reject_unknown(cls, overrides)
        settings: dict[str, Any] = {
            "causal": True,
            "allow_self_transitions": mode == "hmm",
        }
        if mode in {"hmm", "hsmm"}:
            settings.update(use_context_encoder=False, context_dim=None, hidden_dim=None)
        else:
            # Explicit external context does not require an observation encoder.
            settings["use_context_encoder"] = overrides.get("context_dim") is None
        if mode == "hmm":
            settings.update(max_duration=1, duration_tail=False)
        settings.update(overrides)
        config = cls(n_states=n_states, n_features=n_features, **settings)
        if not config.causal:
            raise ValueError(f"mode={mode} requires causal=True")
        if mode in {"hmm", "hsmm"} and (
            config.use_context_encoder
            or config.context_dim is not None
            or config.hidden_dim is not None
        ):
            raise ValueError(f"mode={mode} requires no encoder or neural context dimensions")
        if mode == "hmm" and (
            config.max_duration != 1 or config.duration_tail or not config.allow_self_transitions
        ):
            raise ValueError(
                "mode=hmm requires max_duration=1, duration_tail=False and self transitions"
            )
        if mode in {"hsmm", "nhsmm"} and (config.max_duration < 2 or config.allow_self_transitions):
            raise ValueError(f"mode={mode} requires max_duration>=2 and no self-reset transitions")
        if config.model_mode != mode:
            raise ValueError(f"mode={mode} requires an encoder or explicit context_dim")
        return config

    def to_dict(self) -> dict[str, Any]:
        """Return the complete resolved model configuration."""

        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        strict: bool = True,
    ) -> "ModelConfig":
        """Construct a validated config from a mapping."""

        return _strict_dataclass_from_dict(cls, payload, strict=strict)

    def with_overrides(self, **overrides: Any) -> "ModelConfig":
        """Return a validated copy with explicit tuning overrides."""

        _reject_unknown(type(self), overrides)
        return replace(self, **overrides)


def default_value(name: str) -> Any:
    """Single source for low-level API defaults; no model instantiation."""
    return ModelConfig.__dataclass_fields__[name].default
