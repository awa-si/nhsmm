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
