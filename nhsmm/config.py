from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field, fields, replace
from numbers import Real
from pathlib import Path
from typing import Any, Literal, Mapping

import torch

EPS: float = 1e-12
DTYPE = torch.float32
NEG_INF = torch.finfo(DTYPE).min
MIN_LOGITS: float = NEG_INF
MAX_LOGITS: float = 1e5

logger = logging.getLogger("NHSMM")
logger.addHandler(logging.NullHandler())

VALIDATION_CONFIG_SCHEMA_VERSION = 1


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


def _require_probability(name: str, value: float | None) -> None:
    if value is None:
        return
    _require_finite(name, value)
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be None or finite in [0,1]")


def _require_nonnegative(name: str, value: float | None) -> None:
    if value is not None:
        _require_finite(name, value, minimum=0.0)


def _require_bool(name: str, value: Any) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")


def _reject_unknown(cls: type[Any], payload: Mapping[str, Any]) -> None:
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {unknown}")


def _strict_dataclass_from_dict(cls: type[Any], payload: Mapping[str, Any]) -> Any:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{cls.__name__} payload must be a mapping")
    _reject_unknown(cls, payload)
    return cls(**dict(payload))


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
    min_covar: float = 1e-6
    emission_init_mode: Literal["random", "spread", "kmeans"] = "spread"
    initial_init_mode: Literal["normal", "biased", "uniform"] = "normal"
    duration_init_mode: Literal["normal", "biased", "uniform"] = "normal"
    transition_init_mode: Literal["normal", "biased", "uniform"] = "normal"
    transition_context_max_delta: float = 0.5
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
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise ValueError("seed must be None or an integer")

        _require_finite("pad_value", self.pad_value)
        _require_finite("dropout", self.dropout)
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")

        for name in ("min_covar", "lr", "transition_refine_lr"):
            _require_finite(
                name,
                getattr(self, name),
                minimum=0.0,
                minimum_inclusive=False,
            )
        for name in (
            "transition_context_max_delta",
            "tol",
            "loss_bias",
            "plateau_tol",
        ):
            _require_finite(name, getattr(self, name), minimum=0.0)

        for name in (
            "use_context_encoder",
            "causal",
            "use_scheduler",
            "convergence_stop",
            "verbose",
        ):
            _require_bool(name, getattr(self, name))

        choices = {
            "pool": {"mean", "last", "max", "attn", "mha"},
            "emission_init_mode": {"random", "spread", "kmeans"},
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
        }
        for name, allowed in choices.items():
            value = getattr(self, name)
            if value not in allowed:
                raise ValueError(f"{name} must be one of {sorted(allowed)}, got {value!r}")

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

        if not isinstance(payload, Mapping):
            raise TypeError("ModelConfig payload must be a mapping")
        values = dict(payload)
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if strict and unknown:
            raise ValueError(f"unknown ModelConfig fields: {unknown}")
        for name in unknown:
            values.pop(name, None)
        return cls(**values)

    def with_overrides(self, **overrides: Any) -> "ModelConfig":
        """Return a validated copy with explicit tuning overrides."""

        _reject_unknown(type(self), overrides)
        return replace(self, **overrides)


@dataclass(frozen=True)
class ModelHealthThresholds:
    """Thresholds used to classify fitted-model degeneration."""

    min_effective_states: float = 2.0
    max_state_occupancy: float = 0.80
    min_viterbi_states: int = 2
    min_state_occupancy: float = 0.01
    max_duration_peak: float = 0.95
    max_transition_peak: float = 0.95

    def __post_init__(self) -> None:
        _require_finite(
            "min_effective_states",
            self.min_effective_states,
            minimum=0.0,
            minimum_inclusive=False,
        )
        _require_probability("max_state_occupancy", self.max_state_occupancy)
        if self.max_state_occupancy <= 0.0:
            raise ValueError("max_state_occupancy must be in (0,1]")
        _require_int("min_viterbi_states", self.min_viterbi_states, minimum=1)
        _require_probability("min_state_occupancy", self.min_state_occupancy)
        if self.min_state_occupancy >= 1.0:
            raise ValueError("min_state_occupancy must be in [0,1)")
        for name in ("max_duration_peak", "max_transition_peak"):
            value = getattr(self, name)
            _require_probability(name, value)
            if value <= 0.0:
                raise ValueError(f"{name} must be in (0,1]")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModelHealthThresholds":
        return _strict_dataclass_from_dict(cls, payload)

    def with_overrides(self, **overrides: Any) -> "ModelHealthThresholds":
        _reject_unknown(type(self), overrides)
        return replace(self, **overrides)


@dataclass(frozen=True)
class ValidationDataConfig:
    """Synthetic dataset contract for learning/prediction acceptance."""

    steps: int = 180
    n_train: int = 8
    n_eval: int = 5
    duration_poisson_mean: float = 9.0
    min_duration: int = 2
    oos_seed_offset: int = 10_000

    def __post_init__(self) -> None:
        for name in ("steps", "n_train", "n_eval", "min_duration"):
            _require_int(name, getattr(self, name), minimum=1)
        if (
            isinstance(self.oos_seed_offset, bool)
            or not isinstance(self.oos_seed_offset, int)
            or self.oos_seed_offset == 0
        ):
            raise ValueError("oos_seed_offset must be a non-zero integer")
        _require_finite(
            "duration_poisson_mean",
            self.duration_poisson_mean,
            minimum=0.0,
            minimum_inclusive=False,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ValidationDataConfig":
        return _strict_dataclass_from_dict(cls, payload)

    def with_overrides(self, **overrides: Any) -> "ValidationDataConfig":
        _reject_unknown(type(self), overrides)
        return replace(self, **overrides)


@dataclass(frozen=True)
class ValidationScenarioConfig:
    """Acceptance thresholds for one controlled signal regime."""

    separation: float
    usefulness_required: bool = True
    min_median_oos_accuracy: float | None = None
    min_median_accuracy_gain: float | None = None
    min_median_ll_gain: float | None = None
    max_median_oos_accuracy: float | None = None
    max_abs_median_ari: float | None = None
    max_collapse_fraction: float | None = None
    min_median_boundary_f1: float | None = None
    min_median_boundary_f1_tolerance: float | None = None
    max_median_run_length_rel_error: float | None = None
    max_median_transition_mae: float | None = None
    max_median_occupancy_l1: float | None = None
    max_median_ece: float | None = None
    expect_collapse: bool = False

    def __post_init__(self) -> None:
        _require_finite("separation", self.separation, minimum=0.0)
        _require_bool("usefulness_required", self.usefulness_required)
        _require_bool("expect_collapse", self.expect_collapse)

        for name in (
            "min_median_oos_accuracy",
            "max_median_oos_accuracy",
            "max_abs_median_ari",
            "max_collapse_fraction",
            "min_median_boundary_f1",
            "min_median_boundary_f1_tolerance",
            "max_median_occupancy_l1",
            "max_median_ece",
        ):
            _require_probability(name, getattr(self, name))
        for name in (
            "min_median_accuracy_gain",
            "min_median_ll_gain",
            "max_median_run_length_rel_error",
            "max_median_transition_mae",
        ):
            _require_nonnegative(name, getattr(self, name))

        if self.expect_collapse and self.max_collapse_fraction is not None:
            raise ValueError("expect_collapse and max_collapse_fraction are mutually exclusive")
        if not self.usefulness_required and any(
            getattr(self, name) is not None
            for name in (
                "min_median_boundary_f1",
                "min_median_boundary_f1_tolerance",
                "max_median_run_length_rel_error",
                "max_median_transition_mae",
                "max_median_occupancy_l1",
                "max_median_ece",
                "max_collapse_fraction",
            )
        ):
            raise ValueError("usefulness thresholds must be None when usefulness_required is False")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ValidationScenarioConfig":
        return _strict_dataclass_from_dict(cls, payload)

    def with_overrides(self, **overrides: Any) -> "ValidationScenarioConfig":
        _reject_unknown(type(self), overrides)
        return replace(self, **overrides)


def _default_validation_model() -> ModelConfig:
    return ModelConfig(
        n_states=3,
        n_features=3,
        max_duration=25,
        causal=True,
        n_init=1,
        max_iter=40,
        use_scheduler=False,
        convergence_stop=False,
        verbose=False,
        dropout=0.0,
        use_context_encoder=True,
        emission_init_mode="kmeans",
    )


def _default_validation_scenarios() -> dict[str, ValidationScenarioConfig]:
    return {
        "strong": ValidationScenarioConfig(
            separation=4.0,
            min_median_oos_accuracy=0.95,
            min_median_ll_gain=1.0,
            max_collapse_fraction=0.0,
            min_median_boundary_f1=0.90,
            max_median_run_length_rel_error=0.12,
            max_median_transition_mae=0.10,
            max_median_occupancy_l1=0.10,
            max_median_ece=0.12,
        ),
        "moderate": ValidationScenarioConfig(
            separation=2.0,
            min_median_oos_accuracy=0.90,
            min_median_accuracy_gain=0.02,
            min_median_ll_gain=0.50,
            max_collapse_fraction=0.0,
            min_median_boundary_f1=0.70,
            max_median_run_length_rel_error=0.25,
            max_median_transition_mae=0.18,
            max_median_occupancy_l1=0.16,
            max_median_ece=0.18,
        ),
        "weak": ValidationScenarioConfig(
            separation=1.0,
            min_median_oos_accuracy=0.55,
            min_median_ll_gain=0.10,
            max_collapse_fraction=0.40,
            min_median_boundary_f1_tolerance=0.45,
            max_median_run_length_rel_error=0.50,
            max_median_transition_mae=0.30,
            max_median_occupancy_l1=0.30,
            max_median_ece=0.25,
        ),
        "null": ValidationScenarioConfig(
            separation=0.0,
            usefulness_required=False,
            max_median_oos_accuracy=0.45,
            max_abs_median_ari=0.10,
        ),
    }


@dataclass(frozen=True)
class ValidationConfig:
    """Versioned, serializable learning/prediction validation contract."""

    schema_version: int = VALIDATION_CONFIG_SCHEMA_VERSION
    seeds: tuple[int, ...] = tuple(range(401, 411))
    boundary_tolerance: int = 1
    data: ValidationDataConfig = field(default_factory=ValidationDataConfig)
    model: ModelConfig = field(default_factory=_default_validation_model)
    health: ModelHealthThresholds = field(default_factory=ModelHealthThresholds)
    scenarios: dict[str, ValidationScenarioConfig] = field(
        default_factory=_default_validation_scenarios
    )

    def __post_init__(self) -> None:
        if self.schema_version != VALIDATION_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                "unsupported validation config schema_version "
                f"{self.schema_version}; expected {VALIDATION_CONFIG_SCHEMA_VERSION}"
            )
        if not isinstance(self.seeds, tuple) or not self.seeds:
            raise ValueError("seeds must be a non-empty tuple")
        if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in self.seeds):
            raise ValueError("all seeds must be integers")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        _require_int("boundary_tolerance", self.boundary_tolerance, minimum=0)

        if not isinstance(self.data, ValidationDataConfig):
            raise TypeError("data must be ValidationDataConfig")
        if not isinstance(self.model, ModelConfig):
            raise TypeError("model must be ModelConfig")
        if not isinstance(self.health, ModelHealthThresholds):
            raise TypeError("health must be ModelHealthThresholds")

        if self.model.n_states < 2:
            raise ValueError("validation model requires n_states >= 2")
        if self.model.n_features < self.model.n_states:
            raise ValueError(
                "validation model requires n_features >= n_states for identifiable "
                "synthetic state means"
            )
        if self.data.min_duration > self.model.max_duration:
            raise ValueError("data.min_duration cannot exceed model.max_duration")

        if not isinstance(self.scenarios, dict) or not self.scenarios:
            raise ValueError("scenarios must be a non-empty dict")
        for name, scenario in self.scenarios.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("scenario names must be non-empty strings")
            if not isinstance(scenario, ValidationScenarioConfig):
                raise TypeError(f"scenario {name!r} must be ValidationScenarioConfig")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "seeds": list(self.seeds),
            "boundary_tolerance": self.boundary_tolerance,
            "data": asdict(self.data),
            "model": self.model.to_dict(),
            "health": asdict(self.health),
            "scenarios": {
                name: asdict(scenario) for name, scenario in sorted(self.scenarios.items())
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ValidationConfig":
        if not isinstance(payload, Mapping):
            raise TypeError("ValidationConfig payload must be a mapping")
        _reject_unknown(cls, payload)
        values = dict(payload)

        if "seeds" in values:
            seeds = values["seeds"]
            if not isinstance(seeds, (list, tuple)):
                raise TypeError("seeds must be a list or tuple")
            values["seeds"] = tuple(seeds)
        if "data" in values:
            values["data"] = ValidationDataConfig.from_dict(values["data"])
        if "model" in values:
            values["model"] = ModelConfig.from_dict(values["model"])
        if "health" in values:
            values["health"] = ModelHealthThresholds.from_dict(values["health"])
        if "scenarios" in values:
            scenarios = values["scenarios"]
            if not isinstance(scenarios, Mapping):
                raise TypeError("scenarios must be a mapping")
            values["scenarios"] = {
                str(name): ValidationScenarioConfig.from_dict(config)
                for name, config in scenarios.items()
            }
        return cls(**values)

    @classmethod
    def from_json(cls, path: str | Path) -> "ValidationConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    def with_overrides(
        self,
        *,
        seeds: tuple[int, ...] | None = None,
        boundary_tolerance: int | None = None,
        data: Mapping[str, Any] | None = None,
        model: Mapping[str, Any] | None = None,
        health: Mapping[str, Any] | None = None,
        scenarios: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> "ValidationConfig":
        next_data = self.data if data is None else self.data.with_overrides(**dict(data))
        next_model = self.model if model is None else self.model.with_overrides(**dict(model))
        next_health = self.health if health is None else self.health.with_overrides(**dict(health))

        next_scenarios = dict(self.scenarios)
        if scenarios is not None:
            for name, overrides in scenarios.items():
                if name not in next_scenarios:
                    raise ValueError(f"unknown validation scenario {name!r}")
                next_scenarios[name] = next_scenarios[name].with_overrides(**dict(overrides))

        return replace(
            self,
            seeds=self.seeds if seeds is None else tuple(seeds),
            boundary_tolerance=(
                self.boundary_tolerance if boundary_tolerance is None else boundary_tolerance
            ),
            data=next_data,
            model=next_model,
            health=next_health,
            scenarios=next_scenarios,
        )
