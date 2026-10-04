from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Mapping

from nhsmm.config import ModelConfig
from nhsmm.diagnostics import ModelHealthThresholds

VALIDATION_CONFIG_SCHEMA_VERSION = 1


def _require_probability(name: str, value: float | None) -> None:
    if value is not None and (not math.isfinite(value) or not 0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be None or finite in [0,1]")


def _require_nonnegative(name: str, value: float | None) -> None:
    if value is not None and (not math.isfinite(value) or value < 0.0):
        raise ValueError(f"{name} must be None or finite and >= 0")


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
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1")
        if (
            isinstance(self.oos_seed_offset, bool)
            or not isinstance(self.oos_seed_offset, int)
            or self.oos_seed_offset == 0
        ):
            raise ValueError("oos_seed_offset must be a non-zero integer")
        if not math.isfinite(self.duration_poisson_mean) or self.duration_poisson_mean <= 0.0:
            raise ValueError("duration_poisson_mean must be finite and > 0")

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
        if not math.isfinite(self.separation) or self.separation < 0.0:
            raise ValueError("separation must be finite and >= 0")
        if not isinstance(self.usefulness_required, bool):
            raise ValueError("usefulness_required must be a bool")
        if not isinstance(self.expect_collapse, bool):
            raise ValueError("expect_collapse must be a bool")
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


def _default_model() -> ModelConfig:
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
        emission_init_mode="kmeans",
    )


def _default_scenarios() -> dict[str, ValidationScenarioConfig]:
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
    """Versioned, serializable contract for learning/prediction validation."""

    schema_version: int = VALIDATION_CONFIG_SCHEMA_VERSION
    seeds: tuple[int, ...] = tuple(range(401, 411))
    boundary_tolerance: int = 1
    data: ValidationDataConfig = field(default_factory=ValidationDataConfig)
    model: ModelConfig = field(default_factory=_default_model)
    health: ModelHealthThresholds = field(default_factory=ModelHealthThresholds)
    scenarios: dict[str, ValidationScenarioConfig] = field(default_factory=_default_scenarios)

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
        if (
            isinstance(self.boundary_tolerance, bool)
            or not isinstance(self.boundary_tolerance, int)
            or self.boundary_tolerance < 0
        ):
            raise ValueError("boundary_tolerance must be an integer >= 0")
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
            health = values["health"]
            if not isinstance(health, Mapping):
                raise TypeError("health must be a mapping")
            _reject_unknown(ModelHealthThresholds, health)
            values["health"] = ModelHealthThresholds(**dict(health))
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
        source = Path(path)
        payload = json.loads(source.read_text())
        return cls.from_dict(payload)

    def to_json(self, path: str | Path) -> None:
        target = Path(path)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

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
        next_health = self.health
        if health is not None:
            _reject_unknown(ModelHealthThresholds, health)
            next_health = replace(self.health, **dict(health))
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
