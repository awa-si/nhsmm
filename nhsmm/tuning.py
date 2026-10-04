from __future__ import annotations

import itertools
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

from nhsmm.config import ModelConfig, ValidationConfig

TuneDirection = Literal["maximize", "minimize"]
TuneStrategy = Literal["grid", "random"]
TunableConfig = ModelConfig | ValidationConfig


@dataclass(frozen=True)
class Choice:
    """Finite categorical tuning domain."""

    values: tuple[Any, ...]

    def __init__(self, values: Sequence[Any]) -> None:
        values = tuple(values)
        if not values:
            raise ValueError("Choice requires at least one value")
        object.__setattr__(self, "values", values)

    def grid_values(self) -> tuple[Any, ...]:
        return self.values

    def sample(self, rng: random.Random) -> Any:
        return rng.choice(self.values)


@dataclass(frozen=True)
class IntRange:
    """Inclusive integer tuning range."""

    low: int
    high: int
    step: int = 1

    def __post_init__(self) -> None:
        for name, value in (("low", self.low), ("high", self.high), ("step", self.step)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.step <= 0:
            raise ValueError("step must be > 0")
        if self.high < self.low:
            raise ValueError("high must be >= low")

    def grid_values(self) -> tuple[int, ...]:
        return tuple(range(self.low, self.high + 1, self.step))

    def sample(self, rng: random.Random) -> int:
        values = self.grid_values()
        return rng.choice(values)


@dataclass(frozen=True)
class FloatRange:
    """Finite linear or logarithmic floating-point tuning range."""

    low: float
    high: float
    steps: int
    log: bool = False

    def __post_init__(self) -> None:
        for name, value in (("low", self.low), ("high", self.high)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps < 1:
            raise ValueError("steps must be an integer >= 1")
        if self.high < self.low:
            raise ValueError("high must be >= low")
        if self.log and (self.low <= 0.0 or self.high <= 0.0):
            raise ValueError("log FloatRange requires low/high > 0")

    def grid_values(self) -> tuple[float, ...]:
        if self.steps == 1:
            return (float(self.low),)
        if self.log:
            low = math.log(float(self.low))
            high = math.log(float(self.high))
            return tuple(
                math.exp(low + (high - low) * index / (self.steps - 1))
                for index in range(self.steps)
            )
        return tuple(
            float(self.low) + (float(self.high) - float(self.low)) * index / (self.steps - 1)
            for index in range(self.steps)
        )

    def sample(self, rng: random.Random) -> float:
        if self.log:
            return math.exp(rng.uniform(math.log(float(self.low)), math.log(float(self.high))))
        return rng.uniform(float(self.low), float(self.high))


ParameterDomain = Choice | IntRange | FloatRange


@dataclass(frozen=True)
class TuneEvaluation:
    """One objective value plus optional diagnostic metrics."""

    score: float
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise TypeError("score must be a real number")
        if not math.isfinite(float(self.score)):
            raise ValueError("score must be finite")
        for name, value in self.metrics.items():
            if not isinstance(name, str) or not name:
                raise ValueError("metric names must be non-empty strings")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"metric {name!r} must be a real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"metric {name!r} must be finite")

    def as_dict(self) -> dict[str, object]:
        return {
            "score": float(self.score),
            "metrics": {name: float(value) for name, value in self.metrics.items()},
        }


@dataclass(frozen=True)
class TuneTrial:
    """Materialized candidate and its evaluation."""

    index: int
    overrides: Mapping[str, Any]
    config: TunableConfig
    evaluation: TuneEvaluation

    def as_dict(self) -> dict[str, object]:
        config_payload = (
            self.config.to_dict() if isinstance(self.config, ModelConfig) else self.config.to_dict()
        )
        return {
            "index": self.index,
            "overrides": dict(self.overrides),
            "config": config_payload,
            "evaluation": self.evaluation.as_dict(),
        }


@dataclass(frozen=True)
class TuneReport:
    """Completed tuning run."""

    direction: TuneDirection
    strategy: TuneStrategy
    seed: int
    trials: tuple[TuneTrial, ...]
    best_index: int

    @property
    def best(self) -> TuneTrial:
        return self.trials[self.best_index]

    def as_dict(self) -> dict[str, object]:
        return {
            "direction": self.direction,
            "strategy": self.strategy,
            "seed": self.seed,
            "best_index": self.best_index,
            "best": self.best.as_dict(),
            "trials": [trial.as_dict() for trial in self.trials],
        }

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n")


def apply_config_overrides(
    config: TunableConfig,
    overrides: Mapping[str, Any],
) -> TunableConfig:
    """Apply strict dotted tuning overrides and return a validated copy."""

    result: TunableConfig = config
    for path, value in overrides.items():
        if not isinstance(path, str) or not path:
            raise ValueError("override paths must be non-empty strings")
        parts = path.split(".")

        if isinstance(result, ModelConfig):
            if len(parts) == 2 and parts[0] == "model":
                field_name = parts[1]
            elif len(parts) == 1:
                field_name = parts[0]
            else:
                raise ValueError("ModelConfig tuning paths must be <field> or model.<field>")
            result = result.with_overrides(**{field_name: value})
            continue

        if len(parts) == 2 and parts[0] == "model":
            result = result.with_overrides(model={parts[1]: value})
        elif len(parts) == 2 and parts[0] == "data":
            result = result.with_overrides(data={parts[1]: value})
        elif len(parts) == 2 and parts[0] == "health":
            result = result.with_overrides(health={parts[1]: value})
        elif len(parts) == 3 and parts[0] == "scenario":
            result = result.with_overrides(scenarios={parts[1]: {parts[2]: value}})
        elif path == "boundary_tolerance":
            result = result.with_overrides(boundary_tolerance=value)
        elif path == "seeds":
            if not isinstance(value, (list, tuple)):
                raise TypeError("seeds override must be a list or tuple")
            result = result.with_overrides(seeds=tuple(value))
        else:
            raise ValueError(
                "ValidationConfig tuning paths must be model.<field>, "
                "data.<field>, health.<field>, scenario.<name>.<field>, "
                "boundary_tolerance, or seeds"
            )
    return result


class ConfigTuner:
    """Deterministic tuner over validated NHSMM configuration contracts."""

    def __init__(
        self,
        base_config: TunableConfig,
        space: Mapping[str, ParameterDomain],
        *,
        direction: TuneDirection = "maximize",
        seed: int = 0,
    ) -> None:
        if not isinstance(base_config, (ModelConfig, ValidationConfig)):
            raise TypeError("base_config must be ModelConfig or ValidationConfig")
        if direction not in ("maximize", "minimize"):
            raise ValueError("direction must be 'maximize' or 'minimize'")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        if not isinstance(space, Mapping) or not space:
            raise ValueError("space must be a non-empty mapping")

        self.base_config = base_config
        self.direction = direction
        self.seed = seed
        self.space = dict(space)

        for path, domain in self.space.items():
            if not isinstance(path, str) or not path:
                raise ValueError("tuning paths must be non-empty strings")
            if not isinstance(domain, (Choice, IntRange, FloatRange)):
                raise TypeError(
                    f"tuning domain for {path!r} must be Choice, IntRange, or FloatRange"
                )

        # Validate every finite grid value eagerly against the config contract.
        for path, domain in self.space.items():
            for value in domain.grid_values():
                apply_config_overrides(self.base_config, {path: value})

    def candidates(
        self,
        *,
        strategy: TuneStrategy = "grid",
        trials: int | None = None,
    ) -> tuple[Mapping[str, Any], ...]:
        if strategy not in ("grid", "random"):
            raise ValueError("strategy must be 'grid' or 'random'")
        if trials is not None and (
            isinstance(trials, bool) or not isinstance(trials, int) or trials < 1
        ):
            raise ValueError("trials must be None or an integer >= 1")

        paths = tuple(self.space)
        if strategy == "grid":
            values = [self.space[path].grid_values() for path in paths]
            combinations = tuple(
                dict(zip(paths, combination)) for combination in itertools.product(*values)
            )
            return combinations if trials is None else combinations[:trials]

        if trials is None:
            raise ValueError("random strategy requires trials")
        rng = random.Random(self.seed)
        generated: list[Mapping[str, Any]] = []
        for _ in range(trials):
            generated.append({path: self.space[path].sample(rng) for path in paths})
        return tuple(generated)

    def run(
        self,
        evaluate: Callable[[TunableConfig], TuneEvaluation | float],
        *,
        strategy: TuneStrategy = "grid",
        trials: int | None = None,
    ) -> TuneReport:
        if not callable(evaluate):
            raise TypeError("evaluate must be callable")

        completed: list[TuneTrial] = []
        for index, overrides in enumerate(self.candidates(strategy=strategy, trials=trials)):
            config = apply_config_overrides(self.base_config, overrides)
            raw = evaluate(config)
            evaluation = raw if isinstance(raw, TuneEvaluation) else TuneEvaluation(score=raw)
            completed.append(
                TuneTrial(
                    index=index,
                    overrides=dict(overrides),
                    config=config,
                    evaluation=evaluation,
                )
            )

        if not completed:
            raise RuntimeError("tuning produced no trials")

        def score_key(trial: TuneTrial) -> float:
            return float(trial.evaluation.score)

        best_trial = (
            max(completed, key=score_key)
            if self.direction == "maximize"
            else min(completed, key=score_key)
        )
        return TuneReport(
            direction=self.direction,
            strategy=strategy,
            seed=self.seed,
            trials=tuple(completed),
            best_index=best_trial.index,
        )
