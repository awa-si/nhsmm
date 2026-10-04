from __future__ import annotations

import json

import pytest

from nhsmm import (
    Choice,
    ConfigTuner,
    FloatRange,
    IntRange,
    ModelConfig,
    TuneEvaluation,
    ValidationConfig,
    apply_config_overrides,
)


def test_apply_config_overrides_supports_model_and_validation_paths() -> None:
    model = ModelConfig(n_states=3, n_features=3)
    tuned_model = apply_config_overrides(
        model,
        {"lr": 0.02, "model.max_iter": 55},
    )
    assert tuned_model.lr == 0.02
    assert tuned_model.max_iter == 55
    assert model.lr != tuned_model.lr

    validation = ValidationConfig()
    tuned_validation = apply_config_overrides(
        validation,
        {
            "model.lr": 0.005,
            "data.steps": 240,
            "health.max_state_occupancy": 0.85,
            "scenario.weak.max_collapse_fraction": 0.25,
            "boundary_tolerance": 2,
            "seeds": [1, 2, 3],
        },
    )
    assert tuned_validation.model.lr == 0.005
    assert tuned_validation.data.steps == 240
    assert tuned_validation.health.max_state_occupancy == 0.85
    assert tuned_validation.scenarios["weak"].max_collapse_fraction == 0.25
    assert tuned_validation.boundary_tolerance == 2
    assert tuned_validation.seeds == (1, 2, 3)


def test_apply_config_overrides_rejects_invalid_paths_and_values() -> None:
    config = ValidationConfig()
    with pytest.raises(ValueError, match="ValidationConfig tuning paths"):
        apply_config_overrides(config, {"unknown.path": 1})
    with pytest.raises(ValueError, match="unknown ModelConfig fields"):
        apply_config_overrides(config, {"model.typo": 1})
    with pytest.raises(ValueError, match="lr"):
        apply_config_overrides(config, {"model.lr": 0.0})


def test_grid_tuner_is_deterministic_and_selects_best_config() -> None:
    tuner = ConfigTuner(
        ModelConfig(n_states=3, n_features=3),
        {
            "lr": Choice([0.01, 0.02]),
            "max_iter": IntRange(20, 40, step=20),
        },
        direction="maximize",
    )

    report = tuner.run(
        lambda config: TuneEvaluation(
            score=config.lr * 100.0 + config.max_iter / 100.0,
            metrics={"lr": config.lr},
        )
    )

    assert len(report.trials) == 4
    assert report.best.config.lr == 0.02
    assert report.best.config.max_iter == 40
    assert report.best.evaluation.metrics["lr"] == 0.02


def test_minimize_tuner_selects_lowest_score() -> None:
    tuner = ConfigTuner(
        ModelConfig(n_states=2, n_features=2),
        {"dropout": Choice([0.0, 0.1, 0.2])},
        direction="minimize",
    )

    report = tuner.run(lambda config: config.dropout)

    assert report.best.config.dropout == 0.0
    assert report.best.evaluation.score == 0.0


def test_random_tuner_is_reproducible() -> None:
    space = {
        "lr": FloatRange(1e-4, 1e-2, steps=5, log=True),
        "max_iter": IntRange(20, 60, step=20),
    }
    first = ConfigTuner(
        ModelConfig(n_states=2, n_features=2),
        space,
        seed=17,
    ).candidates(strategy="random", trials=5)
    second = ConfigTuner(
        ModelConfig(n_states=2, n_features=2),
        space,
        seed=17,
    ).candidates(strategy="random", trials=5)

    assert first == second


def test_tune_report_is_json_serializable(tmp_path) -> None:
    tuner = ConfigTuner(
        ValidationConfig(),
        {
            "model.max_iter": Choice([20, 40]),
            "boundary_tolerance": Choice([1, 2]),
        },
    )
    report = tuner.run(
        lambda config: TuneEvaluation(
            score=float(config.model.max_iter),
            metrics={"boundary_tolerance": float(config.boundary_tolerance)},
        )
    )
    path = tmp_path / "tuning.json"

    report.to_json(path)
    payload = json.loads(path.read_text())

    assert payload["best"]["config"]["model"]["max_iter"] == 40
    assert len(payload["trials"]) == 4


def test_choice_requires_values() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Choice([])


def test_range_contracts() -> None:
    with pytest.raises(ValueError):
        IntRange(3, 1)
    with pytest.raises(ValueError):
        IntRange(1, 3, step=0)
    with pytest.raises(ValueError):
        FloatRange(1.0, 0.0, steps=3)
    with pytest.raises(ValueError):
        FloatRange(0.0, 1.0, steps=3, log=True)
    with pytest.raises(ValueError):
        FloatRange(0.1, 1.0, steps=0)
