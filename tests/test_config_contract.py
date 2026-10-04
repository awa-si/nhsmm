from __future__ import annotations

import json

import pytest

from nhsmm import (
    ModelConfig,
    ValidationConfig,
    ValidationScenarioConfig,
)
from scripts.validate_learning_prediction import _apply_overrides


def test_model_config_roundtrip_and_strict_unknown_rejection() -> None:
    config = ModelConfig(n_states=3, n_features=4, max_duration=17, seed=11)
    payload = config.to_dict()

    restored = ModelConfig.from_dict(payload)

    assert restored == config
    with pytest.raises(ValueError, match="unknown ModelConfig fields"):
        ModelConfig.from_dict({**payload, "typo_lr": 0.1})


def test_model_config_with_overrides_is_validated_and_non_mutating() -> None:
    config = ModelConfig(n_states=3, n_features=3)

    tuned = config.with_overrides(lr=0.02, max_iter=75)

    assert config.lr != tuned.lr
    assert config.max_iter != tuned.max_iter
    assert tuned.lr == 0.02
    assert tuned.max_iter == 75
    with pytest.raises(ValueError, match="unknown ModelConfig fields"):
        config.with_overrides(not_a_field=1)


def test_validation_config_roundtrip_is_stable_and_json_serializable(tmp_path) -> None:
    config = ValidationConfig()
    path = tmp_path / "validation.json"

    config.to_json(path)
    restored = ValidationConfig.from_json(path)

    assert restored.to_dict() == config.to_dict()
    json.dumps(restored.to_dict(), sort_keys=True)


def test_validation_config_rejects_unknown_and_incompatible_contracts() -> None:
    payload = ValidationConfig().to_dict()
    payload["unknown"] = True
    with pytest.raises(ValueError, match="unknown ValidationConfig fields"):
        ValidationConfig.from_dict(payload)

    with pytest.raises(ValueError, match="n_features >= n_states"):
        ValidationConfig().with_overrides(model={"n_features": 2})

    with pytest.raises(ValueError, match="cannot exceed model.max_duration"):
        ValidationConfig().with_overrides(data={"min_duration": 30})


def test_validation_scenario_has_no_name_based_policy() -> None:
    scenario = ValidationScenarioConfig(
        separation=1.25,
        min_median_boundary_f1_tolerance=0.40,
        max_collapse_fraction=0.25,
    )

    assert scenario.usefulness_required
    assert scenario.min_median_boundary_f1_tolerance == 0.40


def test_validation_config_cli_style_overrides_are_strict_and_composable() -> None:
    config = ValidationConfig()

    tuned = _apply_overrides(
        config,
        [
            ("model.max_iter", 60),
            ("model.lr", 0.005),
            ("data.steps", 240),
            ("health.max_state_occupancy", 0.85),
            ("boundary_tolerance", 2),
            ("scenario.weak.min_median_boundary_f1_tolerance", 0.50),
        ],
    )

    assert tuned.model.max_iter == 60
    assert tuned.model.lr == 0.005
    assert tuned.data.steps == 240
    assert tuned.health.max_state_occupancy == 0.85
    assert tuned.boundary_tolerance == 2
    assert tuned.scenarios["weak"].min_median_boundary_f1_tolerance == 0.50
    assert config.model.max_iter == 40
    assert config.data.steps == 180

    with pytest.raises(ValueError, match="unknown ModelConfig fields"):
        _apply_overrides(config, [("model.typo", 1)])
    with pytest.raises(ValueError, match="unknown validation scenario"):
        _apply_overrides(config, [("scenario.missing.separation", 2.0)])


def test_validation_config_supports_tuned_dimensions() -> None:
    config = ValidationConfig().with_overrides(
        model={"n_features": 5, "max_duration": 30},
        data={"steps": 64, "n_train": 3, "n_eval": 2},
    )

    assert config.model.n_states == 3
    assert config.model.n_features == 5
    assert config.model.max_duration == 30
    assert config.data.steps == 64
    assert config.data.n_train == 3
    assert config.data.n_eval == 2
