"""Core configuration remains strict; experiment search is host-owned."""

import importlib.util
import subprocess
import sys

import pytest

from nhsmm import ModelConfig, ValidationConfig
from scripts.validate_learning_prediction import _apply_overrides


def test_configs_update_without_mutating_the_base():
    base = ModelConfig(n_states=3, n_features=2)
    changed = base.with_overrides(lr=0.02, max_iter=55)
    assert (changed.lr, changed.max_iter) == (0.02, 55)
    assert base.to_dict() != changed.to_dict()
    with pytest.raises(ValueError, match="unknown ModelConfig"):
        base.with_overrides(typo=1)


def test_validation_updates_are_native_and_strict():
    base = ValidationConfig()
    changed = base.with_overrides(
        model={"lr": 0.005},
        data={"steps": 240},
        health={"max_state_occupancy": 0.85},
        scenarios={"weak": {"max_collapse_fraction": 0.25}},
        boundary_tolerance=2,
        seeds=(1, 2, 3),
    )
    assert changed.model.lr == 0.005
    assert changed.data.steps == 240
    assert changed.scenarios["weak"].max_collapse_fraction == 0.25
    assert changed.seeds == (1, 2, 3)
    assert base.to_dict() != changed.to_dict()


@pytest.mark.parametrize(
    "path,value",
    [
        ("invalid", 1),
        ("model.nested.lr", 0.1),
        ("scenario.weak", {}),
        ("model.typo", 1),
    ],
)
def test_validation_cli_keeps_strict_path_errors(path, value):
    with pytest.raises(ValueError):
        _apply_overrides(ValidationConfig(), [(path, value)])


def test_validation_cli_seeds_and_thresholds_remain_explicit():
    changed = _apply_overrides(
        ValidationConfig(),
        [
            ("seeds", [2, 3]),
            ("boundary_tolerance", 2),
            ("scenario.weak.max_collapse_fraction", 0.25),
        ],
    )
    assert changed.seeds == (2, 3)
    assert changed.boundary_tolerance == 2
    with pytest.raises(TypeError, match="seeds"):
        _apply_overrides(ValidationConfig(), [("seeds", 2)])


def test_model_core_import_has_no_research_framework_dependency():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import nhsmm
assert not {"qlib", "mlflow"}.intersection(sys.modules)
assert not hasattr(nhsmm, "ConfigTuner")
""",
        ],
        check=True,
    )
    assert importlib.util.find_spec("nhsmm.tuning") is None


@pytest.mark.parametrize("module", ["nhsmm.data", "nhsmm.benchmark"])
def test_test_only_data_and_benchmark_are_not_shipped_core_modules(module):
    assert importlib.util.find_spec(module) is None
