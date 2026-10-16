"""Core configuration remains strict; experiment search is host-owned."""

import importlib.util
import subprocess
import sys

import pytest

from nhsmm import ModelConfig


def test_configs_update_without_mutating_the_base():
    base = ModelConfig(n_states=3, n_features=2)
    changed = base.with_overrides(lr=0.02, max_iter=55)
    assert (changed.lr, changed.max_iter) == (0.02, 55)
    assert base.to_dict() != changed.to_dict()
    with pytest.raises(ValueError, match="unknown ModelConfig"):
        base.with_overrides(typo=1)


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
