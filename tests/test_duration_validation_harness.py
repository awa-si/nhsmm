from __future__ import annotations

from scripts.validate_duration_context import _model_config


def test_duration_validation_opts_into_context_encoder() -> None:
    config = _model_config(201)

    assert config.use_context_encoder is True
    assert config.n_states == 2
    assert config.n_features == 3
    assert config.max_duration == 30
    assert config.max_iter == 40
    assert config.n_init == 1
