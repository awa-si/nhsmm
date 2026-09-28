from pathlib import Path

import pytest
import torch

from nhsmm import ModelConfig, NHSMM
from nhsmm.data import SequenceDataset


def _model(**kwargs) -> NHSMM:
    cfg = ModelConfig(
        n_states=2,
        n_features=2,
        max_duration=3,
        causal=True,
        dropout=0.0,
        n_init=1,
        max_iter=1,
        use_scheduler=False,
        convergence_stop=False,
        verbose=False,
        **kwargs,
    )
    model = NHSMM(cfg, device="cpu")
    model.initialize_distributions()
    return model


def test_python_baseline_is_312() -> None:
    text = Path("pyproject.toml").read_text()
    assert 'requires-python = ">=3.12"' in text
    assert 'target-version = ["py312"]' in text
    assert 'target-version = "py312"' in text
    assert 'python_version = "3.12"' in text


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_states", 0),
        ("n_features", 0),
        ("n_heads", 0),
        ("cnn_kernel", 0),
        ("max_duration", 0),
        ("n_init", 0),
        ("max_iter", 0),
        ("transition_refine_steps", -1),
        ("dropout", 1.0),
        ("min_covar", 0.0),
        ("temperature", 0.0),
        ("lr", 0.0),
        ("transition_context_max_delta", -0.1),
    ],
)
def test_model_config_rejects_invalid_public_bounds(field: str, value) -> None:
    kwargs = {"n_states": 2, "n_features": 2}
    kwargs[field] = value
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)


def test_mha_default_head_count_is_compatible_with_default_encoder() -> None:
    model = _model(pool="mha")
    assert model.config.n_heads == 4
    assert model.encoder._mha is not None
    assert model.encoder._mha.embed_dim % model.config.n_heads == 0


@pytest.mark.parametrize("pool", ["attn", "mha"])
def test_learned_pooling_parameters_survive_reset(pool: str) -> None:
    model = _model(pool=pool)
    before = {name: id(parameter) for name, parameter in model.encoder.named_parameters()}
    assert any(name.startswith("_attn_vector") or name.startswith("_mha.") for name in before)

    model.encoder.reset()

    after = {name: id(parameter) for name, parameter in model.encoder.named_parameters()}
    assert after == before
    model_parameter_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assert set(after.values()).issubset(model_parameter_ids)


def test_variable_length_dataset_collates_state_sequence_without_polars() -> None:
    dataset = SequenceDataset(
        n_states=2,
        n_features=2,
        seed=3,
        n_segments_per_state=1,
        context_dim=2,
        variable_length=True,
    )
    assert len(dataset) == 1
    x, context, states = dataset[0]
    assert x.shape[0] == context.shape[0] == states.shape[0]

    x_pad, context_pad, states_pad, lengths = next(iter(dataset.loader(batch_size=1)))
    assert x_pad.shape[:2] == context_pad.shape[:2] == states_pad.shape
    assert lengths.tolist() == [x.shape[0]]
