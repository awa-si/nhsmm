from pathlib import Path

import pytest
import torch

from nhsmm import DefaultEncoder, ModelConfig, NHSMM
from nhsmm.context import align_context_tensor
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
    model_parameter_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
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


def test_external_context_shape_contract() -> None:
    B, T, H = 2, 3, 4
    global_context = align_context_tensor(torch.zeros(H), batch_size=B, timesteps=T, context_dim=H)
    temporal_context = align_context_tensor(
        torch.zeros(T, H), batch_size=B, timesteps=T, context_dim=H
    )
    batch_static_context = align_context_tensor(
        torch.zeros(B, 1, H), batch_size=B, timesteps=T, context_dim=H
    )
    full_context = align_context_tensor(
        torch.zeros(B, T, H), batch_size=B, timesteps=T, context_dim=H
    )
    for tensor in (global_context, temporal_context, batch_static_context, full_context):
        assert tensor.shape == (B, T, H)

    with pytest.raises(ValueError, match=r"2D context must be \[T,H\]"):
        align_context_tensor(torch.zeros(B, H), batch_size=B, timesteps=T, context_dim=H)


def test_ambiguous_2d_context_is_always_temporal_when_batch_equals_time() -> None:
    context = torch.tensor([[10.0], [20.0]])
    aligned = align_context_tensor(context, batch_size=2, timesteps=2, context_dim=1)
    assert aligned[:, :, 0].tolist() == [[10.0, 20.0], [10.0, 20.0]]


@pytest.mark.parametrize(
    "context_factory",
    [
        lambda B, T, H: torch.zeros(H),
        lambda B, T, H: torch.zeros(T, H),
        lambda B, T, H: torch.zeros(B, 1, H),
        lambda B, T, H: torch.zeros(B, T, H),
    ],
)
def test_model_build_sequence_set_accepts_public_context_shapes(context_factory) -> None:
    model = _model(context_dim=2)
    B, T, H = 2, 3, 2
    X = torch.zeros(B, T, 2)
    sequence = model._build_sequence_set(X, context=context_factory(B, T, H))
    assert sequence.contexts.shape == (B, T, H)


def test_model_build_sequence_set_rejects_batch_static_2d_context() -> None:
    model = _model(context_dim=2)
    B, T, H = 2, 3, 2
    with pytest.raises(ValueError, match=r"2D context must be \[T,H\]"):
        model._build_sequence_set(torch.zeros(B, T, 2), context=torch.zeros(B, H))


def test_explicit_context_dim_controls_default_encoder_output_width() -> None:
    model = _model(context_dim=2)
    assert model.context_dim == 2
    assert model.hidden_dim == 2
    assert model.config.context_dim == 2
    assert model.config.hidden_dim == 2
    assert model.encoder.encoder.hidden_dim == 2


def test_explicit_context_dim_rejects_incompatible_custom_encoder() -> None:
    encoder = DefaultEncoder(
        n_features=2,
        hidden_dim=4,
        cnn_channels=5,
        cnn_kernel=3,
        causal=True,
    )
    with pytest.raises(
        ValueError, match=r"encoder output dimension \(4\) must equal context_dim \(2\)"
    ):
        NHSMM(
            ModelConfig(
                n_states=2,
                n_features=2,
                max_duration=3,
                causal=True,
                context_dim=2,
                dropout=0.0,
            ),
            encoder=encoder,
            device="cpu",
        )


def test_noncausal_explicit_context_dim_controls_bidirectional_output_width() -> None:
    config = ModelConfig(
        n_states=2,
        n_features=2,
        max_duration=3,
        causal=False,
        context_dim=2,
        hidden_dim=2,
        dropout=0.0,
    )
    model = NHSMM(config, device="cpu")
    assert model.encoder.encoder.bidirectional is True
    assert model.encoder.encoder.hidden_dim == 1
    assert model.encoder.encoder.out_dim == 2
    assert model.context_dim == 2


def test_noncausal_default_encoder_rejects_odd_context_dim() -> None:
    with pytest.raises(ValueError, match="context_dim must be divisible by 2"):
        NHSMM(
            ModelConfig(
                n_states=2,
                n_features=2,
                max_duration=3,
                causal=False,
                context_dim=3,
                hidden_dim=3,
                dropout=0.0,
            ),
            device="cpu",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dropout", float("nan")),
        ("min_covar", float("nan")),
        ("lr", float("nan")),
        ("transition_refine_lr", float("inf")),
        ("transition_context_max_delta", float("nan")),
        ("tol", float("inf")),
        ("loss_bias", float("nan")),
        ("plateau_tol", float("inf")),
    ],
)
def test_model_config_rejects_non_finite_public_floats(field: str, value) -> None:
    kwargs = {"n_states": 2, "n_features": 2}
    kwargs[field] = value
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pool", "bogus"),
        ("emission_init_mode", "bogus"),
        ("initial_init_mode", "bogus"),
        ("duration_init_mode", "bogus"),
        ("transition_init_mode", "bogus"),
        ("transition_type", "bogus"),
        ("activation", "bogus"),
        ("emission_type", "bogus"),
        ("convergence_mode", "bogus"),
    ],
)
def test_model_config_rejects_invalid_literal_values(field: str, value: str) -> None:
    kwargs = {"n_states": 2, "n_features": 2}
    kwargs[field] = value
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)


@pytest.mark.parametrize(
    "field",
    ["causal", "use_scheduler", "convergence_stop", "verbose"],
)
def test_model_config_rejects_non_boolean_flags(field: str) -> None:
    kwargs = {"n_states": 2, "n_features": 2, field: 1}
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)
