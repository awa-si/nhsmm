from __future__ import annotations

import pytest
import torch

from nhsmm import DefaultEncoder, ModelConfig, NHSMM
from nhsmm.artifact import build_artifact, load_artifact, save_artifact
from nhsmm.context import ContextEncoder
from nhsmm.filtering import filter_model_sequence
from nhsmm.runtime import HSMMFilterRuntime


def _config(**overrides) -> ModelConfig:
    values = dict(
        n_states=3,
        n_features=2,
        max_duration=4,
        causal=True,
        dropout=0.0,
        n_init=1,
        max_iter=1,
        use_scheduler=False,
        convergence_stop=False,
        verbose=False,
        seed=271,
    )
    values.update(overrides)
    return ModelConfig(**values)


def test_context_encoder_is_opt_in_by_default() -> None:
    model = NHSMM(_config(), device="cpu")

    assert model.encoder is None
    assert model.context_dim is None
    assert model.hidden_dim is None

    model.initialize_distributions()
    assert model.dist.initial.context_net is None
    assert model.dist.duration.context_net is None
    assert model.dist.transition.context_net is None
    assert model.dist.emission.context_net is None

    values = model.log_likelihood(torch.randn(2, 6, 2), reduce=False)
    assert values.shape == (2,)
    assert torch.isfinite(values).all()


def test_explicit_encoder_does_not_mutate_reusable_config() -> None:
    config = _config()
    source = NHSMM(_config(use_context_encoder=True), device="cpu")
    encoder = source.encoder

    model = NHSMM(config, encoder=encoder, device="cpu")

    assert model.config.use_context_encoder is True
    assert model.encoder is encoder
    assert config.use_context_encoder is False
    assert config.context_dim is None
    assert config.hidden_dim is None

    plain = NHSMM(config, device="cpu")
    assert plain.encoder is None
    assert plain.config.use_context_encoder is False


def test_plain_hsmm_has_no_trainable_neural_controls() -> None:
    model = NHSMM(_config(), device="cpu")
    model.initialize_distributions()

    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert not any(name.endswith(".delta_scale") for name in trainable)
    assert not any(name.endswith(".log_temperature") for name in trainable)


def test_internal_context_encoder_requires_explicit_opt_in() -> None:
    model = NHSMM(_config(use_context_encoder=True), device="cpu")

    assert isinstance(model.encoder, ContextEncoder)
    assert model.config.use_context_encoder is True
    assert model.context_dim is not None
    assert model.hidden_dim == model.context_dim

    model.initialize_distributions()
    assert model.dist.emission.context_net is not None


def test_explicit_encoder_argument_is_an_opt_in() -> None:
    source = NHSMM(_config(use_context_encoder=True), device="cpu")
    encoder = source.encoder

    model = NHSMM(_config(), encoder=encoder, device="cpu")

    assert model.config.use_context_encoder is True
    assert model.encoder is encoder


def test_external_context_does_not_require_internal_encoder() -> None:
    model = NHSMM(_config(context_dim=3, hidden_dim=3), device="cpu")
    model.initialize_distributions()

    assert model.encoder is None
    observations = torch.randn(2, 5, 2)
    context = torch.randn(2, 5, 3)
    values = model.log_likelihood(observations, context=context, reduce=False)

    assert values.shape == (2,)
    assert torch.isfinite(values).all()


def test_encoderless_runtime_keeps_bounded_observation_state() -> None:
    model = NHSMM(_config(), device="cpu")
    model.initialize_distributions()
    model.eval()

    runtime = HSMMFilterRuntime(model)
    observations = torch.randn(1, 8, 2)
    for timestep in range(observations.shape[1]):
        runtime.step(observations[:, timestep])

    assert runtime.state is not None
    assert runtime.state.encoder_state is None
    assert runtime.state.observations.shape == (1, 1, 2)


def test_encoderless_artifact_marks_encoder_as_absent() -> None:
    model = NHSMM(_config(), device="cpu")
    model.initialize_distributions()

    payload = build_artifact(model)

    assert payload["model_config"]["use_context_encoder"] is False
    assert payload["encoder_config"] is None


def test_encoderless_runtime_matches_batch_filter() -> None:
    model = NHSMM(_config(), device="cpu")
    model.initialize_distributions()
    model.eval()

    observations = torch.randn(2, 7, 2)
    expected = filter_model_sequence(model, observations)
    runtime = HSMMFilterRuntime(model)
    actual = torch.stack(
        [
            runtime.step(observations[:, timestep]).log_posterior
            for timestep in range(observations.shape[1])
        ],
        dim=1,
    )

    torch.testing.assert_close(actual, expected.log_posterior, atol=1e-6, rtol=1e-6)
    assert runtime.state is not None
    assert runtime.state.observations.shape == (2, 1, 2)


def test_encoderless_artifact_round_trip(tmp_path) -> None:
    model = NHSMM(_config(), device="cpu")
    model.initialize_distributions()
    model.eval()
    observations = torch.randn(2, 6, 2)
    expected = model.log_likelihood(observations)

    path = save_artifact(model, tmp_path / "encoderless.nhsmm.pt")
    loaded = load_artifact(path, device="cpu", require_causal=True)
    actual = loaded.log_likelihood(observations)

    assert loaded.encoder is None
    assert loaded.config.use_context_encoder is False
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_artifact_loader_preserves_legacy_encoder_payload_semantics(tmp_path) -> None:
    model = NHSMM(_config(use_context_encoder=True), device="cpu")
    model.initialize_distributions()
    payload = build_artifact(model)
    payload["model_config"].pop("use_context_encoder")

    path = tmp_path / "legacy-encoder-v1.nhsmm.pt"
    torch.save(payload, path)
    loaded = load_artifact(path, device="cpu", require_causal=True)

    assert isinstance(loaded.encoder, ContextEncoder)
    assert loaded.config.use_context_encoder is True


def test_mha_default_head_count_is_compatible_with_default_encoder() -> None:
    model = NHSMM(_config(use_context_encoder=True, pool="mha"), device="cpu")
    assert model.config.n_heads == 4
    assert model.encoder._mha is not None
    assert model.encoder._mha.embed_dim % model.config.n_heads == 0


@pytest.mark.parametrize("pool", ["attn", "mha"])
def test_learned_pooling_parameters_survive_reset(pool: str) -> None:
    model = NHSMM(_config(use_context_encoder=True, pool=pool), device="cpu")
    before = {name: id(parameter) for name, parameter in model.encoder.named_parameters()}
    assert any(name.startswith("_attn_vector") or name.startswith("_mha.") for name in before)

    model.encoder.reset()

    after = {name: id(parameter) for name, parameter in model.encoder.named_parameters()}
    assert after == before
    model_parameter_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    assert set(after.values()).issubset(model_parameter_ids)


def test_explicit_context_dim_controls_default_encoder_output_width() -> None:
    model = NHSMM(
        _config(use_context_encoder=True, context_dim=2, hidden_dim=2),
        device="cpu",
    )
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
            _config(context_dim=2, hidden_dim=2),
            encoder=encoder,
            device="cpu",
        )


def test_noncausal_explicit_context_dim_controls_bidirectional_output_width() -> None:
    model = NHSMM(
        _config(
            causal=False,
            use_context_encoder=True,
            context_dim=2,
            hidden_dim=2,
        ),
        device="cpu",
    )
    assert model.encoder.encoder.bidirectional is True
    assert model.encoder.encoder.hidden_dim == 1
    assert model.encoder.encoder.out_dim == 2
    assert model.context_dim == 2


def test_noncausal_default_encoder_rejects_odd_context_dim() -> None:
    with pytest.raises(ValueError, match="context_dim must be divisible by 2"):
        NHSMM(
            _config(
                causal=False,
                use_context_encoder=True,
                context_dim=3,
                hidden_dim=3,
            ),
            device="cpu",
        )


def test_causal_internal_context_uses_raw_sequence_features() -> None:
    torch.manual_seed(277)
    model = NHSMM(_config(use_context_encoder=True, dropout=0.0), device="cpu")
    model.initialize_distributions(jitter=0.0)
    model.eval()
    model.encoder.context_scale = 1.75
    observations = torch.randn(2, 5, model.config.n_features)
    mask = torch.ones(2, 5, dtype=torch.bool)

    with torch.no_grad():
        raw = model.encoder.encoder(observations, mask=mask)
        sequence = model._build_sequence_set(observations)

    torch.testing.assert_close(sequence.contexts, raw)
    torch.testing.assert_close(sequence.canonical, raw[:, :1])
