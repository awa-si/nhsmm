from __future__ import annotations

import torch

from nhsmm import ModelConfig, NHSMM
from nhsmm.filtering import filter_model_sequence
from nhsmm.runtime import HSMMFilterRuntime


def _make_model() -> NHSMM:
    config = ModelConfig(
        n_states=3,
        n_features=4,
        max_duration=5,
        causal=True,
        dropout=0.0,
        seed=47,
    )
    model = NHSMM(config, device="cpu")
    model.initialize_distributions()
    model.eval()
    return model


def test_default_encoder_stream_matches_full_causal_context() -> None:
    torch.manual_seed(47)
    model = _make_model()
    x = torch.randn(2, 9, model.config.n_features)

    with torch.inference_mode():
        sequence = model._build_sequence_set(x)
        encoder = model.encoder.encoder
        state = encoder.initial_stream_state(
            x.shape[0],
            device=x.device,
            dtype=x.dtype,
        )

        streamed = []
        for t in range(x.shape[1]):
            context, state = encoder.stream_step(x[:, t], state)
            streamed.append(context)

    streamed_context = torch.cat(streamed, dim=1)
    assert torch.allclose(
        streamed_context,
        sequence.contexts,
        atol=1e-6,
        rtol=1e-6,
    )


def test_streaming_lstm_gate_cache_invalidates_without_changing_state_dict() -> None:
    torch.manual_seed(49)
    model = _make_model()
    encoder = model.encoder.encoder
    x = torch.randn(2, 7, model.config.n_features)
    state_keys = set(model.state_dict())

    with torch.inference_mode():
        state = encoder.initial_stream_state(
            x.shape[0],
            device=x.device,
            dtype=x.dtype,
        )
        encoder.stream_step(x[:, 0], state)
    first_key = encoder._stream_lstm_cache_key
    assert first_key is not None
    assert encoder._stream_lstm_weight is not None
    assert encoder._stream_lstm_bias is not None
    assert set(model.state_dict()) == state_keys

    with torch.no_grad():
        encoder.lstm.weight_ih_l0.add_(0.001)

    with torch.inference_mode():
        sequence = model._build_sequence_set(x)
        state = encoder.initial_stream_state(
            x.shape[0],
            device=x.device,
            dtype=x.dtype,
        )
        streamed = []
        for t in range(x.shape[1]):
            context, state = encoder.stream_step(x[:, t], state)
            streamed.append(context)

    assert encoder._stream_lstm_cache_key != first_key
    assert set(model.state_dict()) == state_keys
    torch.testing.assert_close(
        torch.cat(streamed, dim=1),
        sequence.contexts,
        atol=1e-6,
        rtol=1e-6,
    )


def test_streaming_lstm_cache_buffers_follow_dtype_move_without_rebuild() -> None:
    torch.manual_seed(50)
    model = _make_model()
    encoder = model.encoder.encoder
    x = torch.randn(2, model.config.n_features)

    with torch.inference_mode():
        state = encoder.initial_stream_state(
            x.shape[0],
            device=x.device,
            dtype=x.dtype,
        )
        encoder.stream_step(x, state)
    first_key = encoder._stream_lstm_cache_key

    model = model.to(dtype=torch.float64)
    encoder = model.encoder.encoder
    x64 = x.to(dtype=torch.float64)
    with torch.inference_mode():
        state = encoder.initial_stream_state(
            x64.shape[0],
            device=x64.device,
            dtype=x64.dtype,
        )
        encoder.stream_step(x64, state)

    assert encoder._stream_lstm_cache_key == first_key
    assert encoder._stream_lstm_weight is not None
    assert encoder._stream_lstm_bias is not None
    assert encoder._stream_lstm_weight.dtype == torch.float64
    assert encoder._stream_lstm_bias.dtype == torch.float64


def test_streaming_lstm_grad_enabled_path_remains_differentiable() -> None:
    torch.manual_seed(51)
    model = _make_model()
    encoder = model.encoder.encoder
    x = torch.randn(2, model.config.n_features)
    state = encoder.initial_stream_state(
        x.shape[0],
        device=x.device,
        dtype=x.dtype,
    )

    encoder.zero_grad(set_to_none=True)
    context, _ = encoder.stream_step(x, state)
    context.sum().backward()

    gradient = encoder.lstm.weight_ih_l0.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_runtime_incremental_encoder_matches_batch_filter() -> None:
    torch.manual_seed(53)
    model = _make_model()
    x = torch.randn(2, 10, model.config.n_features)

    batch = filter_model_sequence(model, x)
    runtime = HSMMFilterRuntime(model)
    streamed = []

    for t in range(x.shape[1]):
        state = runtime.step(x[:, t], timestamp=t)
        streamed.append(state.log_posterior)
        assert runtime.state is not None
        assert runtime.state.encoder_state is not None
        assert runtime.state.observations.shape == (2, 1, model.config.n_features)

    streamed_trace = torch.stack(streamed, dim=1)
    assert torch.allclose(
        streamed_trace,
        batch.log_posterior,
        atol=1e-6,
        rtol=1e-6,
    )
