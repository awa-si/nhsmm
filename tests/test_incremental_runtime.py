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
