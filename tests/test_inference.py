from __future__ import annotations

import copy

import torch

from nhsmm import ModelConfig, NHSMM, load_inference_model, prepare_inference


def _make_model(*, causal: bool = True) -> NHSMM:
    config = ModelConfig(
        n_states=3,
        n_features=4,
        max_duration=5,
        causal=causal,
        dropout=0.0,
        seed=41,
    )
    model = NHSMM(config, device="cpu")
    model.initialize_distributions()
    return model


def test_prepare_inference_sets_eval_and_freezes_parameters() -> None:
    model = _make_model()
    model.train()

    prepared = prepare_inference(model)

    assert prepared is model
    assert prepared.training is False
    assert all(not parameter.requires_grad for parameter in prepared.parameters())


def test_load_inference_model_round_trips_outputs() -> None:
    torch.manual_seed(43)
    source = prepare_inference(_make_model())
    x = torch.randn(1, 6, source.config.n_features)
    expected = source.log_likelihood(x)

    config = copy.deepcopy(source.config)
    loaded = load_inference_model(
        config,
        source.state_dict(),
        device="cpu",
        require_causal=True,
    )

    actual = loaded.log_likelihood(x)
    assert loaded.training is False
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_load_inference_model_rejects_incompatible_state() -> None:
    source = prepare_inference(_make_model())
    broken = dict(source.state_dict())
    broken.pop(next(iter(broken)))

    try:
        load_inference_model(copy.deepcopy(source.config), broken, device="cpu")
    except ValueError as exc:
        assert "incompatible NHSMM state_dict" in str(exc)
    else:
        raise AssertionError("missing state must fail closed")


def test_load_inference_model_can_require_causal_config() -> None:
    source = prepare_inference(_make_model(causal=False))

    try:
        load_inference_model(
            copy.deepcopy(source.config),
            source.state_dict(),
            device="cpu",
            require_causal=True,
        )
    except ValueError as exc:
        assert "causal=True" in str(exc)
    else:
        raise AssertionError("non-causal config must fail when causal inference is required")


def test_prepare_inference_rejects_nonfinite_persistent_buffer() -> None:
    model = _make_model()
    model.register_buffer("poisoned_buffer", torch.tensor(float("nan")), persistent=True)

    try:
        prepare_inference(model)
    except ValueError as exc:
        assert "model state" in str(exc)
        assert "poisoned_buffer" in str(exc)
    else:
        raise AssertionError("non-finite persistent buffers must fail closed")


def test_load_inference_model_rejects_nonfinite_state_tensor() -> None:
    source = _make_model()
    state = {name: value.detach().clone() for name, value in source.state_dict().items()}
    name = next(key for key, value in state.items() if value.is_floating_point())
    state[name].reshape(-1)[0] = float("inf")

    try:
        load_inference_model(copy.deepcopy(source.config), state, device="cpu")
    except ValueError as exc:
        assert "state_dict" in str(exc)
        assert name in str(exc)
    else:
        raise AssertionError("non-finite state tensors must fail closed")
