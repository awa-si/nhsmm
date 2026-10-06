from __future__ import annotations

import torch

from nhsmm import ModelConfig, NHSMM, prepare_inference


def _model(seed: int = 91) -> NHSMM:
    config = ModelConfig(
        n_states=3,
        n_features=4,
        max_duration=5,
        causal=True,
        use_context_encoder=True,
        dropout=0.0,
        seed=seed,
    )
    model = NHSMM(config, device="cpu")
    model.initialize_distributions()
    model.eval()
    return model


def test_prepare_inference_fuses_context_affine_without_state_dict_change():
    reference = _model()
    prepared = _model()
    prepared.load_state_dict(reference.state_dict(), strict=True)

    context = torch.randn(2, 1, reference.context_dim)
    observation = torch.randn(2, 1, reference.config.n_features)

    with torch.inference_mode():
        expected_initial = reference.dist.initial.log_matrix(context=context, T=1)
        expected_duration = reference.dist.duration.log_matrix(
            context=context,
            T=1,
            soft_dmax=reference.duration_logits_bias,
        )
        expected_transition = reference.dist.transition.log_matrix(context=context, T=1)
        expected_emission = reference.dist.emission.log_prob(observation, context=context)

    state_keys = set(prepared.state_dict())
    prepare_inference(prepared, freeze=True)

    assert set(prepared.state_dict()) == state_keys
    for component in prepared.dist.children():
        assert not component.training
        assert getattr(component, "_inference_context_weight", None) is not None

    with torch.inference_mode():
        actual_initial = prepared.dist.initial.log_matrix(context=context, T=1)
        actual_duration = prepared.dist.duration.log_matrix(
            context=context,
            T=1,
            soft_dmax=prepared.duration_logits_bias,
        )
        actual_transition = prepared.dist.transition.log_matrix(context=context, T=1)
        actual_emission = prepared.dist.emission.log_prob(observation, context=context)

    torch.testing.assert_close(actual_initial, expected_initial, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual_duration, expected_duration, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual_transition, expected_transition, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual_emission, expected_emission, atol=1e-6, rtol=1e-6)


def test_context_fastpath_buffers_follow_dtype_moves_and_freeze_false_disables_them():
    model = prepare_inference(_model(seed=92), freeze=True)
    model = model.to(dtype=torch.float64)

    for component in model.dist.children():
        weight = getattr(component, "_inference_context_weight", None)
        assert weight is not None
        assert weight.dtype == torch.float64

    prepare_inference(model, freeze=False)
    for component in model.dist.children():
        assert getattr(component, "_inference_context_weight", None) is None
        assert getattr(component, "_inference_context_bias", None) is None
