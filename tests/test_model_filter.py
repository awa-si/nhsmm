from __future__ import annotations

import torch

from nhsmm import ModelConfig, NHSMM
from nhsmm.filtering import filter_model_sequence


def _make_model() -> NHSMM:
    config = ModelConfig(
        n_states=3,
        n_features=4,
        max_duration=5,
        causal=True,
        dropout=0.0,
        seed=19,
    )
    model = NHSMM(config, device="cpu")
    model.initialize_distributions()
    model.eval()
    return model


def test_model_filter_trace_is_normalized() -> None:
    torch.manual_seed(19)
    model = _make_model()
    x = torch.randn(2, 7, model.config.n_features)

    trace = filter_model_sequence(model, x)

    assert trace.log_posterior.shape == (2, 7, 3, 5)
    log_z = trace.log_posterior.flatten(2).logsumexp(dim=-1)
    assert torch.allclose(log_z, torch.zeros_like(log_z), atol=1e-6, rtol=1e-6)


def test_model_filter_is_prefix_invariant() -> None:
    torch.manual_seed(23)
    model = _make_model()
    x = torch.randn(1, 8, model.config.n_features)
    prefix_len = 5

    changed_future = x.clone()
    changed_future[:, prefix_len:] = torch.randn_like(changed_future[:, prefix_len:]) * 100.0

    trace_a = filter_model_sequence(model, x)
    trace_b = filter_model_sequence(model, changed_future)

    assert torch.allclose(
        trace_a.log_posterior[:, :prefix_len],
        trace_b.log_posterior[:, :prefix_len],
        atol=1e-6,
        rtol=1e-6,
    )


def test_causal_outputs_are_invariant_to_right_padding() -> None:
    torch.manual_seed(29)
    model = _make_model()
    short = torch.randn(4, model.config.n_features)
    long = torch.randn(7, model.config.n_features)

    single = model._build_sequence_set([short])
    batched = model._build_sequence_set([short, long])

    assert torch.equal(single.lengths, torch.tensor([4]))
    assert torch.equal(batched.lengths, torch.tensor([4, 7]))
    assert torch.allclose(single.contexts[0], batched.contexts[0, :4], atol=1e-6, rtol=1e-6)
    assert torch.allclose(single.log_probs[0], batched.log_probs[0, :4], atol=1e-6, rtol=1e-6)

    with torch.inference_mode():
        alpha_single = model.forward(single)
        alpha_batched = model.forward(batched)

    posterior_single = alpha_single[0, :4] - torch.logsumexp(
        alpha_single[0, :4].flatten(1), dim=1
    ).view(4, 1, 1)
    posterior_batched = alpha_batched[0, :4] - torch.logsumexp(
        alpha_batched[0, :4].flatten(1), dim=1
    ).view(4, 1, 1)
    assert torch.allclose(posterior_single, posterior_batched, atol=1e-6, rtol=1e-6)

    trace_single = filter_model_sequence(model, [short])
    trace_batched = filter_model_sequence(model, [short, long])
    assert torch.allclose(
        trace_single.log_posterior[0, :4],
        trace_batched.log_posterior[0, :4],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.isneginf(trace_batched.log_posterior[0, 4:]).all()


def test_model_filter_requires_eval_and_causal_mode() -> None:
    model = _make_model()
    x = torch.randn(1, 3, model.config.n_features)

    model.train()
    try:
        filter_model_sequence(model, x)
    except RuntimeError as exc:
        assert "eval mode" in str(exc)
    else:
        raise AssertionError("training-mode filtering must fail closed")

    model.eval()
    model.config.causal = False
    try:
        filter_model_sequence(model, x)
    except ValueError as exc:
        assert "causal" in str(exc)
    else:
        raise AssertionError("non-causal model filtering must fail closed")
