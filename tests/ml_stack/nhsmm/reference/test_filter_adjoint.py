"""Exact adjoint VJPs/higher derivatives against full-history autograd."""

import pytest
import torch
from nhsmm.models.filter_grad import _CausalPrefixAdjoint
from nhsmm.models.forward import _causal_prefix_serial


def inputs(duration, tail, independent=False):
    torch.manual_seed(1741)
    b, t, k = 3, 6, 3
    initial = torch.randn(b, k, dtype=torch.float64).log_softmax(-1).requires_grad_()
    emission = torch.randn(b, t, k, dtype=torch.float64).requires_grad_()
    transition = torch.randn(b, t, k, 1 if independent else duration, k, dtype=torch.float64)
    transition = transition.log_softmax(-1).requires_grad_()
    h = torch.rand(b, t, k, duration, dtype=torch.float64) * 0.8 + 0.1
    end = h.log()
    continuation = (1 - h).log()
    if not tail:
        end[..., -1] = 0
        continuation[..., -1] = -torch.inf
    return (
        initial,
        emission,
        transition,
        end.requires_grad_(),
        continuation.requires_grad_(),
        torch.tensor([6, 3, 0]),
        tail,
    )


@pytest.mark.parametrize("duration,tail", [(1, True), (2, False), (2, True), (8, False), (8, True)])
@pytest.mark.parametrize("independent", [False, True])
def test_every_input_vjp_matches_full_history_autograd(duration, tail, independent):
    args = inputs(duration, tail, independent)
    actual = _CausalPrefixAdjoint.apply(*args)
    expected = _causal_prefix_serial(*args, normalized=True)
    assert torch.equal(actual, expected)
    weights = torch.randn_like(actual)
    left = torch.autograd.grad((actual.exp() * weights).sum(), args[:5], retain_graph=True)
    right = torch.autograd.grad((expected.exp() * weights).sum(), args[:5], retain_graph=True)
    for a, b in zip(left, right, strict=True):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-10)
    assert all(torch.equal(g[2], torch.zeros_like(g[2])) for g in left)
    assert torch.equal(left[2][:, -1], torch.zeros_like(left[2][:, -1]))
    assert torch.equal(left[3][:, -1], torch.zeros_like(left[3][:, -1]))
    assert torch.equal(left[4][:, -1], torch.zeros_like(left[4][:, -1]))


def test_gradcheck_and_gradgradcheck():
    args = inputs(2, True)
    xs = tuple(
        p[:1, :3].detach().requires_grad_() if p.ndim > 2 else p[:1].detach().requires_grad_()
        for p in args[:5]
    )
    lengths = torch.tensor([3])

    def f(*ps):
        return _CausalPrefixAdjoint.apply(*ps, lengths, True).exp()

    assert torch.autograd.gradcheck(f, xs, eps=1e-6, atol=1e-5, rtol=1e-3, fast_mode=True)
    assert torch.autograd.gradgradcheck(f, xs, eps=1e-6, atol=1e-5, rtol=1e-3, fast_mode=True)


@pytest.mark.parametrize("tail", [False, True])
def test_hessian_vector_product_matches_serial(tail):
    args = inputs(3, tail)
    weights = torch.randn(3, 6, 3, 3, dtype=torch.float64)
    directions = [torch.randn_like(p) for p in args[:5]]
    outputs = []
    for fast in (True, False):
        alpha = (
            _CausalPrefixAdjoint.apply(*args)
            if fast
            else _causal_prefix_serial(*args, normalized=True)
        )
        loss = (alpha.exp() * weights).sum()
        gradients = torch.autograd.grad(loss, args[:5], create_graph=True)
        outputs.append(
            torch.autograd.grad(
                sum((g * v).sum() for g, v in zip(gradients, directions)),
                args[:5],
                allow_unused=True,
            )
        )
    for a, b in zip(*outputs, strict=True):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize("left_empty", [False, True])
def test_single_supported_logaddexp_has_finite_second_derivative(left_empty):
    from nhsmm.filtering import _safe_logaddexp

    x = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    empty = x.new_tensor(-torch.inf)
    result = _safe_logaddexp(empty, x) if left_empty else _safe_logaddexp(x, empty)
    assert result == x
    assert torch.autograd.gradcheck(
        lambda value: (
            _safe_logaddexp(empty, value) if left_empty else _safe_logaddexp(value, empty)
        ).exp(),
        (x,),
    )
    assert torch.autograd.gradgradcheck(
        lambda value: (
            _safe_logaddexp(empty, value) if left_empty else _safe_logaddexp(value, empty)
        ).exp(),
        (x,),
    )


@pytest.mark.parametrize("tail", [False, True])
def test_dense_batch_and_ragged_owner_values_and_gradients_agree(tail):
    args = inputs(3, tail)
    weights = torch.randn_like(_causal_prefix_serial(*args, normalized=True)[:1])
    dense = _CausalPrefixAdjoint.apply(*(p[:1] for p in args[:6]), tail)
    ragged = _causal_prefix_serial(*args, normalized=True)[:1]
    assert torch.equal(dense, ragged)
    a = torch.autograd.grad((dense.exp() * weights).sum(), args[:5], retain_graph=True)
    b = torch.autograd.grad((ragged.exp() * weights).sum(), args[:5], retain_graph=True)
    for left, right in zip(a, b, strict=True):
        torch.testing.assert_close(left, right, atol=1e-11, rtol=1e-10)


def test_long_dense_tail_prefix_selects_compiled_filter_step(monkeypatch):
    import nhsmm.models.forward as forward

    calls = []

    def capture(*args, **kwargs):
        calls.append(kwargs)
        _, emission, _, _, _, _, _ = args
        batch, steps, states = emission.shape
        duration = args[3].shape[-1]
        return emission.new_zeros(batch, steps, states, duration)

    monkeypatch.setattr(forward, "_causal_prefix_serial", capture)

    def run(steps):
        batch, states, duration = 1, 2, 2
        initial = torch.zeros(batch, states, dtype=torch.float64)
        emission = torch.zeros(batch, steps, states, dtype=torch.float64)
        transition = torch.zeros(batch, steps, states, duration, states, dtype=torch.float64)
        end = torch.full((batch, steps, states, duration), -0.5, dtype=torch.float64)
        continuation = torch.full_like(end, -0.5)
        lengths = torch.tensor([steps])
        return _CausalPrefixAdjoint.apply(
            initial, emission, transition, end, continuation, lengths, True
        )

    run(3)
    assert calls[-1]["compile_step"] is False
    run(16_384)
    assert calls[-1]["compile_step"] is True
    assert calls[-1]["direct_reductions"] is True
