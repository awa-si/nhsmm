"""HSMM large-age frontiers and supervised adjoints against independent chains."""

import numpy as np
import pytest
import torch
from nhsmm import NHSMM, HSMMFilterRuntime, ModelConfig
from nhsmm.filtering import _duration_log_hazard_from_log_p, filter_model_sequence
from nhsmm.models.filter_grad import _CausalPrefixAdjoint


def _array(value):
    return value.detach().cpu().numpy().astype(np.longdouble)


def _softmax(value):
    weights = np.exp(value - value.max(axis=-1, keepdims=True))
    return weights / weights.sum(axis=-1, keepdims=True)


def _large_age_model(dtype, duration, tail):
    model = NHSMM(
        ModelConfig.for_mode(
            "hsmm",
            n_states=3,
            n_features=2,
            max_duration=duration,
            duration_tail=tail,
            dropout=0,
            seed=991,
        ),
        device="cpu",
    )
    model.initialize_distributions()
    model.to(dtype=dtype).eval()
    generator = torch.Generator().manual_seed(723)
    with torch.no_grad():
        model.duration_logits_bias.zero_()
        for head in (model.dist.initial, model.dist.duration, model.dist.transition):
            head.log_temperature.zero_()
            head.logits.copy_(torch.randn(head.logits.shape, generator=generator, dtype=dtype))
        model.dist.emission.mu.copy_(
            torch.tensor([[-0.7, 0.3], [0.2, -0.4], [0.8, 0.6]], dtype=dtype)
        )
        model.dist.emission.log_var.fill_(0.2)
        if tail:
            model.dist.duration.tail_end_probability.copy_(
                torch.tensor([1e-4, 1.0, 0.35], dtype=dtype)
            )
    return model


def _expanded_reference(model, x):
    """Raw-weight expanded Markov chain; no production scores or hazard helpers."""
    k, d = model.config.n_states, model.config.max_duration
    pi = _softmax(_array(model.dist.initial.logits))
    q = _softmax(_array(model.dist.duration.logits))
    edges = _array(model.dist.transition.logits).copy()
    for state in range(k):
        edges[state, :, state] = -np.inf
    edges = _softmax(edges)
    tail = _array(model.dist.duration.tail_end_probability) if model.config.duration_tail else None
    matrix = np.zeros((k * d, k * d), dtype=np.longdouble)
    for state in range(k):
        for age in range(d):
            risk = (
                tail[state]
                if tail is not None and age == d - 1
                else q[state, age] / q[state, age:].sum()
            )
            for target in range(k):
                matrix[state * d + age, target * d] += risk * edges[state, age, target]
            if age < d - 1 or tail is not None:
                matrix[state * d + age, state * d + min(age + 1, d - 1)] += 1 - risk
    np.testing.assert_allclose(matrix.sum(-1), 1, rtol=0, atol=1e-15)
    means = _array(model.dist.emission.mu)
    variance = np.maximum(
        np.logaddexp(0, _array(model.dist.emission.log_var)), model.dist.emission.min_covar
    )
    density = -0.5 * (
        (_array(x)[:, None] - means) ** 2 / variance
        + np.log(variance)
        + np.log(2 * np.longdouble(str(np.pi)))
    ).sum(-1)
    p = np.zeros((k, d), dtype=np.longdouble)
    p[:, 0] = pi
    rows, evidence = [], np.longdouble(0)
    for t, row in enumerate(density):
        if t:
            p = (p.reshape(-1) @ matrix).reshape(k, d)
        offset = row.max()
        p *= np.exp(row - offset)[:, None]
        mass = p.sum()
        evidence += np.log(mass) + offset
        p /= mass
        rows.append(p.copy())
    return np.asarray(rows, dtype=np.float64), float(evidence)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("duration", [17, 64])
@pytest.mark.parametrize("tail", [False, True])
def test_large_age_batch_and_stream_against_expanded_chain(dtype, duration, tail, record_property):
    model = _large_age_model(dtype, duration, tail)
    t = torch.arange(1025, dtype=dtype)
    x = torch.stack((torch.sin(t / 19), torch.cos(t / 31)), -1)
    expected, evidence = _expanded_reference(model, x)
    trace = filter_model_sequence(model, x).log_posterior[0].exp()
    reference = torch.tensor(expected, dtype=dtype)
    tolerance = 2e-5 if dtype == torch.float32 else 2e-12
    torch.testing.assert_close(trace, reference, rtol=0, atol=tolerance)
    runtime = HSMMFilterRuntime(model)
    streamed = torch.stack(
        [runtime.step(row, timestamp=i).log_posterior[0].exp() for i, row in enumerate(x)]
    )
    torch.testing.assert_close(streamed, reference, rtol=0, atol=tolerance)
    likelihood = float(model.log_likelihood(x).detach()[0])
    assert abs(likelihood - evidence) / len(x) < (1e-5 if dtype == torch.float32 else 1e-10)
    assert runtime.state.filter_state.log_posterior.shape == (1, 3, duration)
    if tail:
        assert float(reference[-1, 0, -1]) > 0
    record_property("maximum_batch_posterior_error", float((trace - reference).abs().max()))
    record_property("maximum_stream_posterior_error", float((streamed - reference).abs().max()))


def _probability_frontiers(raw, lengths, tail):
    """Separate differentiable probability-space recurrence with raw normalization."""
    initial, emission, duration, transition, risk = raw
    pi = initial.exp()
    pi = pi / pi.sum(-1, keepdim=True)
    q = duration.exp()
    q = q / q.sum(-1, keepdim=True)
    allowed = ~torch.eye(3, dtype=torch.bool)
    matrix = transition.masked_fill(~allowed[None, None, :, None, :], -torch.inf).exp()
    matrix = matrix / matrix.sum(-1, keepdim=True)
    h = risk.sigmoid()
    output = []
    for b, length in enumerate(lengths.tolist()):
        if not length:
            output.append([])
            continue
        p = torch.cat((pi[b, :, None], pi.new_zeros(3, q.shape[-1] - 1)), -1)
        rows = []
        for t in range(length):
            if t:
                hazard = torch.stack(
                    [
                        q[b, t - 1, :, age] / q[b, t - 1, :, age:].sum(-1)
                        for age in range(q.shape[-1])
                    ],
                    -1,
                )
                if tail:
                    hazard = torch.cat((hazard[:, :-1], h[b, t - 1, :, None]), -1)
                reset = (p[:, :, None] * hazard[:, :, None] * matrix[b, t - 1]).sum((0, 1))
                continued = p * (1 - hazard)
                if tail:
                    ages = torch.cat(
                        (continued[:, :-2], (continued[:, -2] + continued[:, -1])[:, None]), -1
                    )
                else:
                    ages = continued[:, :-1]
                p = torch.cat((reset[:, None], ages), -1)
            p = p * emission[b, t].exp()[:, None]
            p = p / p.sum()
            rows.append(p)
        output.append(rows)
    return output


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("duration", [3, 7])
@pytest.mark.parametrize("tail", [False, True])
def test_ragged_dynamic_supervised_adjoint_against_probability_chain(dtype, duration, tail):
    generator = torch.Generator().manual_seed(199)
    b, t, k = 4, 29, 3
    lengths = torch.tensor([0, 1, 17, t])
    shapes = [(b, k), (b, t, k), (b, t, k, duration), (b, t, k, duration, k), (b, t, k)]
    raw = [
        (torch.randn(shape, generator=generator, dtype=dtype) * 0.3).requires_grad_()
        for shape in shapes
    ]
    initial, emission, duration_logits, transition_logits, risk = raw
    q = duration_logits.log_softmax(-1)
    end, continuation = _duration_log_hazard_from_log_p(
        q.reshape(b * t, k, duration), risk.sigmoid().reshape(b * t, k) if tail else None
    )
    allowed = ~torch.eye(k, dtype=torch.bool)
    transition = transition_logits.masked_fill(
        ~allowed[None, None, :, None, :], -torch.inf
    ).log_softmax(-1)
    actual = _CausalPrefixAdjoint.apply(
        initial.log_softmax(-1),
        emission,
        transition,
        end.reshape(b, t, k, duration),
        continuation.reshape(b, t, k, duration),
        lengths,
        tail,
    )
    reference_raw = [p.detach().clone().requires_grad_() for p in raw]
    reference = _probability_frontiers(reference_raw, lengths, tail)
    actual_loss = actual.new_zeros(())
    expected_loss = actual.new_zeros(())
    for batch, length in enumerate(lengths.tolist()):
        assert torch.isneginf(actual[batch, length:]).all()
        for row in range(length):
            label = (batch + row // 3) % k
            actual_loss = actual_loss - torch.logsumexp(actual[batch, row, label], -1) / 47
            expected_loss = expected_loss - reference[batch][row][label].sum().log() / 47
    tolerance = 2e-5 if dtype == torch.float32 else 2e-12
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=tolerance)
    higher = dtype == torch.float64
    left = torch.autograd.grad(actual_loss, raw, create_graph=higher, allow_unused=True)
    right = torch.autograd.grad(
        expected_loss, reference_raw, create_graph=higher, allow_unused=True
    )
    for parameter, a, e in zip(raw, left, right, strict=True):
        a = torch.zeros_like(parameter) if a is None else a
        e = torch.zeros_like(parameter) if e is None else e
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, e, rtol=0, atol=tolerance)
    if higher:
        directions = [torch.randn(p.shape, generator=generator, dtype=dtype) for p in raw]
        left_hvp = torch.autograd.grad(
            sum((g * v).sum() for g, v in zip(left, directions, strict=True) if g is not None),
            raw,
            allow_unused=True,
        )
        right_hvp = torch.autograd.grad(
            sum((g * v).sum() for g, v in zip(right, directions, strict=True) if g is not None),
            reference_raw,
            allow_unused=True,
        )
        for parameter, a, e in zip(raw, left_hvp, right_hvp, strict=True):
            a = torch.zeros_like(parameter) if a is None else a
            e = torch.zeros_like(parameter) if e is None else e
            assert torch.isfinite(a).all()
            torch.testing.assert_close(a, e, rtol=0, atol=2e-12)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("support", ["dense", "hard", "separated"])
def test_duration_survival_zero_sensitivity_second_derivatives(dtype, support):
    from nhsmm.filtering import _duration_log_survival

    scores = torch.linspace(-0.7, 0.8, 34, dtype=dtype).reshape(2, 17)
    if support == "hard":
        scores[:, -2:] = -torch.inf
        scores[:, 2::4] = -torch.inf
    elif support == "separated":
        scores[:, 1::3] -= 1000
    raw = scores.detach().requires_grad_()
    reference_raw = scores.detach().clone().requires_grad_()
    actual = _duration_log_survival(raw)
    reference = []
    for age in range(17):
        values = reference_raw[:, age:]
        offset = values.detach().amax(-1)
        offset = torch.where(torch.isfinite(offset), offset, torch.zeros_like(offset))
        suffix = (values - offset[:, None]).exp().sum(-1)
        safe = torch.where(suffix > 0, suffix, torch.ones_like(suffix))
        reference.append(torch.where(suffix > 0, safe.log() + offset, -torch.inf))
    expected = torch.stack(reference, -1)
    weights = torch.zeros_like(raw)
    weights[:, :7] = torch.linspace(-0.4, 0.7, 7, dtype=dtype)
    valid = torch.isfinite(expected)
    tolerance = 2e-5 if dtype == torch.float32 else 2e-12
    torch.testing.assert_close(actual, expected, rtol=0, atol=tolerance)
    left = torch.autograd.grad((actual[valid] * weights[valid]).sum(), raw, create_graph=True)[0]
    right = torch.autograd.grad(
        (expected[valid] * weights[valid]).sum(), reference_raw, create_graph=True
    )[0]
    torch.testing.assert_close(left, right, rtol=0, atol=tolerance)
    direction = torch.linspace(-0.3, 0.9, 34, dtype=dtype).reshape_as(raw)
    left_hvp = torch.autograd.grad((left * direction).sum(), raw)[0]
    right_hvp = torch.autograd.grad((right * direction).sum(), reference_raw)[0]
    assert torch.isfinite(left_hvp).all()
    torch.testing.assert_close(left_hvp, right_hvp, rtol=0, atol=tolerance)


@pytest.mark.parametrize("tail", [False, True])
def test_batch_hazards_bundled_only_for_used_hsmm_boundaries(tail, monkeypatch):
    from nhsmm import filtering

    model = _large_age_model(torch.float64, 17, tail)
    original = filtering._duration_log_hazard_from_log_p
    calls = []

    def counted(log_p, tail_end_probability=None):
        calls.append(tuple(log_p.shape))
        return original(log_p, tail_end_probability)

    monkeypatch.setattr(filtering, "_duration_log_hazard_from_log_p", counted)
    xs = [torch.zeros(n, 2, dtype=torch.float64) for n in [0, 1, 13, 19]]
    trace = filtering.filter_model_sequence(model, xs)
    assert calls == [(12, 3, 17), (18, 3, 17)]
    assert trace.lengths.tolist() == [0, 1, 13, 19]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_duration_survival_retained_backward_then_higher_derivatives(dtype):
    from nhsmm.filtering import _duration_log_survival

    raw = torch.linspace(-0.6, 0.9, 34, dtype=dtype).reshape(2, 17).requires_grad_()
    reference_raw = raw.detach().clone().requires_grad_()
    weights = torch.zeros_like(raw)
    weights[:, :7] = torch.linspace(-0.4, 0.7, 7, dtype=dtype)
    loss = (_duration_log_survival(raw) * weights).sum()
    reference = torch.stack([reference_raw[:, age:].exp().sum(-1).log() for age in range(17)], -1)
    expected_loss = (reference * weights).sum()
    expected = torch.autograd.grad(expected_loss, reference_raw, create_graph=True)[0]
    first = torch.autograd.grad(loss, raw, retain_graph=True)[0]
    repeated = torch.autograd.grad(loss, raw, retain_graph=True)[0]
    higher = torch.autograd.grad(loss, raw, create_graph=True)[0]
    tolerance = 2e-5 if dtype == torch.float32 else 2e-12
    torch.testing.assert_close(first, repeated, rtol=0, atol=0)
    torch.testing.assert_close(first, expected, rtol=0, atol=tolerance)
    torch.testing.assert_close(higher, expected, rtol=0, atol=tolerance)
    direction = torch.linspace(-0.3, 0.8, 34, dtype=dtype).reshape_as(raw)
    left = torch.autograd.grad((higher * direction).sum(), raw)[0]
    right = torch.autograd.grad((expected * direction).sum(), reference_raw)[0]
    assert torch.isfinite(left).all()
    torch.testing.assert_close(left, right, rtol=0, atol=tolerance)
