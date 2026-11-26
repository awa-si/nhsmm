"""Timing and probability oracles independent of production recursion."""

import itertools

import pytest
import torch
from nhsmm import NHSMM, CausalContext, HSMMFilterRuntime, ModelConfig
from nhsmm.filtering import filter_model_sequence


def model(D=3, tail=False):
    m = NHSMM(
        ModelConfig(
            n_states=2,
            n_features=1,
            max_duration=D,
            duration_tail=tail,
            causal=True,
            allow_self_transitions=(D == 1 and not tail),
            context_dim=1,
            seed=17,
            dropout=0,
        ),
        device="cpu",
    )
    m.initialize_distributions()
    return m.double().eval()


def inputs():
    x = torch.tensor([-0.3, 0.2, 0.4, -0.1], dtype=torch.float64)[:, None]
    u = torch.tensor([0.7, -0.4, 0.9, 0.2], dtype=torch.float64)[:, None]
    e = torch.cat((u.new_zeros(1, 1), u[:-1]))
    return x, CausalContext(e, u)


@pytest.mark.parametrize("D,tail", [(1, False), (3, False), (2, True)])
def test_probability_product_and_stream_oracles(D, tail):
    m = model(D, tail)
    x, c = inputs()
    pi = m.dist.initial.log_matrix(context=c.emission[:1].unsqueeze(0))[0, 0].exp()
    q = m.dist.duration.log_matrix(
        context=c.boundary.unsqueeze(0), T=len(x), soft_dmax=m.duration_logits_bias
    )[0].exp()
    A = m.dist.transition.log_matrix(context=c.boundary.unsqueeze(0), T=len(x))[0].exp()
    b = m.dist.emission.log_prob(x.unsqueeze(0), context=c.emission.unsqueeze(0))[0].exp()
    total = x.new_zeros(())
    frontier = x.new_zeros(2, D)
    for z in itertools.product(range(2), repeat=len(x)):
        p, age = pi[z[0]] * b[0, z[0]], 1
        for t in range(1, len(x)):
            i, j = z[t - 1], z[t]
            if age > D and not tail:
                p = p * 0
                break
            a = min(age, D) - 1
            h = (
                m.dist.duration.tail_probability()[i]
                if tail and age >= D
                else q[t - 1, i, a] / q[t - 1, i, a:].sum()
            )
            ended = D == 1 and not tail or i != j
            if ended:
                p = p * h * A[t - 1, i, a, j]
                age = 1
            else:
                p = p * (1 - h)
                age += 1
            p = p * b[t, j]
        zt = torch.tensor(z)
        torch.testing.assert_close(m.supervised_log_likelihood(x, zt, context=c)[0].exp(), p)
        total = total + p
        if age <= D or tail:
            frontier[z[-1], min(age, D) - 1] += p
    torch.testing.assert_close(m.log_likelihood(x, context=c)[0].exp(), total)
    trace = filter_model_sequence(m, x, context=c)
    torch.testing.assert_close(trace.log_posterior[0, -1].exp(), frontier / total)
    runtime = HSMMFilterRuntime(m)
    for t in range(len(x)):
        state = runtime.step(x[t], context=CausalContext(c.emission[t], c.boundary[t]))
        torch.testing.assert_close(state.log_posterior[0], trace.log_posterior[0, t])
    torch.testing.assert_close(state.log_posterior.exp().sum(), torch.ones((), dtype=x.dtype))


def test_component_isolation_legacy_and_gradients():
    m = model()
    x, c = inputs()
    legacy = m._build_sequence_set(x, c.boundary)
    same = m._build_sequence_set(x, CausalContext(c.boundary, c.boundary))
    torch.testing.assert_close(legacy.log_probs, same.log_probs)
    e, u = c.emission.clone().requires_grad_(), c.boundary.clone().requires_grad_()
    data = m._build_sequence_set(x, CausalContext(e, u))
    de, du = torch.autograd.grad(data.log_probs.sum(), (e, u), allow_unused=True)
    assert de is not None and torch.isfinite(de).all() and du is None
    changed = m._build_sequence_set(x, CausalContext(e, u + 100))
    torch.testing.assert_close(data.log_probs, changed.log_probs)
    torch.testing.assert_close(data.canonical, changed.canonical)
    grad = torch.autograd.grad(m.log_likelihood(x, context=CausalContext(e, u)).sum(), (e, u))
    assert all(torch.isfinite(g).all() for g in grad)


def test_frozen_base_and_flags_restored():
    m = model()
    x, c = inputs()
    flags = {n: p.requires_grad for n, p in m.named_parameters()}
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    summary = m.fit_supervised(
        x, torch.tensor([0, 0, 1, 1]), context=c, steps=2, lr=0.001, context_only=True
    )
    assert summary["context_only"] and not m.training
    changed = []
    for n, p in m.named_parameters():
        assert p.requires_grad == flags[n]
        if not torch.equal(before[n], p):
            changed.append(n)
            assert ".context_net." in n or "._proj." in n or n.endswith(".delta_scale")
    assert changed


def test_ragged_and_chunks():
    m = model()
    x, c = inputs()
    xs = [x, x[:2]]
    context = CausalContext([c.emission, c.emission[:2]], [c.boundary, c.boundary[:2]])
    aligned = m._align_external_context(xs, context)
    chunks = list(m._optimization_chunks(xs, aligned, 1))
    whole = m.log_likelihood(xs, context=context)
    for i, (chunk, ctx) in enumerate(chunks):
        torch.testing.assert_close(m.log_likelihood(chunk, context=ctx)[0], whole[i])


@pytest.mark.parametrize("field", ["emission", "boundary"])
def test_missing_and_nonfinite_fail_atomically(field):
    m = model()
    x, c = inputs()
    values = dict(emission=c.emission, boundary=c.boundary)
    values[field] = None
    with pytest.raises(TypeError):
        CausalContext(**values)
    values[field] = torch.full_like(c.emission, float("nan"))
    bad = CausalContext(**values)
    with pytest.raises(ValueError):
        m.log_likelihood(x, context=bad)
    rt = HSMMFilterRuntime(m)
    rt.step(x[0], context=CausalContext(c.emission[0], c.boundary[0]))
    before = rt.state
    with pytest.raises(ValueError):
        rt.step(x[1], context=CausalContext(bad.emission[0], bad.boundary[0]))
    assert rt.state is before


def ohlcv_context(close):
    """Synthetic 1m close times: completed 5m/15m/1h bars only, no forward fill."""
    rows = []
    for end in range(60, len(close) + 1, 5):
        # One scalar representation of 1m detail and completed higher TF closes.
        minute_mean = close[end - 5 : end].mean()
        close15 = close[(end // 15) * 15 - 1]
        close60 = close[(end // 60) * 60 - 1]
        rows.append(minute_mean + close15 + close60)
    return torch.stack(rows)[:, None]


def test_completed_multiframe_features_have_no_future_dependency():
    close = torch.arange(1, 91, dtype=torch.float64)
    boundary = ohlcv_context(close)
    altered = close.clone()
    altered[70:] += 10000
    torch.testing.assert_close(boundary[:3], ohlcv_context(altered)[:3])
    # At 70m, the 15m and 1h values still come from their 60m closes.
    assert boundary[2, 0] == close[65:70].mean() + close[59] * 2
    emission = torch.cat((boundary.new_zeros(1, 1), boundary[:-1]))
    assert emission[3, 0] == boundary[2, 0]


def test_context_only_restores_flags_on_failed_update(monkeypatch):
    m = model()
    x, c = inputs()
    flags = {n: p.requires_grad for n, p in m.named_parameters()}

    def fail(*args, **kwargs):
        raise RuntimeError("injected optimizer failure")

    monkeypatch.setattr(torch.optim.Adam, "step", fail)
    with pytest.raises(RuntimeError, match="injected"):
        m.fit_supervised(x, torch.tensor([0, 0, 1, 1]), context=c, steps=1, context_only=True)
    assert not m.training
    assert flags == {n: p.requires_grad for n, p in m.named_parameters()}


def test_noncausal_and_misaligned_context_are_rejected():
    m = model()
    x, c = inputs()
    with pytest.raises(ValueError):
        m.log_likelihood(x, context=CausalContext(c.emission[:-1], c.boundary))
    m.config.causal = False
    with pytest.raises(ValueError, match="causal=True"):
        m.log_likelihood(x, context=c)
