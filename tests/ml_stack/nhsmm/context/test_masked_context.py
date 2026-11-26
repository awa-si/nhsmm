"""Independent mixed-availability probability and gradient checks."""

import itertools

import pytest
import torch
from nhsmm import BASE_CONTEXT, NHSMM, CausalContext, HSMMFilterRuntime, ModelConfig
from nhsmm.context import ContextRouter
from nhsmm.filtering import filter_model_sequence


def model(D=2, tail=False):
    m = NHSMM(
        ModelConfig(
            n_states=3,
            n_features=1,
            max_duration=D,
            causal=True,
            allow_self_transitions=(D == 1 and not tail),
            duration_tail=tail,
            context_dim=2,
            seed=39,
            dropout=0,
        ),
        device="cpu",
    )
    m.initialize_distributions()
    m.double().eval()
    with torch.no_grad():
        for d in (m.dist.initial, m.dist.duration, m.dist.transition, m.dist.emission):
            d.delta_scale.fill_(0.7)
    return m


def fixture():
    x = torch.tensor([[-0.2, 0.4, 0.7], [0.3, -0.2, 0.6]], dtype=torch.float64)[..., None]
    u = torch.tensor(
        [[[0.1, 0.5], [0.8, -0.2], [-0.4, 0.3]], [[0.7, 0.2], [-0.6, 0.3], [0.2, -0.1]]],
        dtype=x.dtype,
    )
    e = torch.cat((u.new_zeros(2, 1, 2), u[:, :-1]), 1)
    mask = torch.tensor([[False, True, False], [True, False, True]])
    return x, CausalContext(e, u, mask)


def ingredients(m, x, c, b):
    pi = None
    em = []
    q = []
    A = []
    for t in range(x.shape[1]):
        e = c.emission[b : b + 1, t : t + 1] if c.available[b, t] else None
        u = c.boundary[b : b + 1, t : t + 1] if c.available[b, t] else None
        if t == 0:
            pi = m.dist.initial.log_matrix(context=e)[0, 0].exp()
        em.append(m.dist.emission.log_prob(x[b : b + 1, t : t + 1], context=e)[0, 0].exp())
        q.append(
            m.dist.duration.log_matrix(context=u, T=1, soft_dmax=m.duration_logits_bias)[0, 0].exp()
        )
        A.append(m.dist.transition.log_matrix(context=u, T=1)[0, 0].exp())
    return pi, em, q, A


def product(m, z, pi, em, q, A):
    D = m.config.max_duration
    tail = m.config.duration_tail
    p = pi[z[0]] * em[0][z[0]]
    age = 1
    for t in range(1, len(z)):
        i, j = z[t - 1], z[t]
        a = min(age, D) - 1
        h = (
            m.dist.duration.tail_probability()[i]
            if tail and age >= D
            else q[t - 1][i, a] / q[t - 1][i, a:].sum()
        )
        if (D == 1 and not tail) or i != j:
            p = p * h * A[t - 1][i, a, j]
            age = 1
        else:
            p = p * (1 - h)
            age += 1
        p = p * em[t][j]
        if age > D and not tail:
            return p * 0, age
    return p, age


@pytest.mark.parametrize("D,tail", [(1, False), (2, False), (2, True)])
def test_joint_marginal_map_and_stream_match_independent_products(D, tail):
    m = model(D, tail)
    x, c = fixture()
    trace = filter_model_sequence(m, x, context=c)
    scores = m.log_likelihood(x, context=c)
    paths = m.predict(x, context=c, verbose=False)
    rt = HSMMFilterRuntime(m)
    for t in range(3):
        state = rt.step(
            x[:, t],
            context=CausalContext(
                c.emission[:, t : t + 1], c.boundary[:, t : t + 1], c.available[:, t : t + 1]
            ),
        )
        torch.testing.assert_close(state.log_posterior, trace.log_posterior[:, t])
    for b in range(2):
        args = ingredients(m, x, c, b)
        values = []
        front = x.new_zeros(3, D)
        zs = []
        for z in itertools.product(range(3), repeat=3):
            p, age = product(m, z, *args)
            values.append(p)
            zs.append(z)
            joint = m.supervised_log_likelihood(
                x[b],
                torch.tensor(z),
                context=CausalContext(c.emission[b], c.boundary[b], c.available[b]),
            )[0]
            torch.testing.assert_close(joint.exp(), p, rtol=1e-10, atol=1e-12)
            if age <= D or tail:
                front[z[-1], min(age, D) - 1] += p
        total = torch.stack(values).sum()
        torch.testing.assert_close(scores[b].exp(), total, rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(trace.log_posterior[b, -1].exp(), front / total)
        expected = zs[int(torch.stack(values).argmax())]
        assert tuple(paths[b].tolist()) == expected


def test_unavailable_nan_values_and_gradients_are_excluded():
    m = model()
    x, c = fixture()
    e = c.emission.clone().requires_grad_()
    u = c.boundary.clone().requires_grad_()
    score = m.log_likelihood(x, context=CausalContext(e, u, c.available)).sum()
    de, du = torch.autograd.grad(score, (e, u))
    assert torch.equal(de[~c.available], torch.zeros_like(de[~c.available]))
    assert torch.equal(du[~c.available], torch.zeros_like(du[~c.available]))
    assert de[c.available].abs().sum() > 0 and du[c.available].abs().sum() > 0
    missing = CausalContext(
        e.detach().masked_fill(~c.available[..., None], float("nan")),
        u.detach().masked_fill(~c.available[..., None], float("inf")),
        c.available,
    )
    torch.testing.assert_close(m.log_likelihood(x, context=missing).sum(), score)
    assert torch.isfinite(de).all() and torch.isfinite(du).all()


def test_all_missing_equals_base_and_never_evaluates_heads(monkeypatch):
    m = model()
    x, c = fixture()
    expected = m.log_likelihood(x, context=BASE_CONTEXT)

    def fail(*args, **kwargs):
        raise AssertionError("masked head executed")

    for d in (m.dist.initial, m.dist.duration, m.dist.transition, m.dist.emission):
        monkeypatch.setattr(d.context_net, "forward", fail)
    c = CausalContext(
        torch.full_like(c.emission, float("nan")),
        torch.full_like(c.boundary, float("inf")),
        torch.tensor(False),
    )
    torch.testing.assert_close(m.log_likelihood(x, context=c), expected)
    rt = HSMMFilterRuntime(m)
    for t in range(3):
        rt.step(
            x[:, t],
            context=CausalContext(
                c.emission[:, t : t + 1], c.boundary[:, t : t + 1], torch.tensor(False)
            ),
        )
    with pytest.raises(ValueError, match="at least one available"):
        m.fit_supervised(
            x, torch.tensor([[0, 0, 1], [1, 2, 2]]), context=c, steps=1, context_only=True
        )


def test_ragged_selection_chunking_and_router_copy_preserve_mask():
    m = model()
    x, c = fixture()
    xs = [x[0], x[1, :2]]
    ragged = CausalContext(
        [c.emission[0], c.emission[1, :2]],
        [c.boundary[0], c.boundary[1, :2]],
        [c.available[0], c.available[1, :2]],
    )
    seq = m._build_sequence_set(xs, ragged)
    router = ContextRouter.from_tensor(seq)
    for item in (router.clone(), router.detach(), router.select([0, 1])):
        assert torch.equal(item.context_mask, seq.context_mask)
    aligned = m._align_external_context(xs, ragged)
    expected = m.log_likelihood(xs, context=ragged)
    for b, (chunk, ctx) in enumerate(m._optimization_chunks(xs, aligned, 1)):
        torch.testing.assert_close(m.log_likelihood(chunk, context=ctx)[0], expected[b])


@pytest.mark.parametrize(
    "mask",
    [torch.ones(2, 3), torch.ones(2, 2, dtype=torch.bool), [torch.ones(3, dtype=torch.bool)]],
)
def test_invalid_masks_reject_before_runtime_mutation(mask):
    m = model()
    x, c = fixture()
    rt = HSMMFilterRuntime(m)
    with pytest.raises((TypeError, ValueError)):
        m.log_likelihood(x, context=CausalContext(c.emission, c.boundary, mask))
    assert rt.state is None


def test_masked_context_only_fit_keeps_base_parameters_fixed():
    m = model()
    x, c = fixture()
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    summary = m.fit_supervised(
        x, torch.tensor([[0, 0, 1], [1, 2, 2]]), context=c, steps=3, lr=0.001, context_only=True
    )
    assert summary["updates"] == 3 and summary["final_nll_per_row"] < summary["initial_nll_per_row"]
    changed = []
    for n, p in m.named_parameters():
        if not torch.equal(p, before[n]):
            changed.append(n)
            assert ".context_net." in n or "._proj." in n or n.endswith(".delta_scale")
    assert changed and not m.training


def test_all_missing_transition_refinement_rejects_before_updates():
    from dataclasses import replace

    m = model()
    x, c = fixture()
    c = CausalContext(c.emission, c.boundary, torch.tensor(False))
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    with pytest.raises(ValueError, match="available context"):
        m._refine_transition_likelihood(x, c, steps=1, lr=0.01)
    cfg = replace(m.config, transition_refine_steps=1, max_iter=1, n_init=1)
    with pytest.raises(ValueError, match="available context"):
        m.optimize(x, context=c, cfg=cfg)
    assert all(torch.equal(p, before[n]) for n, p in m.named_parameters())


@pytest.mark.parametrize("api", ["batch", "stream"])
def test_unsupported_distribution_rejects_mask_before_scoring(monkeypatch, api):
    m = model()
    x, c = fixture()
    monkeypatch.setattr(m.dist.emission, "_masked_context_delta", None)
    with pytest.raises(ValueError, match="row-gating support"):
        if api == "batch":
            m.log_likelihood(x, context=c)
        else:
            rt = HSMMFilterRuntime(m)
            rt.step(
                x[:, 0],
                context=CausalContext(c.emission[:, :1], c.boundary[:, :1], c.available[:, :1]),
            )
            assert rt.state is None
