"""Base-score bypass and intermittent streaming probability products."""

import copy
import itertools

import pytest
import torch
from nhsmm import BASE_CONTEXT, NHSMM, CausalContext, HSMMFilterRuntime, ModelConfig
from nhsmm.filtering import filter_model_sequence


def model(encoder=False, tail=False, D=3):
    m = NHSMM(
        ModelConfig(
            n_states=2,
            n_features=1,
            max_duration=D,
            duration_tail=tail,
            allow_self_transitions=(D == 1 and not tail),
            context_dim=4,
            use_context_encoder=encoder,
            causal=True,
            dropout=0,
            seed=12,
        ),
        device="cpu",
    )
    m.initialize_distributions()
    with torch.no_grad():
        for component in (m.dist.initial, m.dist.duration, m.dist.transition, m.dist.emission):
            component.delta_scale.fill_(1)
    return m.double().eval()


@pytest.mark.parametrize("encoder", [False, True])
def test_base_bypasses_encoder_and_heads_with_batch_parity(encoder, monkeypatch):
    m = model(encoder)
    reference = copy.deepcopy(m)
    reference.encoder = None
    x = torch.linspace(-0.4, 0.7, 10, dtype=torch.float64).reshape(2, 5, 1)

    def fail(*args, **kwargs):
        raise AssertionError("unavailable context network must not run")

    if m.encoder is not None:
        monkeypatch.setattr(m.encoder, "encode", fail)
        monkeypatch.setattr(m.encoder.encoder, "stream_step", fail)
    for component in (m.dist.initial, m.dist.duration, m.dist.transition, m.dist.emission):
        monkeypatch.setattr(component.context_net, "forward", fail)
    torch.testing.assert_close(
        m.log_likelihood(x, context=BASE_CONTEXT), reference.log_likelihood(x)
    )
    trace = filter_model_sequence(m, x, context=BASE_CONTEXT)
    rt = HSMMFilterRuntime(m)
    for t in range(x.shape[1]):
        state = rt.step(x[:, t], context=BASE_CONTEXT)
        torch.testing.assert_close(state.log_posterior, trace.log_posterior[:, t])
    aligned = m._align_external_context(x, BASE_CONTEXT)
    assert all(c is BASE_CONTEXT for _, c in m._optimization_chunks(x, aligned, 1))


@pytest.mark.parametrize("D,tail", [(1, False), (3, False), (2, True)])
def test_intermittent_context_matches_probability_products(D, tail):
    m = model(D=D, tail=tail)
    x = torch.tensor([-0.5, 0.2, 0.3, -0.1], dtype=torch.float64)[:, None]
    u = torch.arange(16, dtype=torch.float64).reshape(4, 4) / 10
    schedule = [
        CausalContext(u[0] * 0, u[0]),
        BASE_CONTEXT,
        CausalContext(u[1], u[2]),
        BASE_CONTEXT,
    ]
    pi = m.dist.initial.log_matrix(context=u[0:1].unsqueeze(0))[0, 0].exp()
    emissions = []
    qs = []
    As = []
    rt = HSMMFilterRuntime(m)
    for t, c in enumerate(schedule):
        e = None if c is BASE_CONTEXT else c.emission.reshape(1, 1, 4)
        b = None if c is BASE_CONTEXT else c.boundary.reshape(1, 1, 4)
        # Initial E[0] is zero, as declared in the schedule.
        if t == 0:
            pi = m.dist.initial.log_matrix(context=e)[0, 0].exp()
        emissions.append(m.dist.emission.log_prob(x[t : t + 1].unsqueeze(0), context=e)[0, 0].exp())
        qs.append(
            m.dist.duration.log_matrix(context=b, T=1, soft_dmax=m.duration_logits_bias)[0, 0].exp()
        )
        As.append(m.dist.transition.log_matrix(context=b, T=1)[0, 0].exp())
        mass = x.new_zeros(2, D)
        for z in itertools.product(range(2), repeat=t + 1):
            p = pi[z[0]] * emissions[0][z[0]]
            age = 1
            for j in range(1, len(z)):
                i, k = z[j - 1], z[j]
                a = min(age, D) - 1
                h = (
                    m.dist.duration.tail_probability()[i]
                    if tail and age >= D
                    else qs[j - 1][i, a] / qs[j - 1][i, a:].sum()
                )
                if D == 1 and not tail or i != k:
                    p = p * h * As[j - 1][i, a, k]
                    age = 1
                else:
                    p = p * (1 - h)
                    age += 1
                p = p * emissions[j][k]
                if age > D and not tail:
                    p = p * 0
                    break
            if age <= D or tail:
                mass[z[-1], min(age, D) - 1] += p
        state = rt.step(x[t], context=c)
        torch.testing.assert_close(state.log_posterior[0].exp(), mass / mass.sum())
        torch.testing.assert_close(rt.state.duration_log_prob.exp()[0], qs[t])
    # None still means internal mode, not an implicit switch of protocol.
    with pytest.raises(ValueError, match="context mode"):
        rt.step(x[-1], context=None)


def test_base_is_not_zero_context_and_neural_only_fit_rejects_it():
    m = model()
    x = torch.ones(3, 1, dtype=torch.float64)
    with torch.no_grad():
        m.dist.emission.context_net[-1].bias.copy_(torch.tensor([1.0, -1.0]))
    a = m._build_sequence_set(x, BASE_CONTEXT).log_probs
    b = m._build_sequence_set(x, torch.zeros(4, dtype=x.dtype)).log_probs
    assert not torch.allclose(a, b)
    with pytest.raises(ValueError, match="active external context"):
        m.fit_supervised(
            x, torch.tensor([0, 0, 1]), context=BASE_CONTEXT, context_only=True, steps=1
        )
