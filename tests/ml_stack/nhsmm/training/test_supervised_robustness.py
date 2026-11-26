"""Independent HMM bridge, neural finite differences and cold-start contracts."""

import itertools
import math
from copy import deepcopy

import pytest
import torch
from nhsmm import NHSMM, ModelConfig
from nhsmm.filtering import filter_model_sequence


def make(D=1, tail=False, context=False, encoder=False):
    m = NHSMM(
        ModelConfig(
            n_states=3,
            n_features=1,
            max_duration=D,
            duration_tail=tail,
            allow_self_transitions=(D == 1 and not tail),
            causal=True,
            use_context_encoder=encoder,
            context_dim=4 if encoder else (1 if context else None),
            hidden_dim=4 if encoder else None,
            n_heads=1,
            dropout=0,
            seed=23,
        ),
        device="cpu",
    )
    m.initialize_distributions()
    m.double().eval()
    return m


@pytest.mark.parametrize("T", [1, 2, 5, 20])
def test_hmm_is_exact_geometric_hsmm(T):
    hmm, hsmm = make(), make(tail=True)
    A = torch.tensor([[0.6, 0.1, 0.3], [0.2, 0.5, 0.3], [0.1, 0.2, 0.7]], dtype=torch.float64)
    with torch.no_grad():
        hmm.dist.transition.logits.copy_(A.log()[:, None, :])
        hsmm.dist.initial.logits.copy_(hmm.dist.initial.logits)
        hsmm.dist.emission.load_state_dict(hmm.dist.emission.state_dict())
        hsmm.dist.transition.logits.copy_(A.log()[:, None, :])
        hsmm.dist.duration.tail_end_probability.copy_(1 - A.diag())
    x = torch.linspace(-0.5, 0.8, T, dtype=torch.float64)[:, None]
    torch.testing.assert_close(
        hmm.log_likelihood(x), hsmm.log_likelihood(x), atol=1e-12, rtol=1e-12
    )
    torch.testing.assert_close(
        filter_model_sequence(hmm, x).state_posterior,
        filter_model_sequence(hsmm, x).state_posterior,
        atol=1e-12,
        rtol=1e-12,
    )
    for z in itertools.islice(itertools.product(range(3), repeat=T), 30):
        labels = torch.tensor(z)
        torch.testing.assert_close(
            hmm.supervised_log_likelihood(x, labels),
            hsmm.supervised_log_likelihood(x, labels),
            atol=1e-12,
            rtol=1e-12,
        )


def independent_dynamic_joint(m, x, z, c):
    """Probability products, suffix sums and Gaussian density; no hazard kernels."""
    with torch.no_grad():
        pi = m.dist.initial.log_matrix(context=c[None, :1])[0, 0].exp().numpy()
        q = (
            m.dist.duration.log_matrix(context=c[None], soft_dmax=m.duration_logits_bias)[0]
            .exp()
            .numpy()
        )
        A = m.dist.transition.log_matrix(context=c[None])[0].exp().numpy()
        loc, covariance = m.dist.emission(context=c[None])
        loc = loc[0, :, :, 0].numpy()
        var = covariance[0, :, :, 0, 0].numpy()
        value = float(pi[z[0]])
        age = 0
        for t, state in enumerate(z.tolist()):
            value *= math.exp(
                -0.5 * (float(x[t, 0]) - loc[t, state]) ** 2 / var[t, state]
            ) / math.sqrt(2 * math.pi * var[t, state])
            if t == len(z) - 1:
                break
            hazard = q[t, state, age] / q[t, state, age:].sum()
            if z[t + 1] != state:
                value *= hazard * A[t, state, age, int(z[t + 1])]
                age = 0
            else:
                value *= 1 - hazard
                age += 1
        return math.log(value)


def test_neural_gradients_independent_finite_differences():
    m = make(D=3, context=True)
    x = torch.tensor([[-0.2], [0.4], [0.3], [-0.1], [0.2]], dtype=torch.float64)
    c = torch.tensor([[0.7], [-0.9], [0.2], [0.8], [-0.4]], dtype=torch.float64)
    z = torch.tensor([0, 0, 1, 2, 2])
    score = m.supervised_log_likelihood(x, z, context=c).sum()
    assert abs(float(score.detach()) - independent_dynamic_joint(m, x, z, c)) < 1e-12
    probes = [
        (m.dist.initial.context_net[-1].bias, 0),
        (m.dist.duration.context_net[-1].bias, 1),
        (m.dist.transition.context_net[-1].bias, 4),
        (m.dist.emission.context_net[-1].bias, 0),
        (m.dist.duration._proj.weight, 0),
        (m.dist.emission.mu, 1),
    ]
    gradients = torch.autograd.grad(score, [p for p, _ in probes])
    for (p, index), gradient in zip(probes, gradients, strict=True):
        original = p.detach().reshape(-1)[index].item()
        with torch.no_grad():
            p.reshape(-1)[index] = original + 1e-5
        plus = independent_dynamic_joint(m, x, z, c)
        with torch.no_grad():
            p.reshape(-1)[index] = original - 1e-5
        minus = independent_dynamic_joint(m, x, z, c)
        with torch.no_grad():
            p.reshape(-1)[index] = original
        assert float(gradient.reshape(-1)[index]) == pytest.approx(
            (plus - minus) / 2e-5, abs=2e-8, rel=2e-7
        )


def test_internal_causal_encoder_observed_paths_sum_to_marginal_and_differentiate():
    m = make(D=2, encoder=True)
    x = torch.tensor([[-0.4], [0.2], [0.5]], dtype=torch.float64)
    values = [
        m.supervised_log_likelihood(x, torch.tensor(z))[0]
        for z in itertools.product(range(3), repeat=3)
    ]
    marginal = torch.logsumexp(torch.stack(values), 0)
    torch.testing.assert_close(marginal, m.log_likelihood(x)[0], atol=1e-12, rtol=1e-12)
    parameters = list(m.encoder.parameters())
    gradients = torch.autograd.grad(marginal, parameters, allow_unused=True)
    assert any(g is not None and bool(g.abs().sum() > 0) for g in gradients)
    assert all(g is None or bool(torch.isfinite(g).all()) for g in gradients)


@pytest.mark.parametrize("mode", ["random", "spread", "kmeans", "kmeans_cluster_variance"])
@pytest.mark.parametrize("kind", ["gaussian", "studentt"])
def test_emission_default_initialization_honors_configuration(mode, kind):
    m = NHSMM(
        ModelConfig(n_states=3, n_features=2, emission_init_mode=mode, emission_type=kind, seed=31),
        device="cpu",
    )
    m.initialize_distributions()
    actual = m.dist.emission
    expected = deepcopy(actual)
    torch.manual_seed(84)
    actual.initialize()
    torch.manual_seed(84)
    expected.initialize(mode=mode if mode in ("random", "spread") else "spread")
    for name, value in actual.state_dict().items():
        assert torch.equal(value, expected.state_dict()[name])
