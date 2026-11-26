"""Independent observed-path checks; no production forward in the oracle."""

import itertools
import math

import pytest
import torch
from nhsmm import NHSMM, ModelConfig, build_artifact, load_inference_model


def model(D=3, tail=False, context=False):
    m = NHSMM(
        ModelConfig(
            n_states=2,
            n_features=1,
            max_duration=D,
            causal=True,
            duration_tail=tail,
            allow_self_transitions=(D == 1 and not tail),
            context_dim=1 if context else None,
            seed=17,
            dropout=0,
        ),
        device="cpu",
    )
    m.initialize_distributions()
    m.double().eval()
    return m


def reference(m, x, z):
    """Independent probability products, homogeneous distributions only."""
    pi = m.dist.initial.logits.softmax(-1)
    A = m.dist.transition.logits.clone()
    if not m.config.allow_self_transitions:
        A = A.masked_fill(torch.eye(2, dtype=torch.bool)[:, None, :], -torch.inf)
    A = A.softmax(-1)
    q = (m.dist.duration.logits + torch.sigmoid(m.duration_logits_bias).log()).softmax(-1)
    var = torch.nn.functional.softplus(m.dist.emission.log_var)
    emission = torch.exp(
        -0.5
        * (
            ((x[:, None, :] - m.dist.emission.mu) ** 2 / var) + var.log() + math.log(2 * math.pi)
        ).sum(-1)
    )
    p = pi[z[0]] * emission[0, z[0]]
    age = 1
    D = m.config.max_duration
    for t in range(1, len(z)):
        previous = z[t - 1]
        if age > D and not m.config.duration_tail:
            return p * 0
        index = min(age, D) - 1
        if m.config.duration_tail and age >= D:
            hazard = m.dist.duration.tail_probability()[previous]
        else:
            hazard = q[previous, index] / q[previous, index:].sum()
        ended = m.config.allow_self_transitions or z[t] != previous
        if ended:
            p = p * hazard * A[previous, index, z[t]]
            age = 1
        else:
            p = p * (1 - hazard)
            age += 1
        p = p * emission[t, z[t]]
    return p


@pytest.mark.parametrize("D,tail", [(1, False), (2, False), (3, False), (1, True), (2, True)])
@pytest.mark.parametrize("T", [1, 2, 4])
def test_all_paths_match_probability_oracle_and_marginal(D, tail, T):
    m = model(D, tail)
    x = torch.linspace(-0.4, 0.7, T, dtype=torch.float64)[:, None]
    scores = []
    for path in itertools.product(range(2), repeat=T):
        z = torch.tensor(path)
        actual = m.supervised_log_likelihood(x, z)[0]
        expected = reference(m, x, z)
        torch.testing.assert_close(actual.exp(), expected, atol=1e-13, rtol=1e-12)
        scores.append(actual)
    torch.testing.assert_close(
        torch.logsumexp(torch.stack(scores), 0), m.log_likelihood(x)[0], atol=1e-12, rtol=1e-12
    )


@pytest.mark.parametrize("D,tail", [(1, False), (3, False), (2, True)])
def test_path_gradients_match_probability_products(D, tail):
    m = model(D, tail)
    x = torch.tensor([[-0.2], [0.4], [0.1], [-0.3]], dtype=torch.float64)
    z = torch.tensor([0, 0, 1, 1])
    parameters = [
        m.dist.initial.logits,
        m.dist.duration.logits,
        m.dist.transition.logits,
        m.dist.emission.mu,
        m.dist.emission.log_var,
    ]
    if tail:
        parameters.append(m.dist.duration.tail_end_probability)
    a = torch.autograd.grad(m.supervised_log_likelihood(x, z).sum(), parameters, allow_unused=True)
    b = torch.autograd.grad(reference(m, x, z).log(), parameters, allow_unused=True)
    for p, left, right in zip(parameters, a, b, strict=True):
        left = torch.zeros_like(p) if left is None else left
        right = torch.zeros_like(p) if right is None else right
        torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-12)


def test_ragged_and_censoring():
    m = model(3)
    xs = [torch.zeros(1, 1), torch.zeros(3, 1), torch.zeros(0, 1)]
    zs = [torch.tensor([1]), torch.tensor([0, 0, 0]), torch.empty(0, dtype=torch.long)]
    batch = m.supervised_log_likelihood(xs, zs)
    individual = torch.stack(
        [m.supervised_log_likelihood(x, z)[0] for x, z in zip(xs, zs, strict=True)]
    )
    torch.testing.assert_close(batch, individual)
    assert batch[-1] == 0
    # A final episode at maximum age is alive; no terminal hazard is multiplied.
    torch.testing.assert_close(batch[1].exp(), reference(m, xs[1].double(), zs[1]))


@pytest.mark.parametrize(
    "bad",
    [torch.tensor([0.0, 1.0]), torch.tensor([0]), torch.tensor([0, 2]), torch.tensor([0, -1])],
)
def test_invalid_labels_rejected(bad):
    with pytest.raises(ValueError):
        model().supervised_log_likelihood(torch.zeros(2, 1), bad)


def test_ambiguous_self_resets_rejected():
    m = NHSMM(ModelConfig(n_states=2, n_features=1, causal=True, max_duration=3))
    m.initialize_distributions()
    with pytest.raises(ValueError, match="self_transitions"):
        m.supervised_log_likelihood(torch.zeros(2, 1), torch.tensor([0, 0]))


def test_artifact_roundtrip_preserves_structural_exclusion():
    m = model()
    payload = build_artifact(m)
    restored = load_inference_model(
        ModelConfig.from_dict(payload["model_config"]), payload["state_dict"], device="cpu"
    )
    assert not restored.config.allow_self_transitions
    assert torch.isneginf(
        restored.dist.transition.log_matrix()[0, 0].diagonal(dim1=0, dim2=2)
    ).all()
    x = torch.randn(4, 1, dtype=torch.float64)
    torch.testing.assert_close(m.log_likelihood(x), restored.log_likelihood(x))


def test_supervised_learning_improves_joint_score_and_keeps_labels_fixed():
    m = model(1)
    x = torch.tensor([[-3.0], [-2.8], [3.0], [3.2]], dtype=torch.float64)
    z = torch.tensor([0, 0, 1, 1])
    summary = m.fit_supervised(x, z, steps=60, lr=0.08)
    assert summary["updates"] == 60
    assert summary["final_nll_per_row"] < summary["initial_nll_per_row"] - 1
    assert torch.equal(m.predict(x, verbose=False)[0], z)


def test_neural_context_joint_matches_marginal_sum():
    m = model(2, context=True)
    x = torch.tensor([[-0.3], [0.4], [0.1]], dtype=torch.float64)
    c = torch.tensor([[1.0], [-1.0], [0.5]], dtype=torch.float64)
    scores = [
        m.supervised_log_likelihood(x, torch.tensor(z), context=c)[0]
        for z in itertools.product(range(2), repeat=3)
    ]
    torch.testing.assert_close(
        torch.logsumexp(torch.stack(scores), 0),
        m.log_likelihood(x, context=c)[0],
        atol=1e-12,
        rtol=1e-12,
    )


def test_noncausal_and_impossible_training_rejected_before_update():
    m = model(2)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    with pytest.raises(ValueError, match="zero or non-finite"):
        m.fit_supervised(torch.zeros(3, 1), torch.tensor([0, 0, 0]))
    for k, v in m.state_dict().items():
        assert torch.equal(v, before[k])
    m.config.causal = False
    with pytest.raises(ValueError, match="causal"):
        m.supervised_log_likelihood(torch.zeros(1, 1), torch.tensor([0]))
