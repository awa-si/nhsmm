"""Independent synthetic oracle for contextual D+ hazard."""

import itertools

import pytest
import torch
from nhsmm import NHSMM, CausalContext, HSMMFilterRuntime, ModelConfig
from nhsmm.filtering import filter_model_sequence


def _model(*, dynamic=True):
    model = NHSMM(
        ModelConfig.for_mode(
            "nhsmm",
            n_states=2,
            n_features=1,
            context_dim=1,
            max_duration=2,
            duration_tail=True,
            duration_tail_context=dynamic,
            dropout=0,
            seed=17,
        ),
        device="cpu",
    )
    model.initialize_distributions()
    return model.double().eval()


def _inputs():
    x = torch.tensor([-0.3, 0.2, 0.4, -0.1, 0.15], dtype=torch.float64)[:, None]
    u = torch.tensor([-0.9, -0.5, 1.2, 1.1, -1.0], dtype=torch.float64)[:, None]
    return x, CausalContext(torch.cat((u.new_zeros(1, 1), u[:-1])), u)


def test_contextual_tail_exact_path_oracle_and_stream():
    model = _model()
    x, context = _inputs()
    with torch.no_grad():
        model.dist.duration.tail_end_probability.copy_(
            torch.tensor([0.04, 0.06], dtype=torch.float64)
        )
        model.dist.duration.tail_context_head.weight.copy_(
            torch.tensor([[3.0], [2.0]], dtype=torch.float64)
        )
    h = model.dist.duration.tail_probability(context=context.boundary[None])[0]
    assert h[2, 0] > 0.5 and h[0, 0] < 0.01
    q = model.dist.duration.log_matrix(
        context=context.boundary[None], T=len(x), soft_dmax=model.duration_logits_bias
    )[0].exp()
    a = model.dist.transition.log_matrix(context=context.boundary[None], T=len(x))[0].exp()
    pi = model.dist.initial.log_matrix(context=context.emission[:1][None])[0, 0].exp()
    b = model.dist.emission.log_prob(x[None], context=context.emission[None])[0].exp()
    total = x.new_zeros(())
    frontier = x.new_zeros(2, 2)
    for z in itertools.product(range(2), repeat=len(x)):
        score = pi[z[0]] * b[0, z[0]]
        age = 1
        for t in range(1, len(x)):
            i, j = z[t - 1], z[t]
            hazard = h[t - 1, i] if age >= 2 else q[t - 1, i, 0]
            ended = i != j
            score = score * (hazard * a[t - 1, i, min(age, 2) - 1, j] if ended else 1 - hazard)
            score = score * b[t, j]
            age = 1 if ended else age + 1
        torch.testing.assert_close(
            model.supervised_log_likelihood(x, torch.tensor(z), context=context)[0].exp(),
            score,
        )
        total += score
        frontier[z[-1], min(age, 2) - 1] += score
    torch.testing.assert_close(model.log_likelihood(x, context=context)[0].exp(), total)
    trace = filter_model_sequence(model, x, context=context)
    torch.testing.assert_close(trace.log_posterior[0, -1].exp(), frontier / total)
    runtime = HSMMFilterRuntime(model)
    for t in range(len(x)):
        state = runtime.step(
            x[t],
            context=CausalContext(context.emission[t], context.boundary[t]),
        )
        torch.testing.assert_close(state.log_posterior[0], trace.log_posterior[0, t])
        torch.testing.assert_close(runtime.state.tail_end_probability[0], h[t])
    model.zero_grad()
    model.log_likelihood(x, context=context).sum().backward()
    gradient = model.dist.duration.tail_context_head.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_contextual_tail_mask_and_exact_compatibility():
    model = _model()
    x, context = _inputs()
    head = model.dist.duration.tail_context_head
    with torch.no_grad():
        head.weight.fill_(5)
        model.dist.duration.tail_end_probability.fill_(1)
    h = model.dist.duration.tail_probability(context=context.boundary[None])
    torch.testing.assert_close(h, torch.ones_like(h))
    with torch.no_grad():
        model.dist.duration.tail_end_probability.fill_(0.2)
    mask = torch.tensor([[True, False, True, True, True]])
    bad = context.boundary[None].clone()
    bad[0, 1] = float("nan")
    result = model.dist.duration.tail_probability(context=bad, context_mask=mask)
    torch.testing.assert_close(result[0, 1], model.dist.duration.tail_probability())
    assert torch.isfinite(result).all()
    result.sum().backward()
    assert torch.isfinite(head.weight.grad).all()
    assert head.weight.grad.abs().sum() > 0


def test_contextual_tail_is_explicit_and_causal():
    with pytest.raises(ValueError, match="requires duration_tail"):
        ModelConfig.for_mode(
            "nhsmm",
            n_states=2,
            n_features=1,
            context_dim=1,
            duration_tail_context=True,
        )
    with pytest.raises(ValueError, match="requires causal"):
        ModelConfig(
            n_states=2,
            n_features=1,
            context_dim=1,
            duration_tail=True,
            duration_tail_context=True,
            causal=False,
        )
    legacy = _model(dynamic=False)
    assert legacy.dist.duration.tail_context_head is None
    assert not any("tail_context_head" in name for name in legacy.state_dict())


def test_context_only_supervision_updates_tail_head_without_baseline():
    model = _model()
    x, context = _inputs()
    labels = torch.tensor([0, 0, 0, 1, 1])
    head_before = model.dist.duration.tail_context_head.weight.detach().clone()
    base_before = model.dist.duration.tail_end_probability.detach().clone()
    summary = model.fit_supervised(x, labels, context=context, steps=2, lr=0.01, context_only=True)
    assert summary["final_nll_per_row"] < summary["initial_nll_per_row"]
    assert not torch.equal(head_before, model.dist.duration.tail_context_head.weight)
    torch.testing.assert_close(base_before, model.dist.duration.tail_end_probability)
