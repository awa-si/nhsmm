"""Independent probability-product oracle for prefix-state supervision."""

import pytest
import torch

from .neural_parameter_reference import enumerate_weights
from .test_neural_parameter_math import fixture, parts


def reference(model, x, context, labels):
    ingredients = parts(model, x, context)
    terms = []
    for n in range(1, len(x) + 1):
        rows = enumerate_weights(ingredients, hmm=True, length=n)
        total = torch.stack([w for _, w, _ in rows]).sum()
        selected = torch.stack([w for path, w, _ in rows if path[-1] == int(labels[n - 1])]).sum()
        terms.append((selected / total).log())
    return torch.stack(terms).sum()


@pytest.mark.parametrize("available", [False, True])
def test_every_prefix_score_and_all_parameter_context_gradients(available):
    model, x, c = fixture(1, False, available)
    z = torch.tensor([0, 1, 2, 0])
    actual = model.supervised_filter_log_score(x, z, context=c).sum()
    expected = reference(model, x, c, z)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
    parameters = [p for p in model.parameters() if p.requires_grad] + [c.emission, c.boundary]
    left = torch.autograd.grad(actual, parameters, allow_unused=True, retain_graph=True)
    right = torch.autograd.grad(expected, parameters, allow_unused=True, retain_graph=True)
    for p, a, b in zip(parameters, left, right, strict=True):
        a = torch.zeros_like(p) if a is None else a
        b = torch.zeros_like(p) if b is None else b
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-9)
    assert torch.equal(left[-2][~c.available], torch.zeros_like(left[-2][~c.available]))
    assert torch.equal(left[-1][~c.available], torch.zeros_like(left[-1][~c.available]))
    assert torch.equal(left[-1][-1], torch.zeros_like(left[-1][-1]))
    from nhsmm import CausalContext

    for n in range(1, len(x) + 1):
        prefix = CausalContext(c.emission[:n], c.boundary[:n], c.available[:n])
        torch.testing.assert_close(
            model.supervised_filter_log_score(x[:n], z[:n], context=prefix).sum(),
            reference(model, x[:n], prefix, z[:n]),
            atol=1e-12,
            rtol=1e-10,
        )


def test_gradient_matches_central_differences():
    model, x, c = fixture(1, False)
    z = torch.tensor([0, 1, 2, 0])
    probes = [
        model.dist.initial.logits,
        model.dist.transition.logits,
        model.dist.emission.mu,
        model.dist.emission.context_net[-1].bias,
    ]

    def score():
        return model.supervised_filter_log_score(x, z, context=c).sum()

    gradients = torch.autograd.grad(score(), probes)
    for p, g in zip(probes, gradients, strict=True):
        original = float(p.detach().flatten()[0])
        values = []
        try:
            for sign in (1, -1):
                with torch.no_grad():
                    p.view(-1)[0] = original + sign * 2e-5
                values.append(float(score().detach()))
        finally:
            with torch.no_grad():
                p.view(-1)[0] = original
        assert float(g.flatten()[0]) == pytest.approx(
            (values[0] - values[1]) / 4e-5, abs=2e-5, rel=3e-4
        )


def test_previous_true_labels_never_drive_filter_history():
    model, x, _ = fixture(1, False)
    a = torch.tensor([0, 1, 2, 0])
    b = a.clone()
    b[0] = 2
    difference = model.supervised_filter_log_score(x, a) - model.supervised_filter_log_score(x, b)
    first = model.supervised_filter_log_score(x[:1], a[:1]) - model.supervised_filter_log_score(
        x[:1], b[:1]
    )
    torch.testing.assert_close(difference, first, atol=1e-12, rtol=1e-10)


def test_ragged_padding_and_empty_sequences_reset_independently():
    model, x, _ = fixture(1, False)
    xs = [x, x[:2], x[:0]]
    zs = [torch.tensor([0, 1, 2, 0]), torch.tensor([2, 1]), torch.empty(0, dtype=torch.long)]
    batch = model.supervised_filter_log_score(xs, zs)
    separate = torch.cat([model.supervised_filter_log_score(a, b) for a, b in zip(xs, zs)])
    torch.testing.assert_close(batch, separate, atol=1e-12, rtol=1e-10)
    assert batch[-1] == 0
    torch.testing.assert_close(model.supervised_filter_log_score(xs, zs, reduce=True), batch.sum())


@pytest.mark.parametrize(
    "duration,tail,causal", [(2, False, False), (1, True, False), (1, False, False)]
)
def test_unsupported_models_reject_before_optimizer(duration, tail, causal, monkeypatch):
    model, x, _ = fixture(duration, tail)
    model.config = model.config.with_overrides(causal=causal)

    def forbidden(*a, **kw):
        raise AssertionError("optimizer must not be constructed")

    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    with pytest.raises(ValueError, match="causal"):
        model.fit_supervised(
            x, torch.tensor([0, 1, 2, 0]), objective="causal_filter_log_score", steps=1
        )


@pytest.mark.parametrize(
    "bad", [torch.tensor([0, 1]), torch.tensor([0.0, 1.0, 2.0, 0.0]), torch.tensor([0, 1, 3, 0])]
)
def test_invalid_labels_reject_before_updates(bad, monkeypatch):
    model, x, _ = fixture(1, False)

    def forbidden(*a, **kw):
        raise AssertionError("optimizer must not be constructed")

    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    with pytest.raises(ValueError):
        model.fit_supervised(x, bad, objective="causal_filter_log_score", steps=1)


def test_zero_support_is_not_clipped_and_cannot_be_optimized(monkeypatch):
    model, x, _ = fixture(1, False)
    monkeypatch.setattr(
        model.dist.initial,
        "log_matrix",
        lambda **kw: x.new_tensor([[[0.0, -torch.inf, -torch.inf]]]),
    )
    labels = torch.tensor([1, 1, 2, 0])
    assert torch.isneginf(model.supervised_filter_log_score(x, labels)).all()
    with pytest.raises(ValueError, match="non-finite"):
        model.fit_supervised(x, labels, objective="causal_filter_log_score", steps=1)


def test_extreme_rare_events_keep_finite_unclipped_log_score_and_gradients():
    model, x, _ = fixture(1, False)
    x = x + 1000
    score = model.supervised_filter_log_score(x, torch.tensor([0, 1, 2, 0])).sum()
    assert torch.isfinite(score) and score < -1000
    g = torch.autograd.grad(score, model.dist.emission.mu)[0]
    assert torch.isfinite(g).all()


def test_new_fit_reports_objective_and_default_stays_generative():
    model, x, _ = fixture(1, False)
    z = torch.tensor([0, 1, 2, 0])
    mode = model.training
    result = model.fit_supervised(x, z, objective="causal_filter_log_score", steps=5, lr=0.01)
    assert result["objective"] == "causal_filter_log_score" and result["updates"] == 5
    assert result["final_nll_per_row"] < result["initial_nll_per_row"]
    assert model.training == mode
    default = model.fit_supervised(x, z, steps=1)
    assert default["objective"] == "observed_joint_causal_prefix"
    with pytest.raises(ValueError, match="objective"):
        model.fit_supervised(x, z, objective="unknown")
