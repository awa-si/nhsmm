"""Numerical training failures must not become successful updates/restarts."""

import pytest
import torch
from nhsmm import NHSMM, ModelConfig

X = torch.tensor([[-1.0], [0.0], [1.0], [0.5]])
CTX = torch.tensor([[[0.0, 1.0], [1.0, 0.0], [0.2, 0.8], [0.8, 0.2]]])


def make(**changes):
    cfg = ModelConfig(
        2,
        1,
        max_duration=2,
        allow_self_transitions=False,
        causal=True,
        n_init=1,
        max_iter=1,
        dropout=0.0,
        verbose=False,
        use_scheduler=False,
        convergence_stop=False,
        optimizer_initialization="current",
        seed=71,
    ).with_overrides(**changes)
    model = NHSMM(cfg, device="cpu")
    model.initialize_distributions()
    model.eval()
    return model


def track_steps(monkeypatch):
    calls = []
    original = torch.optim.Adam.step

    def step(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", step)
    return calls


def snapshot(model):
    return {k: v.clone() for k, v in model.state_dict().items()}


def unchanged(model, before):
    assert not model.training
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_latent_bad_gradient_never_steps(monkeypatch, value):
    model = make()
    before = snapshot(model)
    calls = track_steps(monkeypatch)
    model.dist.emission.mu.register_hook(lambda g: torch.full_like(g, value))
    with pytest.raises(RuntimeError, match="non-finite"):
        model.optimize(X)
    assert not calls
    unchanged(model, before)
    with pytest.raises(RuntimeError, match="No completed"):
        model.optimization_summary()


def test_finite_elements_with_overflowing_total_norm_never_step(monkeypatch):
    model = make()
    before = snapshot(model)
    calls = track_steps(monkeypatch)
    model.dist.emission.mu.register_hook(lambda g: torch.full_like(g, 3e38))
    with pytest.raises(RuntimeError, match="non-finite"):
        model.optimize(X)
    assert not calls
    unchanged(model, before)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("which", ["score", "loss"])
def test_invalid_chunk_objective_never_steps(monkeypatch, which, bad):
    model = make()
    calls = track_steps(monkeypatch)
    before = snapshot(model)
    p = model.dist.emission.mu

    def invalid(*args, **kwargs):
        finite = p.sum() * 0
        invalid = finite + bad
        return (invalid, finite) if which == "score" else (finite, invalid)

    monkeypatch.setattr(model, "_compute_loss", invalid)
    with pytest.raises(FloatingPointError, match="non-finite"):
        model.optimize(X)
    assert not calls
    unchanged(model, before)


def test_accumulated_finite_scores_cannot_overflow_silently(monkeypatch):
    model = make(optimizer_batch_size=1)
    calls = track_steps(monkeypatch)
    p = model.dist.emission.mu
    monkeypatch.setattr(model, "_compute_loss", lambda *a, **k: (p.sum() * 0 + 3e38, p.sum() * 0))
    with pytest.raises(FloatingPointError, match="non-finite"):
        model.optimize(X.unsqueeze(0).repeat(2, 1, 1))
    assert not calls


@pytest.mark.parametrize("phase", ["monitor", "final"])
def test_bad_target_score_cannot_publish_success(monkeypatch, phase):
    model = make()
    calls = track_steps(monkeypatch)
    count = []

    def score(*args, **kwargs):
        count.append(1)
        return torch.tensor(float("nan") if phase == "monitor" or len(count) > 1 else 0.0)

    monkeypatch.setattr(model, "_target_training_score", score)
    with pytest.raises(FloatingPointError, match="non-finite"):
        model.optimize(X)
    assert len(calls) == (0 if phase == "monitor" else 1)
    with pytest.raises(RuntimeError, match="No completed"):
        model.optimization_summary()


@pytest.mark.parametrize("failure", ["loss", "gradient", "norm", "final"])
def test_refinement_failure_restores_mode_and_trainability(monkeypatch, failure):
    model = make(context_dim=2)
    flags = {name: p.requires_grad for name, p in model.named_parameters()}
    before = snapshot(model)
    calls = track_steps(monkeypatch)
    transition = model.dist.transition
    if failure in ("gradient", "norm"):
        magnitude = float("nan") if failure == "gradient" else 3e38
        transition.delta_scale.register_hook(lambda g: torch.full_like(g, magnitude))
    else:
        original = model._compute_loss
        count = []

        def invalid(*args, **kwargs):
            count.append(1)
            if failure == "loss" or len(count) > 1:
                zero = transition.delta_scale.sum() * 0
                return zero + float("nan"), zero + float("nan")
            return original(*args, **kwargs)

        monkeypatch.setattr(model, "_compute_loss", invalid)
    error = RuntimeError if failure in ("gradient", "norm") else FloatingPointError
    with pytest.raises(error, match="non-finite"):
        model._refine_transition_likelihood(X, CTX, steps=1, lr=0.01)
    assert len(calls) == (1 if failure == "final" else 0)
    assert {name: p.requires_grad for name, p in model.named_parameters()} == flags
    assert not model.training
    if failure != "final":
        unchanged(model, before)


def test_supervised_clipping_rejects_overflowing_norm(monkeypatch):
    model = make(supervised_grad_clip_norm=1.0)
    before = snapshot(model)
    calls = track_steps(monkeypatch)
    model.dist.emission.mu.register_hook(lambda g: torch.full_like(g, 3e38))
    with pytest.raises(RuntimeError, match="non-finite"):
        model.fit_supervised(X, torch.tensor([0, 0, 1, 1]), steps=1)
    assert not calls
    unchanged(model, before)
