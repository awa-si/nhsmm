import copy

import pytest
import torch

from nhsmm import ModelConfig, NHSMM


def problem(mode):
    torch.set_num_threads(1)
    torch.manual_seed(401919)
    cfg = ModelConfig(
        n_states=2,
        n_features=2,
        max_duration=3,
        causal=True,
        use_context_encoder=mode == "encoder",
        context_dim=1 if mode == "external" else None,
        dropout=0.0,
        n_init=1,
        max_iter=2,
        use_scheduler=False,
        convergence_stop=False,
        verbose=False,
        optimizer_temperature_mode="constant",
        optimizer_initialization="current",
        loss_bias=0.2,
    )
    model = NHSMM(cfg, device="cpu")
    model.initialize_distributions()
    if mode == "plain":
        model.double()
    with torch.no_grad():
        model.duration_logits_bias.copy_(torch.tensor([[0.2, 0.7, 1.1], [0.3, 0.5, 1.2]]))
    x = [torch.randn(7, 2), torch.randn(4, 2), torch.randn(6, 2)]
    ctx = [torch.randn(len(item), 1) for item in x] if mode == "external" else None
    return model, x, ctx


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_chunk_size(value):
    with pytest.raises(ValueError, match="optimizer_batch_size"):
        ModelConfig(n_states=2, n_features=1, optimizer_batch_size=value)


@pytest.mark.parametrize("mode", ["plain", "external", "encoder"])
def test_accumulated_loss_gradients_and_target_match_full_batch(mode):
    full, x, ctx = problem(mode)
    chunked = copy.deepcopy(full)
    options = dict(loss_bias=0.2, it=0, max_iter=2, t_min=1.0, t_max=1.0)
    ll, loss = full._compute_loss(x, context=ctx, **options)
    loss.backward()
    aligned = chunked._align_external_context(x, ctx)
    chunk_ll, chunk_loss = chunked._backward_training_loss(x, aligned, 2, **options)
    atol = 1e-10 if mode == "plain" else 2e-5
    torch.testing.assert_close(chunk_ll, ll.detach(), atol=atol, rtol=2e-6)
    torch.testing.assert_close(chunk_loss, loss.detach(), atol=atol, rtol=2e-6)
    for (name, a), (_, b) in zip(full.named_parameters(), chunked.named_parameters(), strict=True):
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=atol, rtol=2e-5, msg=name)
    target = full._target_training_score(x, full._align_external_context(x, ctx), 1.0)
    actual = chunked._target_training_score(x, aligned, 1.0, 2)
    torch.testing.assert_close(target, actual, atol=atol, rtol=2e-6)


@pytest.mark.parametrize("mode", ["plain", "external", "encoder"])
def test_actual_accumulated_adam_matches_full_updates(mode):
    full, x, ctx = problem(mode)
    chunked = copy.deepcopy(full)
    full.optimize(x, context=ctx)
    chunked.optimize(x, context=ctx, cfg=chunked.config.with_overrides(optimizer_batch_size=2))
    for (name, a), (_, b) in zip(full.named_parameters(), chunked.named_parameters(), strict=True):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-4, msg=name)
    assert len(chunked._optimization_history[0]) == 2
    assert chunked._restart_scores == pytest.approx(full._restart_scores, abs=1e-4)


def test_empty_sequence_chunk_and_uneven_tensor_chunks():
    full, x, _ = problem("plain")
    x.insert(1, torch.empty(0, 2))
    chunked = copy.deepcopy(full)
    full.optimize(x)
    chunked.optimize(x, cfg=chunked.config.with_overrides(optimizer_batch_size=1))
    assert chunked._restart_scores == pytest.approx(full._restart_scores, abs=1e-8)
    tensor = torch.randn(5, 4, 2)
    parts = list(chunked._optimization_chunks(tensor, None, 2))
    assert [len(part) for part, _ in parts] == [2, 2, 1]
    assert torch.equal(torch.cat([part for part, _ in parts]), tensor)
