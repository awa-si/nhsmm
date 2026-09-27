from __future__ import annotations

import torch

from nhsmm import ModelConfig, NHSMM


def test_optimize_updates_causal_encoder_parameters() -> None:
    torch.manual_seed(17)
    cfg = ModelConfig(
        n_states=2,
        n_features=3,
        max_duration=6,
        causal=True,
        dropout=0.0,
        seed=17,
        n_init=1,
        max_iter=1,
        use_scheduler=False,
        convergence_stop=False,
        verbose=False,
    )
    model = NHSMM(cfg, device="cpu")
    model.initialize_distributions()

    trainable = [p for p in model.parameters() if p.requires_grad]
    encoder = [p for p in model.encoder.parameters() if p.requires_grad]
    assert encoder
    assert {id(p) for p in encoder}.issubset({id(p) for p in trainable})

    before = [p.detach().clone() for p in encoder]
    x = torch.randn(3, 24, cfg.n_features)
    model.optimize(x)

    assert model._optimizer is not None
    optimizer_ids = {
        id(p)
        for group in model._optimizer.param_groups
        for p in group["params"]
    }
    assert {id(p) for p in encoder}.issubset(optimizer_ids)
    assert any(not torch.equal(a, b.detach()) for a, b in zip(before, encoder))


def test_training_default_has_nontrivial_iteration_budget() -> None:
    cfg = ModelConfig(n_states=2, n_features=3)
    assert cfg.max_iter == 40
