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


def test_run_initialization_does_not_warm_start_encoder_or_duration_bias() -> None:
    torch.manual_seed(23)
    cfg = ModelConfig(
        n_states=2,
        n_features=2,
        max_duration=5,
        causal=True,
        dropout=0.0,
        seed=23,
        n_init=2,
        max_iter=1,
        use_scheduler=False,
        convergence_stop=False,
        verbose=False,
    )
    model = NHSMM(cfg, device="cpu")
    model.initialize_distributions()

    encoder_state = model._clone_state_dict(model.encoder)
    previous_dist_id = id(model.dist)
    with torch.no_grad():
        next(model.encoder.parameters()).add_(1.0)
        model.duration_logits_bias.zero_()

    model._initialize_run_state(
        1,
        observations=torch.randn(64, cfg.n_features),
        encoder_state=encoder_state,
        emission_init_mode="spread",
    )

    assert id(model.dist) != previous_dist_id
    current_encoder = model.encoder.state_dict()
    assert current_encoder.keys() == encoder_state.keys()
    assert all(torch.equal(current_encoder[key], encoder_state[key]) for key in encoder_state)
    assert torch.equal(model.duration_logits_bias, torch.ones_like(model.duration_logits_bias))


def test_best_parameter_snapshot_is_independent_and_restores_duration_bias() -> None:
    torch.manual_seed(29)
    cfg = ModelConfig(n_states=2, n_features=2, max_duration=4, causal=True, seed=29)
    model = NHSMM(cfg, device="cpu")
    model.initialize_distributions()

    with torch.no_grad():
        model.duration_logits_bias.copy_(torch.arange(8, dtype=torch.float32).reshape(2, 4))
    expected_bias = model.duration_logits_bias.detach().clone()
    expected_dist = model._clone_state_dict(model.dist)
    model._snapshot_best_params()

    with torch.no_grad():
        model.duration_logits_bias.fill_(-7.0)
        for parameter in model.dist.parameters():
            parameter.add_(3.0)

    model._restore_best_params()

    assert torch.equal(model.duration_logits_bias, expected_bias)
    restored = model.dist.state_dict()
    assert restored.keys() == expected_dist.keys()
    assert all(torch.equal(restored[key], expected_dist[key]) for key in expected_dist)


def test_kmeans_emission_initialization_recovers_separated_centers() -> None:
    torch.manual_seed(31)
    cfg = ModelConfig(
        n_states=3,
        n_features=2,
        max_duration=5,
        causal=True,
        seed=31,
        emission_init_mode="kmeans",
    )
    model = NHSMM(cfg, device="cpu")
    model.initialize_distributions()

    truth = torch.tensor([[-4.0, 0.0], [0.0, 4.0], [4.0, 0.0]])
    observations = torch.cat(
        [center + 0.05 * torch.randn(40, 2) for center in truth], dim=0
    )
    model._initialize_run_state(
        0,
        observations=observations,
        encoder_state=model._clone_state_dict(model.encoder),
        emission_init_mode="kmeans",
    )

    learned = model.dist.emission.base.detach()
    distances = torch.cdist(truth, learned)
    assert torch.all(distances.min(dim=1).values < 0.25)


def test_optimize_runs_multiple_restarts_without_warm_start_errors() -> None:
    torch.manual_seed(37)
    cfg = ModelConfig(
        n_states=2,
        n_features=2,
        max_duration=5,
        causal=True,
        dropout=0.0,
        seed=37,
        n_init=2,
        max_iter=1,
        use_scheduler=False,
        convergence_stop=False,
        verbose=False,
    )
    model = NHSMM(cfg, device="cpu")
    model.initialize_distributions()
    x = torch.randn(3, 20, cfg.n_features)

    model.optimize(x)

    assert set(model._best_state) >= {"dist", "encoder", "duration_logits_bias"}
    assert torch.isfinite(model.log_likelihood(x, reduce=True))


def test_scalar_external_context_uses_non_degenerate_distribution_hidden_space() -> None:
    torch.manual_seed(31)
    cfg = ModelConfig(
        n_states=3,
        n_features=3,
        max_duration=1,
        causal=True,
        context_dim=1,
        seed=31,
        n_init=1,
        max_iter=1,
        use_scheduler=False,
        convergence_stop=False,
        verbose=False,
        dropout=0.0,
    )
    model = NHSMM(cfg, device="cpu")
    model.initialize_distributions()

    first = model.dist.transition.context_net[0]
    assert first.out_features >= 16

    x = torch.randn(2, 12, cfg.n_features)
    context = torch.randint(0, 2, (2, 12, 1), dtype=torch.float32)
    model.optimize(x, context=context)

    probs = model.dist.transition.expected_probs(
        context=torch.tensor([[[0.0]], [[1.0]]])
    )
    assert torch.isfinite(probs).all()
    assert probs.shape[-3:] == (cfg.n_states, cfg.max_duration, cfg.n_states)


def test_transition_refinement_only_updates_context_modulation() -> None:
    torch.manual_seed(37)
    cfg = ModelConfig(
        n_states=3,
        n_features=3,
        max_duration=1,
        causal=True,
        context_dim=1,
        seed=37,
        n_init=1,
        max_iter=2,
        use_scheduler=False,
        convergence_stop=False,
        verbose=False,
        dropout=0.0,
        transition_context_max_delta=1.5,
    )
    model = NHSMM(cfg, device="cpu")
    model.initialize_distributions()
    x = torch.randn(3, 20, cfg.n_features)
    context = torch.randint(0, 2, (3, 20, 1), dtype=torch.float32)
    model.optimize(x, context=context)

    transition = model.dist.transition
    refined_ids = {id(p) for p in transition.context_net.parameters()} | {
        id(transition.delta_scale)
    }
    frozen_before = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if id(p) not in refined_ids
    }
    refined_before = [
        p.detach().clone()
        for p in [*transition.context_net.parameters(), transition.delta_scale]
    ]

    ll = model._refine_transition_likelihood(
        x,
        context,
        steps=3,
        lr=1e-2,
    )

    assert torch.isfinite(torch.tensor(ll))
    assert all(
        torch.equal(frozen_before[name], p.detach())
        for name, p in model.named_parameters()
        if id(p) not in refined_ids
    )
    assert any(
        not torch.equal(before, after.detach())
        for before, after in zip(
            refined_before,
            [*transition.context_net.parameters(), transition.delta_scale],
        )
    )

def test_transition_refinement_is_opt_in() -> None:
    cfg = ModelConfig(n_states=2, n_features=3)
    assert cfg.transition_refine_steps == 0
    assert cfg.transition_refine_lr == 3e-2
    assert cfg.transition_context_max_delta == 0.5


def test_transition_context_capacity_is_transition_specific() -> None:
    cfg = ModelConfig(
        n_states=3,
        n_features=3,
        context_dim=2,
        hidden_dim=2,
        transition_context_max_delta=1.5,
    )
    model = NHSMM(cfg, device="cpu")
    model.initialize_distributions()

    assert model.dist.transition.max_delta == 1.5
    assert model.dist.initial.max_delta == 0.5
    assert model.dist.duration.max_delta == 0.5
    assert model.dist.emission.max_delta == 0.5
