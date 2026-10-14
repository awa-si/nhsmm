"""Independent regressions discovered by the complete-code audit."""

import pytest
import torch
from torch import nn

from nhsmm import ModelConfig, NHSMM, HSMMFilterRuntime
from nhsmm.artifact import build_artifact
from nhsmm.context import ContextEncoder, SequenceSet


def model(**kwargs):
    cfg = ModelConfig(
        n_states=2,
        n_features=1,
        max_duration=2,
        causal=True,
        use_context_encoder=False,
        dropout=0.0,
        seed=82,
        **kwargs,
    )
    m = NHSMM(cfg, device="cpu")
    m.initialize_distributions()
    m.eval()
    return m


def test_artifact_is_independent_snapshot_of_cpu_parameters():
    m = model()
    artifact = build_artifact(m)
    name = "dist.emission.mu"
    before = artifact["state_dict"][name].clone()
    with torch.no_grad():
        m.dist.emission.mu.add_(5.0)
    torch.testing.assert_close(artifact["state_dict"][name], before)


class IdentityEncoder(nn.Module):
    causal = False

    def forward(self, x, mask=None):
        return x


def test_multihead_pooling_ignores_right_padding():
    torch.manual_seed(83)
    enc = ContextEncoder(IdentityEncoder(), pool="mha", n_heads=1, layer_norm=False)
    enc.eval()
    short = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]])
    _, a, _ = enc(short, return_context=True)
    padded = torch.cat([short, torch.zeros(1, 4, 2)], dim=1)
    mask = torch.tensor([[True, True, False, False, False, False]])
    _, b, _ = enc(padded, mask=mask, return_context=True)
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)


def test_multihead_pooling_empty_rows_are_finite_zero():
    enc = ContextEncoder(IdentityEncoder(), pool="mha", n_heads=1, layer_norm=False)
    _, canonical, _ = enc(
        torch.zeros(2, 3, 2), mask=torch.zeros(2, 3, dtype=torch.bool), return_context=True
    )
    assert torch.isfinite(canonical).all()
    torch.testing.assert_close(canonical, torch.zeros_like(canonical))


def test_failed_runtime_initialization_does_not_lock_context_mode():
    m = model(context_dim=1)
    runtime = HSMMFilterRuntime(m)
    with pytest.raises(ValueError):
        runtime.step(torch.zeros(1), context=torch.zeros(2, 2, 1))
    assert runtime.state is None
    state = runtime.step(torch.zeros(1))
    assert torch.isfinite(state.state_posterior).all()


def test_causal_d1_tail_viterbi_competes_continuation_with_boundary():
    m = NHSMM(
        ModelConfig(
            n_states=2,
            n_features=1,
            max_duration=1,
            causal=True,
            duration_tail=True,
            duration_tail_init_probability=0.01,
            use_context_encoder=False,
            seed=91,
        ),
        device="cpu",
    )
    m.initialize_distributions()
    with torch.no_grad():
        m.dist.initial.logits.copy_(torch.tensor([0.0, -20.0]))
        m.dist.transition.logits.copy_(torch.tensor([[[0.01, 0.99]], [[0.99, 0.01]]]).log())
    seq = SequenceSet(
        sequences=torch.zeros(1, 2, 1),
        lengths=torch.tensor([2]),
        masks=torch.ones(1, 2, 1, dtype=torch.bool),
        contexts=torch.empty(1, 2, 0),
        canonical=torch.empty(1, 1, 0),
        log_probs=torch.tensor([[[0.0, -20.0], [0.0, torch.log(torch.tensor(2.0))]]]),
    )
    # From initial state 0, continuing has weight .99; changing has weight
    # .01*.99*2=.0198. Independent two-row path arithmetic favors staying.
    assert 0.99 > 0.01 * 0.99 * 2
    assert m._viterbi(seq)[0].tolist() == [0, 0]


def test_noncausal_tail_at_finite_boundary_has_finite_gradients():
    m = NHSMM(
        ModelConfig(
            n_states=2,
            n_features=1,
            max_duration=2,
            causal=False,
            duration_tail=True,
            duration_tail_init_probability=1.0,
            use_context_encoder=False,
            seed=97,
        ),
        device="cpu",
    )
    m.initialize_distributions()
    loss = -m.log_likelihood(torch.tensor([[[0.0], [0.2], [0.1]]]), reduce=True)
    loss.backward()
    assert torch.isfinite(m.dist.duration.tail_end_probability.grad).all()


def test_categorical_zero_mass_has_impossible_log_probability_and_finite_entropy():
    from nhsmm.distributions.default import Categorical

    dist = Categorical(probs=torch.tensor([1.0, 0.0]))
    assert torch.isneginf(dist.log_prob(torch.tensor(1)))
    assert dist.entropy().item() == 0.0


def test_stream_encoder_cache_tracks_replacement_parameter_identity():
    from nhsmm.encoder import DefaultEncoder

    torch.manual_seed(99)
    enc = DefaultEncoder(n_features=1, hidden_dim=2, cnn_channels=2, causal=True, dropout=0.0)
    enc.eval()
    with torch.no_grad():
        enc.conv.weight.fill_(0.3)
        enc.conv.bias.copy_(torch.tensor([0.1, -0.2]))
    x = torch.tensor([[[0.2]]])
    with torch.no_grad():
        for _ in range(2):
            value = enc.lstm.weight_ih_l0.detach().clone()
            value[:, 0] += 0.7
            enc.lstm.weight_ih_l0 = nn.Parameter(value)
            streamed, _ = enc.stream_step(x)
        batched = enc(x, return_sequence=True)
    torch.testing.assert_close(streamed, batched, atol=1e-6, rtol=1e-6)


def test_convergence_stop_false_runs_full_iteration_budget():
    import types

    m = model()
    calls = []

    def constant_loss(self, X, **kwargs):
        calls.append(kwargs["it"])
        zero = self.duration_logits_bias.sum() * 0.0
        return zero + 1.0, zero

    m._compute_loss = types.MethodType(constant_loss, m)
    cfg = m.config.with_overrides(
        n_init=1, max_iter=8, convergence_stop=False, plateau_window=2, use_scheduler=False
    )
    m.optimize(torch.zeros(1, 4, 1), cfg=cfg)
    assert calls[:-1] == [j for i in range(8) for j in (i, 0)]
    assert len(calls) == 17


def test_optimize_rejects_duration_tail_structure_change():
    m = model()
    cfg = m.config.with_overrides(duration_tail=True, n_init=1, max_iter=1)
    with pytest.raises(ValueError, match="duration_tail"):
        m.optimize(torch.zeros(1, 4, 1), cfg=cfg)


def test_grid_trial_limit_bounds_cartesian_consumption(monkeypatch):
    import itertools
    import nhsmm.tuning as tuning

    original = itertools.product
    consumed = []

    def counting_product(*args):
        for value in original(*args):
            consumed.append(value)
            yield value

    monkeypatch.setattr(tuning.itertools, "product", counting_product)
    tuner = tuning.ConfigTuner(
        ModelConfig(n_states=2, n_features=1),
        {"max_iter": tuning.Choice([1, 2, 3]), "lr": tuning.Choice([0.1, 0.2, 0.3])},
    )
    assert len(tuner.candidates(trials=1)) == 1
    assert len(consumed) == 1
