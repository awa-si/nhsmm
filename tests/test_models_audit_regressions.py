"""Regression contracts for the seven 2026-10-15 models audit findings."""

import copy
from unittest.mock import patch

import pytest
import torch

from nhsmm import ModelConfig, NHSMM
from nhsmm.context import SequenceSet
from nhsmm.distributions import Duration, Transition
from nhsmm.models import DistributionSet


def model(**overrides):
    options = dict(
        n_states=2,
        n_features=2,
        max_duration=3,
        causal=True,
        n_init=1,
        max_iter=2,
        optimizer_initialization="current",
        optimizer_temperature_mode="constant",
        use_scheduler=False,
        convergence_stop=False,
        dropout=0.0,
        seed=1701,
        verbose=False,
    )
    options.update(overrides)
    result = NHSMM(ModelConfig(**options), device="cpu")
    result.initialize_distributions()
    result.eval()
    return result


def scores(log_probs):
    log_probs = torch.tensor(log_probs, dtype=torch.float64)
    t = len(log_probs)
    return SequenceSet(
        sequences=torch.zeros(1, t, 2, dtype=torch.float64),
        lengths=torch.tensor([t]),
        masks=torch.ones(1, t, 1, dtype=torch.bool),
        contexts=torch.zeros(1, t, 0, dtype=torch.float64),
        canonical=torch.zeros(1, 1, 0, dtype=torch.float64),
        log_probs=log_probs.unsqueeze(0),
    )


@pytest.mark.parametrize("chunk", [None, 1])
def test_empty_entries_are_neutral_for_objective_and_actual_updates(chunk):
    full = model(optimizer_batch_size=chunk)
    mixed = copy.deepcopy(full)
    x = [torch.randn(4, 2), torch.randn(5, 2)]
    padded = [x[0], torch.empty(0, 2), x[1], torch.empty(0, 2)]
    expected = full.log_likelihood(x)
    actual = mixed.log_likelihood(padded)
    torch.testing.assert_close(actual[[0, 2]], expected)
    torch.testing.assert_close(actual[[1, 3]], torch.zeros(2))
    assert torch.isfinite(mixed._target_training_score(padded, None, 1.0, chunk))
    full.optimize(x)
    mixed.optimize(padded)
    for name, value in full.state_dict().items():
        torch.testing.assert_close(mixed.state_dict()[name], value, atol=2e-6, rtol=2e-5)
    assert mixed._restart_scores == pytest.approx(full._restart_scores, abs=2e-5)


@pytest.mark.parametrize("kind", ["zero_batch", "zero_time", "empty_list", "empty_entries"])
@pytest.mark.parametrize("initialization", ["current", "restart"])
def test_empty_training_rejected_before_mutation(kind, initialization):
    m = model(optimizer_initialization=initialization)
    inputs = {
        "zero_batch": torch.empty(0, 4, 2),
        "zero_time": torch.empty(2, 0, 2),
        "empty_list": [],
        "empty_entries": [torch.empty(0, 2)],
    }
    before = copy.deepcopy(m.state_dict())
    with pytest.raises(ValueError, match="at least one.*observation|at least one.*sequence"):
        m.optimize(inputs[kind])
    assert not m.training
    for name, value in before.items():
        torch.testing.assert_close(m.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("initialization", ["current", "restart"])
def test_refinement_context_rejected_before_mutation(initialization):
    m = model(context_dim=1, transition_refine_steps=1, optimizer_initialization=initialization)
    before = copy.deepcopy(m.state_dict())
    with pytest.raises(ValueError, match="requires explicit external context"):
        m.optimize(torch.randn(1, 4, 2))
    assert not m.training
    for name, value in before.items():
        torch.testing.assert_close(m.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("initialize_after_conversion", [False, True])
def test_float64_internal_encoder_inputs_gradients_and_restart(initialize_after_conversion):
    m = model(use_context_encoder=True, max_iter=1, optimizer_initialization="restart")
    m.double()
    if initialize_after_conversion:
        m.initialize_distributions()
    x = torch.randn(1, 6, 2, dtype=torch.float64)
    sequence = m._build_sequence_set(x)
    assert sequence.sequences.dtype == sequence.contexts.dtype == torch.float64
    ll = m.log_likelihood(x, reduce=True)
    assert ll.dtype == torch.float64
    ll.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in m.encoder.parameters())
    m.optimize(x)
    assert all(p.dtype == torch.float64 for p in m.parameters())
    assert torch.isfinite(m.log_likelihood(x)).all()


@pytest.mark.parametrize("tail", [False, True])
def test_noncausal_float64_density_and_gradient_match_normal(tail):
    m = model(
        n_states=1,
        n_features=1,
        max_duration=1,
        causal=False,
        duration_tail=tail,
        duration_tail_init_probability=1.0,
    ).double()
    with torch.no_grad():
        m.dist.emission.mu.zero_()
        m.dist.emission.log_var.zero_()
    x = torch.tensor([[[1e20]]], dtype=torch.float64)
    mu = m.dist.emission.mu
    var = torch.nn.functional.softplus(m.dist.emission.log_var)
    expected = torch.distributions.Normal(mu, var.sqrt()).log_prob(x).sum()
    actual = m.log_likelihood(x, reduce=True)
    expected_grad = torch.autograd.grad(expected, mu)[0]
    actual_grad = torch.autograd.grad(actual, mu)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=0)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-12, atol=0)


@pytest.mark.parametrize("log_probs", [[[-torch.inf], [0.0]], [[0.0], [-torch.inf]]])
def test_zero_evidence_fails_before_normalization_and_public_wrapper(log_probs):
    m = model(n_states=1, max_duration=1)
    sequence = scores(log_probs)
    with pytest.raises(ValueError, match="evidence"):
        m.forward(sequence)
    with patch.object(m, "_build_sequence_set", return_value=sequence):
        with pytest.raises(ValueError, match="evidence"):
            m.log_likelihood(torch.zeros(1, 2, 2))


def test_public_likelihood_does_not_hide_nan_scores():
    m = model()
    with patch.object(m, "forward", return_value=torch.full((1, 2, 2, 3), torch.nan)):
        with pytest.raises(ValueError, match="NaN"):
            m.log_likelihood(torch.zeros(1, 2, 2))


class OnlyTwo(Duration):
    def _apply_constraints(self, logits, mask=None):
        return super()._apply_constraints(
            logits, mask=torch.tensor([False, True], device=logits.device)
        )


class StateOnlyTransition(Transition):
    def __init__(self, *args, **kwargs):
        kwargs["max_duration"] = None
        super().__init__(*args, **kwargs)


@pytest.mark.parametrize("tail", [False, True])
@pytest.mark.parametrize("duration_dependent", [False, True])
@pytest.mark.parametrize("causal", [False, True])
def test_viterbi_rejects_impossible_terminal_support(tail, duration_dependent, causal):
    m = model(max_duration=2, causal=causal, duration_tail=tail)
    options = dict(config=m.config, duration=OnlyTwo)
    if not duration_dependent:
        options["transition"] = StateOnlyTransition
    m.dist = DistributionSet(**options)
    m.dist.initialize()
    if causal:
        sequence = scores([[-torch.inf, -torch.inf]])
        with pytest.raises(ValueError, match="Viterbi path"):
            m._viterbi(sequence)
    else:
        x = torch.zeros(1, 1, 2)
        assert torch.isneginf(m.log_likelihood(x)).all()
        with pytest.raises(ValueError, match="Viterbi path"):
            m.decode(x, verbose=False)


@pytest.mark.parametrize("tail", [False, True])
@pytest.mark.parametrize("causal", [False, True])
def test_empty_inference_contracts_across_branches(tail, causal):
    m = model(causal=causal, duration_tail=tail)
    for x in [torch.empty(2, 0, 2), []]:
        count = len(x)
        sequence = m._build_sequence_set(x)
        assert m.forward(sequence).shape == (count, 0, 2, 3)
        ll = m.log_likelihood(x)
        assert ll.shape == (count,)
        torch.testing.assert_close(ll, torch.zeros(count))
        torch.testing.assert_close(m.predict(x, mode="log_likelihood", verbose=False), ll)
        assert all(len(p) == 0 for p in m.predict(x, verbose=False))
    with pytest.raises(ValueError, match="Unsupported decoding mode"):
        m.predict(torch.empty(1, 0, 2), mode="bogus", verbose=False)


def test_empty_internal_encoder_inference():
    m = model(use_context_encoder=True).double()
    x = torch.empty(2, 0, 2, dtype=torch.float64)
    torch.testing.assert_close(m.log_likelihood(x), torch.zeros(2, dtype=torch.float64))
    assert m.forward(m._build_sequence_set(x)).shape == (2, 0, 2, 3)


def test_empty_causal_forward_still_rejects_timestep():
    m = model()
    sequence = m._build_sequence_set(torch.empty(1, 0, 2))
    with pytest.raises(ValueError, match="timestep"):
        m.forward(sequence, timestep=0)
