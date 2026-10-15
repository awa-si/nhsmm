from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch

from nhsmm import ModelConfig, NHSMM, context_effect, evaluate_validation_snapshot
from nhsmm.validation.alignment import (
    StateAlignment,
    _hungarian,
    align_state_centers,
    finite_correlation,
)
from nhsmm.validation.context_evidence import ContextEvidenceConfig, evaluate_context_replication
from nhsmm.validation.split_fit import split_fit_context_evidence
from nhsmm.validation.state_effect import (
    evaluate_state_context_replication,
    split_fit_state_context_evidence,
)


def _centers(_model):
    return np.arange(3, dtype=float).reshape(3, 1)


def _matrix(_model, _context):
    return np.ones((3, 3))


POLICY = ContextEvidenceConfig(replication_corr_min=0.0, min_replica_amplitude=0.0)


@pytest.mark.parametrize("exclude_self", [True, False])
def test_negative_transition_mass_cannot_be_evidence(exclude_self):
    value = np.array([[1.0, -0.2, 0.8], [0.3, 0.2, 0.5], [0.2, 0.3, 0.5]])
    with pytest.raises(ValueError, match="non-negative"):
        evaluate_context_replication(
            object(),
            object(),
            [-1, 1],
            effect_matrix=lambda m, c: value,
            config=ContextEvidenceConfig(0.0, 0.0, exclude_self_transitions=exclude_self),
            centers=_centers,
        )


@pytest.mark.parametrize("state_effect", [False, True])
def test_constant_effect_is_not_selected_at_zero_thresholds(state_effect):
    evaluator = evaluate_state_context_replication if state_effect else evaluate_context_replication
    kwargs = {"effect_tensor" if state_effect else "effect_matrix": _matrix}
    result = evaluator(object(), object(), [-1, 1], config=POLICY, centers=_centers, **kwargs)
    assert not result.selected
    assert result.replication_corr == 0.0
    assert result.min_replica_amplitude == 0.0


@pytest.mark.parametrize("splitter", [split_fit_context_evidence, split_fit_state_context_evidence])
@pytest.mark.parametrize("split", [True, 2.9, "2"])
def test_split_index_must_be_an_integer_before_fitting(splitter, split):
    calls = []
    with pytest.raises(ValueError, match="integer"):
        splitter(
            np.zeros((6, 2)),
            np.zeros((6, 1)),
            fit_model=lambda *args: calls.append(args),
            evidence_contexts=[-1, 1],
            config=POLICY,
            centers=_centers,
            split_index=split,
            **(
                {"effect_matrix": _matrix}
                if splitter is split_fit_context_evidence
                else {"effect_tensor": _matrix}
            ),
        )
    assert calls == []


@pytest.mark.parametrize("splitter", [split_fit_context_evidence, split_fit_state_context_evidence])
def test_split_grid_is_validated_before_fitting(splitter):
    calls = []
    with pytest.raises(ValueError, match="two"):
        splitter(
            np.zeros((6, 2)),
            np.zeros((6, 1)),
            fit_model=lambda *args: calls.append(args),
            evidence_contexts=iter([0]),
            config=POLICY,
            centers=_centers,
            **(
                {"effect_matrix": _matrix}
                if splitter is split_fit_context_evidence
                else {"effect_tensor": _matrix}
            ),
        )
    assert calls == []


@pytest.mark.parametrize("splitter", [split_fit_context_evidence, split_fit_state_context_evidence])
def test_split_fit_rejects_reused_model_instance(splitter):
    same = object()
    with pytest.raises(ValueError, match="distinct"):
        splitter(
            np.zeros((6, 2)),
            np.zeros((6, 1)),
            fit_model=lambda *args: same,
            evidence_contexts=[-1, 1],
            config=POLICY,
            centers=_centers,
            **(
                {"effect_matrix": _matrix}
                if splitter is split_fit_context_evidence
                else {"effect_tensor": _matrix}
            ),
        )


def _model():
    model = NHSMM(
        ModelConfig(
            n_states=3,
            n_features=2,
            max_duration=4,
            context_dim=2,
            hidden_dim=2,
            seed=17,
            causal=True,
        ),
        device="cpu",
    )
    model.initialize_distributions()
    return model


@pytest.mark.parametrize("shape", [(2, 1, 3, 3), (1, 2, 3, 3), (2, 1, 3, 4, 3), (1, 2, 3, 4, 3)])
def test_transition_adapter_does_not_discard_batch_or_time(shape):
    model = _model()
    model.dist.transition.log_matrix = lambda **kw: torch.full(shape, -np.log(3.0))
    with pytest.raises(ValueError, match="B=T=1"):
        context_effect(model, [0.0, 0.0], component="transition")


@pytest.mark.parametrize("component", ["initial", "duration", "emission", "transition"])
@pytest.mark.parametrize("failing", [False, True])
def test_context_probe_preserves_mixed_submodule_modes(component, failing):
    model = _model()
    model.train()
    # A deliberate eval-only subtree inside a training model.
    model.dist.emission.eval()
    before = [(module, module.training) for module in model.modules()]
    target = getattr(model.dist, component)
    method = {
        "initial": "expected_probs",
        "duration": "log_matrix",
        "transition": "log_matrix",
        "emission": "forward",
    }[component]
    if failing:

        def explode(**kwargs):
            raise RuntimeError("probe failure")

        setattr(target, method, explode)
        with pytest.raises(RuntimeError, match="probe failure"):
            context_effect(model, [0.0, 0.0], component=component)
    else:
        context_effect(model, [0.0, 0.0], component=component)
    assert [module.training for module, _ in before] == [mode for _, mode in before]


@pytest.mark.parametrize("failing", [False, True])
def test_snapshot_preserves_mixed_modes(failing):
    model = _model()
    model.train()
    model.dist.emission.eval()
    before = [(module, module.training) for module in model.modules()]
    if failing:
        model.log_likelihood = lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("failure")
        )
        with pytest.raises(RuntimeError, match="failure"):
            evaluate_validation_snapshot(model, torch.ones(1, 4, 2))
    else:
        evaluate_validation_snapshot(model, torch.ones(1, 4, 2))
    assert [module.training for module, _ in before] == [mode for _, mode in before]


@pytest.mark.parametrize("a,b", [([], []), ([[1.0, 2.0]], [[1.0, 2.0]])])
def test_correlation_requires_nonempty_vectors(a, b):
    with pytest.raises(ValueError, match="non-empty.*vectors"):
        finite_correlation(a, b)


def test_correlation_is_finite_for_extreme_and_singleton_inputs():
    assert finite_correlation([1.0], [1.0]) == 0.0
    assert finite_correlation([1e308, -1e308, 0.0], [1e308, -1e308, 0.0]) == pytest.approx(1.0)
    assert finite_correlation([1e-30, -1e-30], [-1e-30, 1e-30]) == pytest.approx(-1.0)


@pytest.mark.parametrize("centers", [np.empty((0, 1)), np.empty((2, 0))])
def test_alignment_requires_nonempty_states_and_features(centers):
    with pytest.raises(ValueError, match="non-empty"):
        align_state_centers(centers, centers)


@pytest.mark.parametrize("permutation", [(0, 0), (1, 2), (-1, 0), (0.5, 1)])
def test_alignment_requires_a_bijection(permutation):
    with pytest.raises(ValueError, match="permutation"):
        StateAlignment(permutation, 0.0)


def test_alignment_assignment_matches_exhaustive_reference():
    rng = np.random.default_rng(91)
    for n in range(1, 6):
        cost = rng.normal(size=(n, n))
        assignment = _hungarian(cost)
        actual = cost[np.arange(n), assignment].sum()
        expected = min(
            sum(cost[i, p[i]] for i in range(n)) for p in itertools.permutations(range(n))
        )
        assert actual == pytest.approx(expected)


def test_large_transition_weights_normalize_without_overflow():
    with np.errstate(over="raise", invalid="raise"):
        result = evaluate_context_replication(
            object(),
            object(),
            [-1, 1],
            effect_matrix=lambda m, c: np.full((3, 3), 1e308),
            config=POLICY,
            centers=_centers,
        )
    assert not result.selected
    assert np.isfinite(result.replication_corr)


@pytest.mark.parametrize("state_effect", [False, True])
def test_effect_callbacks_may_reuse_a_scratch_buffer(state_effect):
    scratch = np.ones((3, 3))
    slope = np.array([[0.0, 0.2, -0.2], [0.2, 0.0, -0.2], [-0.2, 0.2, 0.0]])

    def callback(model, context):
        scratch[:] = 1.0 + float(context) * slope
        return scratch

    evaluator = evaluate_state_context_replication if state_effect else evaluate_context_replication
    result = evaluator(
        object(),
        object(),
        [-1.0, 1.0],
        config=ContextEvidenceConfig(0.8, 0.05),
        centers=_centers,
        **({"effect_tensor": callback} if state_effect else {"effect_matrix": callback}),
    )
    assert result.selected
    assert result.replication_corr == pytest.approx(1.0)


def test_state_effect_rejects_empty_trailing_axis():
    with pytest.raises(ValueError, match="non-empty"):
        evaluate_state_context_replication(
            object(),
            object(),
            [-1.0, 1.0],
            config=POLICY,
            centers=_centers,
            effect_tensor=lambda m, c: np.empty((3, 0)),
        )


def test_assignment_handles_large_finite_costs_without_overflow():
    cost = np.array([[1e308, 2e307], [2e307, 1e308]])
    with np.errstate(over="raise", invalid="raise"):
        assert tuple(_hungarian(cost)) == (1, 0)


def test_state_effect_direction_handles_extreme_finite_amplitudes():
    with np.errstate(over="raise", invalid="raise"):
        result = evaluate_state_context_replication(
            object(),
            object(),
            [-1.0, 1.0],
            config=ContextEvidenceConfig(0.5, 0.1),
            centers=_centers,
            effect_tensor=lambda m, c: np.full((3, 2), c * 1e200),
        )
    assert result.selected
    assert result.replication_corr == pytest.approx(1.0)
    assert result.direction_agreement == 1.0
