from __future__ import annotations

import pytest
import torch

from nhsmm import (
    ModelConfig,
    ModelHealthThresholds,
    NHSMM,
    evaluate_model_health,
)


def _model() -> NHSMM:
    model = NHSMM(
        ModelConfig(
            n_states=3,
            n_features=2,
            max_duration=5,
            causal=True,
            dropout=0.0,
            seed=17,
            max_iter=1,
            use_scheduler=False,
            convergence_stop=False,
            verbose=False,
        ),
        device="cpu",
    )
    model.initialize_distributions()
    return model


def test_model_health_report_is_observational_and_normalized() -> None:
    model = _model()
    model.train()
    x = torch.randn(2, 8, 2)

    report = evaluate_model_health(model, x)

    assert model.training
    torch.testing.assert_close(report.occupancy.sum(), torch.tensor(1.0))
    torch.testing.assert_close(report.viterbi_usage.sum(), torch.tensor(1.0))
    assert 1.0 <= report.effective_states <= 3.0
    assert 0.0 <= report.min_occupancy <= report.max_occupancy <= 1.0
    assert 1 <= report.viterbi_states_used <= 3
    assert report.posterior_entropy_mean >= 0.0
    assert report.duration_entropy_mean >= 0.0
    assert 0.0 <= report.duration_peak_mean <= report.duration_peak_max <= 1.0
    assert report.transition_entropy_mean >= 0.0
    assert 0.0 <= report.transition_peak_mean <= report.transition_peak_max <= 1.0
    assert report.as_dict()["healthy"] == report.healthy


def test_model_health_handles_variable_length_sequences() -> None:
    model = _model()
    x = [torch.randn(4, 2), torch.randn(9, 2)]

    decoded = model.predict(x, mode="viterbi", verbose=False)
    report = evaluate_model_health(model, x)

    assert [len(path) for path in decoded] == [4, 9]
    assert report.occupancy.shape == (3,)
    assert report.viterbi_usage.shape == (3,)


def test_model_health_rejects_empty_input() -> None:
    model = _model()

    with pytest.raises(ValueError, match="at least one valid timestep"):
        evaluate_model_health(model, torch.empty(1, 0, 2))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_effective_states": 0.0},
        {"max_state_occupancy": 0.0},
        {"min_viterbi_states": 0},
        {"min_state_occupancy": 1.0},
        {"max_duration_peak": 0.0},
        {"max_transition_peak": 1.1},
    ],
)
def test_model_health_thresholds_validate_contract(kwargs) -> None:
    with pytest.raises(ValueError):
        ModelHealthThresholds(**kwargs)
