"""Independent duration-law and censoring checks for the diagnostic."""

import math

import numpy as np
import pytest
import torch

from scripts.diagnose_model_conform_duration import (
    D,
    PMF,
    SIGMA,
    MEANS,
    complete_duration_nll,
    duration_mean,
    generate,
    law,
    oracle_duration_fit,
    make_model,
    optimization_evidence,
    survival,
)


def test_geometric_tail_survival_and_mean_match_explicit_mass_sum():
    for p, h in zip(PMF, [0.12, 0.35]):
        lengths = np.arange(1, 4000)
        mass = np.where(
            lengths < D,
            p[np.minimum(lengths - 1, D - 1)],
            p[-1] * h * (1 - h) ** np.maximum(lengths - D, 0),
        )
        assert mass.sum() == pytest.approx(1.0, abs=1e-13)
        assert duration_mean(p, h) == pytest.approx(float(lengths @ mass), abs=1e-12)
        for age in [1, 3, 7, 8, 9, 30]:
            assert survival(p, h, age) == pytest.approx(mass[lengths >= age].sum(), abs=1e-13)


def test_terminal_censoring_uses_survival_instead_of_point_mass():
    hazards = np.array([0.12, 0.35])
    ledger = [
        [
            {"state": 0, "observed": 3, "censored": False},
            {"state": 1, "observed": 10, "censored": True},
        ]
    ]
    expected = -math.log(PMF[0, 2]) - math.log(PMF[1, -1] * (1 - 0.35) ** 2)
    assert complete_duration_nll(PMF, hazards, ledger) == pytest.approx(expected)
    finite = [[{"state": 0, "observed": D, "censored": True}]]
    assert complete_duration_nll(PMF, np.ones(2), finite) == pytest.approx(-math.log(PMF[0, -1]))


@pytest.mark.parametrize("tail", [False, True])
def test_sequences_start_at_episode_and_only_final_episode_is_censored(tail):
    x, y, ledger = generate(17, tail=tail, count=3, steps=128)
    xx, yy, ll = generate(17, tail=tail, count=3, steps=128)
    np.testing.assert_array_equal(x, xx)
    np.testing.assert_array_equal(y, yy)
    assert ll == ledger
    assert np.isfinite(x).all()
    for labels, episodes in zip(y, ledger):
        assert sum(e["observed"] for e in episodes) == 128
        assert all(not e["censored"] for e in episodes[:-1])
        assert episodes[-1]["censored"]
        position = 0
        for e in episodes:
            assert np.all(labels[position : position + e["observed"]] == e["state"])
            if not tail:
                assert e["observed"] <= D
            position += e["observed"]


def test_oracle_fit_preserves_known_emission_law_and_moves_duration():
    x, _, _ = generate(53, tail=True, count=2, steps=24)
    model = oracle_duration_fit(72, True, x, iterations=2)
    np.testing.assert_allclose(model.dist.emission.mu.detach().numpy(), MEANS)
    np.testing.assert_allclose(
        torch.nn.functional.softplus(model.dist.emission.log_var).detach().numpy(),
        SIGMA**2,
        rtol=1e-6,
    )
    p, h = law(model)
    np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-6)
    assert np.max(np.abs(p - 1 / D)) > 1e-5
    assert np.all((h > 0) & (h <= 1))
    assert not model.duration_logits_bias.requires_grad
    assert model.dist.duration.logits.requires_grad
    assert model._diagnostic_completed_iterations == 2


def test_joint_fixed_budget_evidence_excludes_final_restart_score_and_rejects_missing_step():
    x, _, _ = generate(55, tail=False, count=1, steps=12)
    model = make_model(73, False, 3)
    model.optimize(torch.tensor(x))
    result = optimization_evidence(model, 3)
    assert result["completed_iterations_per_restart"] == [3]
    assert not result["convergence_stop"]
    model._convergence.scores[1, 0] = float("nan")
    with pytest.raises(AssertionError, match="fixed iteration budget"):
        optimization_evidence(model, 3)


def test_ledger_coverage_separates_completed_tail_and_censoring():
    from scripts.diagnose_model_conform_duration import ledger_coverage

    rows = [
        [
            {"state": 0, "observed": 8, "censored": False},
            {"state": 1, "observed": 10, "censored": True},
        ]
    ]
    c = ledger_coverage(rows)
    assert c[0]["complete_D_plus_episodes"] == 1
    assert c[1]["complete_D_plus_episodes"] == 0
    assert c[1]["censored_episodes"] == 1
