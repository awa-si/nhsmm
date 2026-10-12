"""Common-target timing, train-only mapping and family initialization controls."""

import numpy as np
import pytest
import torch

from nhsmm import ModelConfig
from scripts.diagnose_trading_feature_contract import (
    MatchedFamilyModel,
    WARMUP,
    aligned_probability,
    fit_return_head,
    return_design,
    return_scores,
    state_scores,
)


def test_permutation_maps_probability_columns_instead_of_refitting_on_test():
    p = np.array([[[0.1, 0.2, 0.3, 0.4]]])
    np.testing.assert_array_equal(aligned_probability(p, (2, 0, 3, 1)), [[[0.2, 0.4, 0.1, 0.3]]])


def test_next_return_design_uses_only_current_return_and_posterior():
    r = np.arange(100, dtype=float)[None]
    p = np.full((1, 100, 4), 0.25)
    design = return_design(r, p, 0, 1)
    np.testing.assert_array_equal(design[:, 1], np.arange(WARMUP, 99))
    assert design.shape == (35, 6)
    changed = r.copy()
    changed[:, -1] = 999
    np.testing.assert_array_equal(design, return_design(changed, p, 0, 1))


def test_train_return_head_is_frozen_under_test_mutation_and_scores_same_target():
    rng = np.random.default_rng(6)
    r = rng.normal(0, 0.002, (4, 100))
    p = np.full((4, 100, 4), 0.25)
    head = fit_return_head(r, p)
    baseline = fit_return_head(r, None)
    test = rng.normal(0, 0.002, (2, 100))
    assert return_scores(test, p[:2], head)["pairs"] == 70
    assert return_scores(test, None, baseline)["pairs"] == 70
    assert return_scores(test, p[:2], head)["next_return_log_score"] == pytest.approx(
        return_scores(test, None, baseline)["next_return_log_score"], abs=1e-12
    )
    before = head["beta"].copy()
    test[:] = 100
    return_scores(test, p[:2], head)
    np.testing.assert_array_equal(before, head["beta"])


def test_Student_t_initial_base_variance_matches_Gaussian_cluster_variance():
    torch.manual_seed(8)
    x = torch.cat([torch.randn(60, 2) * 0.4 - 2, torch.randn(60, 2) * 0.7 + 2])
    models = []
    for family in ("gaussian", "studentt"):
        model = MatchedFamilyModel(
            ModelConfig(
                n_states=2,
                n_features=2,
                max_duration=4,
                use_context_encoder=False,
                emission_type=family,
            ),
            device="cpu",
        )
        model.initialize_distributions()
        torch.manual_seed(9)
        model._initialize_emission_from_observations(x, cluster_variance=True)
        models.append(model)
    gaussian, student = [m.dist.emission for m in models]
    torch.testing.assert_close(gaussian.mu, student.loc)
    variance = torch.nn.functional.softplus(gaussian.log_var)
    scale = torch.nn.functional.softplus(student.scale_param)
    df = torch.nn.functional.softplus(student.dof) + 2
    torch.testing.assert_close(
        variance, scale.square() * df[:, None] / (df[:, None] - 2), rtol=1e-6, atol=1e-6
    )
    assert np.isfinite(fit_return_head(np.ones((2, 100)), None)["variances"]).all()


def test_common_return_score_in_raw_units_matches_independent_normal_density():
    returns = np.linspace(-0.004, 0.005, 100)[None]
    posterior = np.full((1, 100, 4), 0.25)
    parameters = {"mean": 0.001, "scale": 0.003, "beta": np.zeros(6), "variances": np.full(4, 2.0)}
    actual = return_scores(returns, posterior, parameters)
    distribution = torch.distributions.Normal(
        torch.tensor(0.001, dtype=torch.float64),
        torch.tensor(0.003 * np.sqrt(2), dtype=torch.float64),
    )
    target = torch.tensor(returns[:, WARMUP + 1 :].reshape(-1))
    assert actual["next_return_log_score"] == pytest.approx(
        float(distribution.log_prob(target).mean()), abs=1e-12
    )


def test_Float32_zero_state_probability_uses_explicit_finite_log_score_floor():
    p = np.zeros((1, 100, 4), dtype=np.float32)
    p[:, :, 0] = 1
    labels = np.ones((1, 100), dtype=int)
    result = state_scores(p, labels, (0, 1, 2, 3))
    assert np.isfinite(result["state_log_score"])
    assert result["state_log_score"] == pytest.approx(np.log(1e-300))
    assert result["zero_true_state_probabilities"] == 100 - WARMUP


@pytest.mark.parametrize("family", ["gaussian", "studentt"])
def test_both_family_production_optimize_paths_complete_one_real_update(family):
    cfg = ModelConfig(
        n_states=2,
        n_features=2,
        max_duration=4,
        causal=True,
        duration_tail=True,
        use_context_encoder=False,
        n_init=1,
        max_iter=1,
        convergence_stop=False,
        use_scheduler=False,
        emission_type=family,
        emission_init_mode="kmeans" if family == "studentt" else "kmeans_cluster_variance",
    )
    model = MatchedFamilyModel(cfg, device="cpu")
    model.initialize_distributions()
    torch.manual_seed(8)
    x = torch.randn(2, 12, 2)
    model.optimize(x)
    assert torch.isfinite(model._convergence.scores).sum(0).tolist() == [1]
