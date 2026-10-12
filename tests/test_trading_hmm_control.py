"""Independent state-only recursion and matched HMM restart controls."""

import itertools

import numpy as np
import pytest
import torch

from nhsmm import ModelConfig
from scripts.diagnose_trading_calibration_budget import BudgetCheckpointModel
from scripts.diagnose_trading_hmm_control import (
    HMMControlModel,
    checked_hmm_filter,
    hsmm_initial_states,
    numpy_hmm,
)


def test_numpy_hmm_matches_complete_prefix_path_enumeration():
    initial = np.array([[0.4, 0.6]])
    transition = np.array(
        [
            [
                [[0.8, 0.2], [0.1, 0.9]],
                [[0.6, 0.4], [0.3, 0.7]],
                [[0.9, 0.1], [0.4, 0.6]],
                [[0.7, 0.3], [0.2, 0.8]],
            ]
        ]
    )
    emission = np.array([[[0.5, 0.2], [0.1, 0.8], [0.9, 0.1], [0.2, 0.6]]])
    actual, evidence = numpy_hmm(np.log(initial), np.log(transition), np.log(emission))
    for t in range(4):
        posterior = np.zeros(2)
        for path in itertools.product(range(2), repeat=t + 1):
            mass = initial[0, path[0]] * emission[0, 0, path[0]]
            for row in range(1, t + 1):
                mass *= (
                    transition[0, row - 1, path[row - 1], path[row]] * emission[0, row, path[row]]
                )
            posterior[path[-1]] += mass
        np.testing.assert_allclose(
            np.exp(actual[0, t]), posterior / posterior.sum(), rtol=0, atol=1e-12
        )
    assert evidence[0] == pytest.approx(np.log(posterior.sum()), abs=1e-12)


@pytest.mark.parametrize("family", ["gaussian", "studentt"])
def test_reconstructed_hsmm_initializer_matches_real_production_checkpoint_and_hmm(family):
    cfg = ModelConfig(
        n_states=2,
        n_features=2,
        max_duration=4,
        duration_tail=True,
        causal=True,
        context_dim=1,
        use_context_encoder=False,
        n_init=1,
        max_iter=1,
        use_scheduler=False,
        convergence_stop=False,
        seed=901,
        emission_type=family,
        emission_init_mode="kmeans" if family == "studentt" else "kmeans_cluster_variance",
    )
    torch.manual_seed(8)
    x, c = torch.randn(2, 10, 2), torch.randn(2, 10, 1)
    em, prior = hsmm_initial_states(cfg, x)
    model = BudgetCheckpointModel(cfg, device="cpu", checkpoint_iteration=0)
    model.initialize_distributions()
    model.optimize(x, context=c)
    for prefix, params in (("dist.emission.", em), ("dist.initial.", prior)):
        for key, value in params.items():
            torch.testing.assert_close(value, model.checkpoint[prefix + key], rtol=0, atol=0)
    hmm = HMMControlModel(
        cfg.with_overrides(max_duration=1, duration_tail=False, max_iter=3),
        device="cpu",
        emission_state=em,
        initial_state=prior,
        checkpoint_iteration=0,
    )
    hmm.initialize_distributions()
    hmm.optimize(x, context=c)
    assert torch.isfinite(hmm._convergence.scores).sum(0).tolist() == [3]
    for prefix, params in (("dist.emission.", em), ("dist.initial.", prior)):
        for key, value in params.items():
            torch.testing.assert_close(value, hmm.checkpoint[prefix + key], rtol=0, atol=0)
    hmm.eval()
    logp, check = checked_hmm_filter(hmm, x, c)
    assert logp.shape == (2, 10, 2)
    assert check["max_log_posterior_error"] < 1e-5
    assert check["likelihood_error_per_row"] < 1e-5
    with pytest.raises(ValueError, match="D1"):
        checked_hmm_filter(model, x, c)
