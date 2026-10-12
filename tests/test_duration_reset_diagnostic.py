"""Independent controls for the bounded reset-capacity experiment."""

import numpy as np
import pytest
import torch

from scripts.diagnose_duration_resets import (
    censored_duration_mle,
    fit,
    independent_log_likelihood,
)
from scripts.diagnose_model_conform_duration import D, PMF, TAIL, generate, law


def test_generator_no_resets_makes_episode_boundaries_observable_and_keeps_default():
    matrix = np.array([[0, 1], [1, 0]])
    _, labels, ledger = generate(9, tail=True, count=2, steps=40, transition=matrix)
    for y, episodes in zip(labels, ledger):
        assert np.sum(y[1:] != y[:-1]) == len(episodes) - 1
        assert all(a["state"] != b["state"] for a, b in zip(episodes, episodes[1:]))
    default = generate(19, tail=False)
    explicit = generate(19, tail=False, transition=np.array([[0.02, 0.98], [0.98, 0.02]]))
    np.testing.assert_array_equal(default[0], explicit[0])
    assert default[2] == explicit[2]
    with pytest.raises(ValueError, match="normalized"):
        generate(19, tail=False, transition=np.ones((2, 2)))


def test_labeled_tail_mle_uses_censored_exposure_without_an_end():
    ledger = [
        [
            {"state": 0, "observed": D + 2, "censored": False},
            {"state": 0, "observed": D + 3, "censored": True},
        ]
    ]
    p, h, history = censored_duration_mle(ledger, tail=True, iterations=3)
    assert p[0, -1] == 1
    assert h[0] == pytest.approx(1 / 6)
    assert np.max(np.diff(history)) <= 1e-10


def test_labeled_finite_censoring_allocates_only_surviving_buckets():
    ledger = [[{"state": 0, "observed": 6, "censored": True}]]
    p, h, history = censored_duration_mle(ledger, tail=False, iterations=2)
    np.testing.assert_array_equal(p[0, :5], 0)
    np.testing.assert_allclose(p[0, 5:], 1 / 3)
    np.testing.assert_array_equal(h, 1)
    assert history[-1] == pytest.approx(0)


@pytest.mark.parametrize("tail", [False, True])
def test_numpy_likelihood_matches_model_and_transition_controls(tail):
    transition = np.array([[0.02, 0.98], [0.98, 0.02]])
    x, _, _ = generate(7, tail=tail, count=2, steps=18)
    for arm in ("age_specific_resets", "tied_resets", "no_resets", "known_transition"):
        model = fit(11, tail, x, transition, arm, 2)
        p, h = law(model)
        with torch.no_grad():
            trans = model.dist.transition.log_matrix(T=1).exp().reshape(2, D, 2).numpy()
            ll = float(model.log_likelihood(torch.tensor(x, dtype=torch.float64), reduce=True))
        assert independent_log_likelihood(x, p, h, trans) == pytest.approx(ll, abs=1e-6)
        assert model._diagnostic_completed_iterations == 2
        assert not model.duration_logits_bias.requires_grad
        if arm == "no_resets":
            np.testing.assert_array_equal(trans[0, :, 0], 0)
            np.testing.assert_array_equal(trans[1, :, 1], 0)
        if arm == "tied_resets":
            np.testing.assert_allclose(trans, np.repeat(trans[:, :1], D, axis=1), atol=0)
        if arm == "known_transition":
            np.testing.assert_allclose(trans, np.repeat(transition[:, None], D, axis=1))
    np.testing.assert_allclose(PMF.sum(1), 1)
    assert np.all(TAIL > 0)
