import numpy as np
import pytest

from scripts.compare_unlabelled_duration_em import expectation
from scripts.joint_gaussian_duration import chunk_statistics as prior_statistics
from scripts.joint_gaussian_duration import initialize_from_observations, scores
from scripts.joint_static_chain import MASK, chunk_statistics, full_mstep, objective, statistics
from scripts.static_chain_check import numerical_check
from scripts.generate_trading_regime_dataset import BASE_TRANSITION
from scripts.verify_isolated_trading_hsmm import sequences


def problem(steps=60):
    x, _ = sequences(401933, count=3, steps=steps, sigma=2.0)
    p = np.full((4, 260), 1 / 260)
    means, variances = initialize_from_observations(x)
    transition = np.array(
        [[0, 0.2, 0.3, 0.5], [0.5, 0, 0.2, 0.3], [0.4, 0.4, 0, 0.2], [0.2, 0.5, 0.3, 0]]
    )
    initial = np.array([0.1, 0.2, 0.3, 0.4])
    return x, p, means, variances, transition, initial


def test_independent_counts_likelihood_and_chunking():
    args = problem()
    x, p, m, v, t, i = args
    s = statistics(*args, chunk=2)
    ll, counts, complete, _ = expectation(scores(x, m, v), p, t, i)
    np.testing.assert_allclose(s["ll"], ll.sum(), atol=1e-8, rtol=0)
    np.testing.assert_allclose(s["duration_counts"], counts, atol=1e-8, rtol=0)
    assert s["transition_counts"].sum() == pytest.approx(complete.sum(), abs=1e-8)
    assert s["initial_counts"].sum() == pytest.approx(len(x), abs=1e-8)
    assert (s["transition_counts"][~MASK] == 0).all()
    full = statistics(*args, chunk=3)
    for k in s:
        np.testing.assert_allclose(s[k], full[k], atol=1e-8, rtol=0)


def test_previous_known_chain_owner_unchanged_equivalence():
    x, p, m, v, _, _ = problem()
    old = prior_statistics(x, p, m, v)
    new = chunk_statistics(x, p, m, v, BASE_TRANSITION, np.full(4, 0.25))
    for k in old:
        np.testing.assert_array_equal(old[k], new[k])


def test_censored_single_row_has_no_transition():
    args = problem()
    args = (args[0][:, :1], *args[1:])
    s = statistics(*args)
    assert s["transition_counts"].sum() == 0
    np.testing.assert_allclose(s["initial_counts"], s["occupancy"], atol=1e-12, rtol=0)
    updated = full_mstep(s)
    np.testing.assert_allclose(updated[3], MASK / 3, atol=1e-12, rtol=0)


def test_public_all_five_parameter_groups_and_filter():
    args = problem()
    for _ in range(2):
        checks = numerical_check(*args)
        assert max(checks.values()) < 1e-7
        assert max(v for k, v in checks.items() if "filter" in k) < 1e-8
        updated = full_mstep(statistics(*args))
        args = (args[0], *updated[:5])


def test_actual_full_em_monotonic_and_inputs_unmodified():
    args = problem()
    original_args = args
    originals = [a.copy() for a in args]
    last = -np.inf
    for _ in range(5):
        s = statistics(*args)
        value = objective(s["ll"], args[1], args[4], args[5])
        assert value >= last - 1e-6
        last = value
        updated = full_mstep(s)
        assert updated[-1]["maximum_normalized_anchored_gradient"] <= 1e-8
        args = (args[0], *updated[:5])
    for original, current in zip(originals, original_args, strict=True):
        np.testing.assert_array_equal(original, current)


@pytest.mark.parametrize("kind", ["self", "zero", "rows", "initial", "nan"])
def test_invalid_chain_laws_rejected(kind):
    x, p, m, v, t, i = problem()
    if kind == "self":
        t[0, 0] = 0.1
    elif kind == "zero":
        t[0, 1] = 0
    elif kind == "rows":
        t[0] *= 2
    elif kind == "initial":
        i[0] = 0
    else:
        t[0, 1] = np.nan
    with pytest.raises(ValueError):
        statistics(x, p, m, v, t, i)
