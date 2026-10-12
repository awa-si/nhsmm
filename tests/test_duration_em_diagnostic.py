"""Independent path enumeration of unlabeled EM sufficient statistics."""

import numpy as np
import pytest
import torch

from scripts.diagnose_duration_em import expectation, em_fit, gaussian_emissions, model_from_law
from scripts.diagnose_model_conform_duration import D, generate
from scripts.diagnose_tail_coverage import TRANSITION


def enumerate_counts(emission, pmf, hazard, transition):
    width = pmf.shape[1]
    total = 0.0
    buckets = np.zeros_like(pmf)
    ends, stays = np.zeros(2), np.zeros(2)

    def walk(t, state, bucket, age, probability, counts, tail_ends, tail_stays):
        nonlocal total, buckets, ends, stays
        probability *= emission[t, state]
        if t == len(emission) - 1:
            total += probability
            buckets += probability * counts
            ends += probability * tail_ends
            stays += probability * tail_stays
            return
        if age < bucket:
            walk(t + 1, state, bucket, age + 1, probability, counts, tail_ends, tail_stays)
            return
        is_tail = bucket == width - 1
        if is_tail:
            new_stays = tail_stays.copy()
            new_stays[state] += 1
            walk(
                t + 1,
                state,
                bucket,
                age,
                probability * (1 - hazard[state]),
                counts,
                tail_ends,
                new_stays,
            )
        end_probability = hazard[state] if is_tail else 1
        for dest in range(2):
            for drawn in range(width):
                new_counts = counts.copy()
                new_counts[dest, drawn] += 1
                new_ends = tail_ends.copy()
                new_ends[state] += is_tail
                walk(
                    t + 1,
                    dest,
                    drawn,
                    0,
                    probability * end_probability * transition[state, dest] * pmf[dest, drawn],
                    new_counts,
                    new_ends,
                    tail_stays,
                )

    for state in range(2):
        for bucket in range(width):
            counts = np.zeros_like(pmf)
            counts[state, bucket] = 1
            walk(0, state, bucket, 0, 0.5 * pmf[state, bucket], counts, np.zeros(2), np.zeros(2))
    return np.log(total), buckets / total, ends / total, stays / total


@pytest.mark.parametrize("width", [1, 2, 3])
def test_forward_backward_counts_match_complete_generative_path_enumeration(width):
    rng = np.random.default_rng(8)
    pmf = rng.dirichlet(np.ones(width), size=2)
    hazards = np.array([0.2, 0.7])
    emissions = rng.uniform(0.2, 0.9, size=(1, 4, 2))
    actual = expectation(emissions, pmf, hazards, TRANSITION)
    expected = enumerate_counts(emissions[0], pmf, hazards, TRANSITION)
    for key, value in zip(("ll", "bucket_counts", "tail_ends", "tail_stays"), expected):
        np.testing.assert_allclose(actual[key], value, atol=1e-12)


def test_single_row_is_right_censored_with_no_tail_end_or_stay():
    pmf = np.full((2, D), 1 / D)
    actual = expectation(np.array([[[0.4, 0.8]]]), pmf, np.array([0.2, 0.7]), TRANSITION)
    np.testing.assert_array_equal(actual["tail_ends"], 0)
    np.testing.assert_array_equal(actual["tail_stays"], 0)
    np.testing.assert_allclose(actual["bucket_counts"].sum(1), [1 / 3, 2 / 3])


def test_unlabeled_EM_is_monotone_and_matches_public_model_likelihood():
    x, _, _ = generate(51, tail=True, count=2, steps=24)
    p, h, history = em_fit(x, TRANSITION, 4)
    assert len(history) == 5
    assert np.min(np.diff(history)) >= -1e-8
    np.testing.assert_allclose(p.sum(1), 1)
    assert np.all((h > 0) & (h <= 1))
    model = model_from_law(52, p, h)
    with torch.no_grad():
        ll = float(model.log_likelihood(torch.tensor(x, dtype=torch.float64), reduce=True))
    assert ll == pytest.approx(history[-1], abs=1e-10)
    assert expectation(gaussian_emissions(x.astype(float)), p, h, TRANSITION)[
        "ll"
    ] == pytest.approx(ll, abs=1e-10)
