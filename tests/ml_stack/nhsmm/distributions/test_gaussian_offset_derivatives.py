"""Real Gaussian emission derivatives beside large common log-density offsets."""

import itertools
import math

import numpy as np
import pytest
import torch

from tests.ml_stack.nhsmm.filtering.test_viterbi_numerics import identical_emissions


def occupancy_derivatives(duration, tail, length, labels, filtering):
    """Enumerate a raw probability chain and prefix-conditioned occupancies."""
    matrix = np.zeros((2 * duration, 2 * duration))
    for state, age in itertools.product(range(2), range(duration)):
        hazard = 0.25 if tail and age == duration - 1 else 1 / (duration - age)
        for dest in range(2):
            matrix[state * duration + age, dest * duration] += hazard * (
                0.9 if state == dest else 0.1
            )
        if age + 1 < duration or tail:
            matrix[state * duration + age, state * duration + min(age + 1, duration - 1)] += (
                1 - hazard
            )
    histories = [((state * duration,), probability) for state, probability in enumerate((0.1, 0.9))]
    derivative = np.zeros((length, 2))
    for t in range(length):
        total = sum(w for _, w in histories)
        selected = sum(w for path, w in histories if path[-1] // duration == labels[t])
        for path, weight in histories:
            for i, cell in enumerate(path):
                if filtering:
                    derivative[i, cell // duration] += weight * (
                        (1 / selected if path[-1] // duration == labels[t] else 0) - 1 / total
                    )
                elif t == length - 1:
                    derivative[i, cell // duration] += weight / total
        if t + 1 < length:
            histories = [
                (path + (dest,), weight * matrix[path[-1], dest])
                for path, weight in histories
                for dest in range(2 * duration)
                if matrix[path[-1], dest] > 0
            ]
    return derivative


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("duration,tail", [(1, False), (3, False), (1, True), (3, True)])
@pytest.mark.parametrize("filtering", [False, True])
def test_actual_gaussian_mean_variance_and_input_gradients(dtype, duration, tail, filtering):
    value = identical_emissions(dtype, True, duration, tail)
    labels = torch.tensor([1, 0, 1, 1])
    g = torch.tensor(
        occupancy_derivatives(duration, tail, 4, labels.tolist(), filtering), dtype=dtype
    )
    large = 1e4 if dtype == torch.float32 else 1e9
    tolerance = 2e-5 if dtype == torch.float32 else 2e-12
    for sign in (-1, 1):
        x = torch.full((4, 1), sign * large, dtype=dtype, requires_grad=True)
        score = (
            value.supervised_filter_log_score(x, labels).sum()
            if filtering
            else value.log_likelihood(x).sum()
        )
        emission = value.dist.emission
        actual = torch.autograd.grad(score, (emission.mu, emission.log_var, x))
        # The fixture has variance softplus(log(expm1(1))) = 1. Its independent
        # density derivatives are x, (x^2 - 1) * sigmoid(raw_variance) / 2, -x.
        derivative_scale = 1 / (1 + math.exp(-float(emission.log_var.detach()[0, 0])))
        expected = (
            g.sum(0)[:, None] * sign * large,
            g.sum(0)[:, None] * 0.5 * (large**2 - 1) * derivative_scale,
            -g.sum(1)[:, None] * sign * large,
        )
        scales = (large, large**2, large)
        for observed, wanted, scale in zip(actual, expected, scales, strict=True):
            assert torch.isfinite(observed).all()
            torch.testing.assert_close(
                observed / scale, wanted / scale, atol=tolerance, rtol=tolerance
            )
