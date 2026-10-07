from __future__ import annotations

import math

import pytest
import torch

from tests.reference.hsmm_exact import duration_hazards_from_pmf, duration_hazards_from_tail_mass


def test_tail_hazard_one_reduces_to_current_finite_support_hazards() -> None:
    torch.manual_seed(1201)
    log_mass = torch.randn(3, 2, 5, dtype=torch.float64).log_softmax(-1)
    tail_end = torch.ones(3, 2, dtype=torch.float64)

    expected_end, expected_continue = duration_hazards_from_pmf(log_mass)
    actual_end, actual_continue = duration_hazards_from_tail_mass(log_mass, tail_end)

    torch.testing.assert_close(actual_end, expected_end, atol=0.0, rtol=0.0)
    torch.testing.assert_close(actual_continue, expected_continue, atol=0.0, rtol=0.0)


def test_tail_bucket_is_memoryless_and_remains_normalized() -> None:
    log_mass = torch.log(torch.tensor([[[0.2, 0.3, 0.5]]], dtype=torch.float64))
    tail_end = torch.tensor([[0.25]], dtype=torch.float64)

    log_end, log_continue = duration_hazards_from_tail_mass(log_mass, tail_end)

    torch.testing.assert_close(log_end[0, 0, 2].exp(), torch.tensor(0.25, dtype=torch.float64))
    torch.testing.assert_close(log_continue[0, 0, 2].exp(), torch.tensor(0.75, dtype=torch.float64))
    total = torch.logaddexp(log_end, log_continue).exp()
    torch.testing.assert_close(total, torch.ones_like(total), atol=2e-15, rtol=2e-15)


def test_implied_tail_duration_distribution_normalizes() -> None:
    q_tail = 0.37
    h_tail = 0.23
    finite_mass = 0.63

    tail_sum = sum(q_tail * ((1.0 - h_tail) ** offset) * h_tail for offset in range(1000))

    assert finite_mass + tail_sum == pytest.approx(1.0, abs=1e-12)


def test_d1_tail_hazard_one_matches_every_boundary_end_case() -> None:
    log_mass = torch.zeros(4, 3, 1, dtype=torch.float64)
    tail_end = torch.ones(4, 3, dtype=torch.float64)

    log_end, log_continue = duration_hazards_from_tail_mass(log_mass, tail_end)

    assert torch.equal(log_end, torch.zeros_like(log_end))
    assert torch.isneginf(log_continue).all()


def test_tail_contract_rejects_invalid_tail_hazard() -> None:
    log_mass = torch.zeros(1, 1, 1, dtype=torch.float64)
    for value in (0.0, -0.1, 1.1, math.inf, math.nan):
        with pytest.raises((ValueError, RuntimeError)):
            duration_hazards_from_tail_mass(
                log_mass,
                torch.tensor([[value]], dtype=torch.float64),
            )
