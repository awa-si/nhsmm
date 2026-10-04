from __future__ import annotations

import pytest
import torch

from nhsmm.convergence import Convergence


def test_plateau_converges_after_exact_window_observations() -> None:
    monitor = Convergence(
        n_init=1,
        max_iter=10,
        mode="plateau",
        plateau_window=5,
        plateau_tol=0.0,
        verbose=False,
    )

    results = [monitor.update(1.0, iteration, 0) for iteration in range(5)]

    assert results == [False, False, False, False, True]
    assert monitor.all_converged


def test_delta_zero_tolerances_accept_exact_zero_change() -> None:
    monitor = Convergence(
        n_init=1,
        max_iter=5,
        mode="delta",
        patience=2,
        tol=0.0,
        rel_tol=0.0,
        verbose=False,
    )

    assert not monitor.update(2.0, 0, 0)
    assert not monitor.update(2.0, 1, 0)
    assert monitor.update(2.0, 2, 0)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf")])
def test_update_rejects_non_finite_scores_and_clears_converged_flag(score: float) -> None:
    monitor = Convergence(
        n_init=1,
        max_iter=5,
        mode="plateau",
        plateau_window=2,
        plateau_tol=0.0,
        verbose=False,
    )
    monitor.update(1.0, 0, 0)
    assert monitor.update(1.0, 1, 0)
    assert monitor.all_converged

    with pytest.raises(ValueError, match="score must be finite"):
        monitor.update(score, 2, 0)

    assert not monitor.all_converged


@pytest.mark.parametrize(
    ("iteration", "init_idx"),
    [(-1, 0), (4, 0), (0, -1), (0, 2)],
)
def test_update_rejects_out_of_range_indices(iteration: int, init_idx: int) -> None:
    monitor = Convergence(n_init=2, max_iter=3, verbose=False)

    with pytest.raises(IndexError):
        monitor.update(1.0, iteration, init_idx)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_init": 0},
        {"max_iter": 0},
        {"patience": 0},
        {"plateau_window": 0},
        {"mode": "invalid"},
        {"tol": -1.0},
        {"rel_tol": float("nan")},
        {"plateau_tol": float("inf")},
        {"patience_scale": -1.0},
        {"patience": 3, "patience_max": 2},
    ],
)
def test_constructor_rejects_invalid_contract(kwargs) -> None:
    defaults = {"n_init": 1, "max_iter": 5, "verbose": False}
    defaults.update(kwargs)

    with pytest.raises(ValueError):
        Convergence(**defaults)


class _Optimizer:
    def __init__(self, lr: float):
        self.param_groups = [{"lr": lr}]


class _Scheduler:
    def __init__(self, lr: float):
        self.optimizer = _Optimizer(lr)
        self.scores: list[float] = []

    def step(self, score: float) -> None:
        self.scores.append(score)


def test_auto_patience_scales_with_learning_rate_without_tensor_allocation() -> None:
    monitor = Convergence(
        n_init=1,
        max_iter=10,
        patience=3,
        patience_scale=1.0,
        verbose=False,
    )
    scheduler = _Scheduler(1e-2)
    monitor.attach_scheduler(scheduler)

    assert monitor._effective_patience() == 3
    scheduler.optimizer.param_groups[0]["lr"] = 1e-3
    assert monitor._effective_patience() == 6

    monitor.update(torch.tensor(1.0), 0, 0)
    assert scheduler.scores == [1.0]
