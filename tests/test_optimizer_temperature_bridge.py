"""Bounded diagnostic: temperature changes the objective, not only step size."""

import numpy as np
import torch

from scripts.verify_isolated_trading_hsmm import make_model, sequences


def test_unit_temperature_loss_matches_public_likelihood():
    torch.set_num_threads(1)
    x, _ = sequences(401915, count=2, steps=40, sigma=2.0)
    model = make_model(np.full((4, 260), 1 / 260), sigma=2.0)
    observations = torch.tensor(x)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        public = model.log_likelihood(observations, reduce=True)
        exact, loss = model._compute_loss(
            observations, loss_bias=0.0, it=0, max_iter=3, t_min=1.0, t_max=1.0
        )
    torch.testing.assert_close(exact, public, atol=1e-7, rtol=0)
    torch.testing.assert_close(loss, -exact, atol=1e-7, rtol=0)
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    assert model.dist.duration.oracle_support.all()  # Full configured support; no true-law mask.


def test_scheduling_changes_fixed_parameter_likelihood():
    torch.set_num_threads(1)
    x, _ = sequences(401915, count=2, steps=40, sigma=2.0)
    model = make_model(np.full((4, 260), 1 / 260), sigma=2.0)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        scheduled = [
            float(model._compute_loss(torch.tensor(x), loss_bias=0.0, it=i, max_iter=3)[0])
            for i in range(3)
        ]
    assert np.isfinite(scheduled).all()
    assert max(scheduled) - min(scheduled) > 1e-6
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
