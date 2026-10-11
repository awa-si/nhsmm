import numpy as np
import pytest
import torch

from nhsmm import ModelConfig, NHSMM
from nhsmm.config import EPS
from scripts.diagnose_trading_duration_prior import (
    DurationPilotNHSMM,
    geometric_duration_mass,
)


@pytest.mark.parametrize("duration,mean", [(1, 2), (4, 8), (64, 128)])
def test_geometric_bucket_preserves_survival_and_constant_hazard(duration, mean):
    p, h = geometric_duration_mass(duration, mean)
    assert p.sum() == pytest.approx(1)
    for age in range(1, duration):
        assert p[age - 1] / p[age - 1 :].sum() == pytest.approx(h)
    assert p[-1] == pytest.approx((1 - h) ** (duration - 1))


def test_duration_pilot_injects_prior_after_each_fresh_restart():
    cfg = ModelConfig(
        n_states=2,
        n_features=1,
        max_duration=4,
        duration_tail=True,
        emission_init_mode="kmeans_cluster_variance",
        verbose=False,
    )
    model = DurationPilotNHSMM(cfg, device="cpu")
    model.duration_pilot_init = "geometric128"
    model.initialize_distributions()
    x = torch.linspace(-1, 1, 12)[:, None]
    old = model.dist.duration
    p, h = geometric_duration_mass(4, 128)
    for restart in range(2):
        model._initialize_run_state(restart, observations=x)
        assert model.dist.duration is not old
        old = model.dist.duration
        np.testing.assert_allclose(
            model.dist.duration.logits.detach().softmax(-1), np.broadcast_to(p, (2, 4)), rtol=1e-6
        )
        assert torch.all(model.dist.duration.tail_probability() == h)


@pytest.mark.parametrize("initial,direction", [(1 / 128, 1.0), (1.0, -1.0)])
def test_optimize_tail_can_recover_after_step_crosses_probability_boundary(initial, direction):
    class DirectionChangeNHSMM(NHSMM):
        def _compute_loss(self, X, context=None, *, it=0, **kwargs):
            tail = self.dist.duration.tail_end_probability
            self.raw_seen.append(float(tail.detach()[0]))
            sign = direction if it == 0 else -direction
            loss = sign * self.dist.duration.tail_probability().sum()
            return -loss, loss

    cfg = ModelConfig(
        n_states=1,
        n_features=1,
        max_duration=4,
        causal=True,
        duration_tail=True,
        duration_tail_init_probability=initial,
        n_init=1,
        max_iter=6,
        lr=0.008,
        use_scheduler=False,
        convergence_stop=False,
        verbose=False,
    )
    model = DirectionChangeNHSMM(cfg, device="cpu")
    model.raw_seen = []
    model.initialize_distributions()
    model.optimize(torch.zeros(1, 8, 1))
    assert all(EPS * 0.99 <= value <= 1.0 for value in model.raw_seen)
    final = float(model.dist.duration.tail_end_probability.detach()[0])
    assert EPS < final < 1.0
