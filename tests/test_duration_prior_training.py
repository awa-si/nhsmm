import pytest
import torch

from nhsmm import ModelConfig, NHSMM
from nhsmm.config import EPS


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


@pytest.mark.parametrize("boundary", [EPS, 1.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_feasible_tail_boundary_preserves_gradient_across_torch_versions(boundary, dtype):
    model = NHSMM(
        ModelConfig(n_states=1, n_features=1, max_duration=2, causal=True, duration_tail=True),
        device="cpu",
    )
    model.initialize_distributions()
    model.to(dtype=dtype)
    parameter = model.dist.duration.tail_end_probability
    with torch.no_grad():
        parameter.fill_(boundary)
    model.dist.duration.tail_probability().sum().backward()
    torch.testing.assert_close(parameter.grad, torch.ones_like(parameter))
