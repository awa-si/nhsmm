"""Training failure boundaries must preserve public model behavior."""

import pytest
import torch

from nhsmm import ModelConfig, NHSMM


class InterruptedScore(NHSMM):
    """Exercise subclass objective hooks before and after an Adam update."""

    def __init__(self, config, fail_at):
        super().__init__(config, device="cpu")
        self.fail_at = fail_at
        self.score_calls = 0

    def _target_training_score(self, *args, **kwargs):
        self.score_calls += 1
        if self.score_calls == self.fail_at:
            raise RuntimeError("interrupted target score")
        return super()._target_training_score(*args, **kwargs)


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("fail_at,completed", [(1, 0), (2, 1)])
def test_optimize_restores_mode_at_real_update_boundaries(training, fail_at, completed):
    config = ModelConfig(
        n_states=2,
        n_features=1,
        max_duration=3,
        causal=True,
        seed=823,
        n_init=1,
        max_iter=1,
        dropout=0.0,
        use_scheduler=False,
        convergence_stop=False,
        verbose=False,
        optimizer_initialization="current",
    )
    model = InterruptedScore(config, fail_at)
    model.initialize_distributions()
    model.train(training)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    observations = torch.tensor([[0.2], [-0.1], [1.3], [0.4]])

    with pytest.raises(RuntimeError, match="interrupted target score"):
        model.optimize(observations)

    assert all(module.training is training for module in model.modules())
    assert model.score_calls == fail_at
    steps = [int(state["step"]) for state in model._optimizer.state.values()]
    if completed:
        assert steps and set(steps) == {completed}
        assert any(
            not torch.equal(before[name], parameter) for name, parameter in model.named_parameters()
        )
    else:
        assert steps == []
        assert all(
            torch.equal(before[name], parameter) for name, parameter in model.named_parameters()
        )
    # Histories describe pre-update scores; an interrupted final score must
    # never be reported as a completed restart.
    assert len(model._optimization_history[0]) == completed
    assert model._restart_scores == []
