from __future__ import annotations

import torch

from nhsmm import ModelConfig, NHSMM
from nhsmm.filtering import filter_model_sequence
from nhsmm.runtime import HSMMFilterRuntime


def _model(*, tail: bool, tail_probability: float = 1.0) -> NHSMM:
    model = NHSMM(
        ModelConfig(
            n_states=3,
            n_features=2,
            max_duration=4,
            causal=True,
            duration_tail=tail,
            duration_tail_init_probability=tail_probability,
            dropout=0.0,
            seed=211,
            verbose=False,
        ),
        device="cpu",
    )
    model.initialize_distributions(jitter=0.0)
    model.eval()
    return model


def _copy_shared_state(source: NHSMM, target: NHSMM) -> None:
    source_state = source.state_dict()
    target_state = target.state_dict()
    shared = {name: value for name, value in source_state.items() if name in target_state}
    missing, unexpected = target.load_state_dict(shared, strict=False)
    assert not unexpected
    assert missing == ["dist.duration.tail_end_probability"]


def test_tail_hazard_one_matches_finite_causal_forward_exactly() -> None:
    finite = _model(tail=False)
    tailed = _model(tail=True, tail_probability=1.0)
    _copy_shared_state(finite, tailed)
    with torch.no_grad():
        tailed.dist.duration.tail_end_probability.fill_(1.0)

    torch.manual_seed(212)
    x = torch.randn(2, 9, 2)

    finite_alpha = finite.forward(finite._build_sequence_set(x))
    tailed_alpha = tailed.forward(tailed._build_sequence_set(x))
    torch.testing.assert_close(tailed_alpha, finite_alpha, atol=0.0, rtol=0.0)


def test_tail_batch_filter_and_runtime_are_equivalent() -> None:
    model = _model(tail=True, tail_probability=0.3)
    torch.manual_seed(213)
    x = torch.randn(2, 12, 2)

    batch = filter_model_sequence(model, x)
    runtime = HSMMFilterRuntime(model)
    streamed = [runtime.step(x[:, t]).log_posterior for t in range(x.shape[1])]

    torch.testing.assert_close(
        torch.stack(streamed, dim=1),
        batch.log_posterior,
        atol=1e-6,
        rtol=1e-6,
    )
    assert runtime.state is not None
    assert runtime.state.tail_end_probability is not None
    torch.testing.assert_close(
        runtime.state.tail_end_probability,
        torch.full((2, 3), 0.3),
        atol=1e-6,
        rtol=1e-6,
    )


def test_tail_filter_stays_normalized_beyond_max_duration() -> None:
    model = _model(tail=True, tail_probability=0.2)
    torch.manual_seed(214)
    x = torch.randn(1, 30, 2)
    trace = filter_model_sequence(model, x)

    mass = trace.log_posterior.exp().sum(dim=(2, 3))
    torch.testing.assert_close(mass, torch.ones_like(mass), atol=2e-6, rtol=2e-6)
    assert torch.isfinite(trace.log_posterior[:, -1]).any()
