"""Boundary checks for the zero-update frozen-artifact diagnostic."""

import numpy as np
import pytest

from scripts.diagnose_trading_gradient_flow import FEATURES, observations


def protocol():
    return {
        "features": list(FEATURES), "ranges_rows_half_open": {"train": [0, 36000]},
        "window_anchors": {"train": [0]}, "window_width": 480,
        "feature_scaler_train_only": {"mean": [0.] * 5, "std": [1.] * 5},
    }


def test_observation_loader_needs_no_ground_truth():
    data = {name: np.arange(480, dtype=np.float32) for name in FEATURES}
    result = observations(data, protocol(), "train", smoke=True)
    assert result.shape == (1, 32, 5)
    np.testing.assert_array_equal(result[:, :, 0], np.arange(32)[None, :])


def test_test_split_is_never_admitted():
    with pytest.raises(ValueError, match="only train and valid"):
        observations({}, {}, "test")


@pytest.mark.parametrize("anchors", [[-1], [35521], [0, 100], [480, 0]])
def test_observation_loader_rejects_invalid_windows(anchors):
    p = protocol()
    p["window_anchors"]["train"] = anchors
    with pytest.raises(ValueError, match="window"):
        observations({}, p, "train")


def test_observation_loader_rejects_nonfinite_scaler():
    p = protocol()
    p["feature_scaler_train_only"]["std"][0] = float("nan")
    data = {name: np.zeros(480) for name in FEATURES}
    with pytest.raises(ValueError, match="scale"):
        observations(data, p, "train")

def test_duration_summary_uses_model_dimensions_and_gate():
    import torch
    from nhsmm import ModelConfig, NHSMM
    from scripts.diagnose_trading_gradient_flow import duration_law

    model = NHSMM(ModelConfig(n_states=4, n_features=5, max_duration=64,
                              duration_tail=True, causal=True, verbose=False), device="cpu")
    model.initialize_distributions()
    with torch.no_grad():
        model.duration_logits_bias[:, 0] = -4.0
    pmf, hazard = duration_law(model)
    assert pmf.shape == (4, 64)
    assert hazard.shape == (4,)
    np.testing.assert_allclose(
        pmf, model.dist.duration.log_matrix(
            soft_dmax=model.duration_logits_bias
        ).exp().detach().numpy()[0, 0], rtol=1e-6,
    )


def test_training_reconstruction_restores_flags_and_preserves_weights(tmp_path):
    import torch
    from nhsmm import ModelConfig, NHSMM, save_artifact
    from scripts.diagnose_trading_gradient_flow import gradient_summary, training_model

    original = NHSMM(ModelConfig(n_states=4, n_features=5, max_duration=64,
                                 duration_tail=True, causal=True, verbose=False), device="cpu")
    original.initialize_distributions()
    path = save_artifact(original, tmp_path / "model.pt")
    restored = training_model(path)
    for name, parameter in restored.named_parameters():
        assert parameter.requires_grad == dict(original.named_parameters())[name].requires_grad
    for name, tensor in restored.state_dict().items():
        torch.testing.assert_close(tensor, original.state_dict()[name], rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="missing expected training gradient"):
        gradient_summary(restored)
    restored.log_likelihood(torch.randn(1, 8, 5), reduce=True).backward()
    assert set(gradient_summary(restored)["groups"]) >= {
        "duration", "duration_gate", "tail_hazard", "emission", "transition", "initial",
    }
