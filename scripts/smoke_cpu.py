"""Public CPU package smoke; canonical model tests remain in Nautilus.

doc:nhsmm.public-package-smoke; docs/testing.md
"""

from importlib.metadata import version
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

import nhsmm
from nhsmm import HSMMFilterRuntime, ModelConfig, NHSMM, load_artifact, save_artifact
from nhsmm.filtering import filter_model_sequence
from nhsmm.forecast import REGIMES


def main():
    torch.set_num_threads(1)
    cases = []
    for dtype in (torch.float32, torch.float64):
        for n_states in (2, 4):
            for mode, tail in (("hmm", False), ("hsmm", False), ("hsmm", True), ("nhsmm", False)):
                options = {"duration_tail": tail}
                if mode != "hmm":
                    options["max_duration"] = 3
                config = ModelConfig.for_mode(
                    mode, n_states=n_states, n_features=1, dropout=0.0, seed=7, **options
                )
                assert config.with_overrides(max_iter=20).max_iter == 20
                model = NHSMM(config, device="cpu")
                model.initialize_distributions()
                model.to(dtype=dtype).eval()
                x = torch.tensor(
                    [[-2.0], [-1.1], [0.0], [1.1], [2.0], [0.3], [-0.7], [1.8]], dtype=dtype
                )
                with torch.inference_mode():
                    expected = filter_model_sequence(model, x)
                with TemporaryDirectory(prefix="nhsmm-smoke-") as directory:
                    artifact = Path(directory) / "model.pt"
                    save_artifact(model, artifact)
                    restored = load_artifact(artifact, require_causal=True)
                    assert not restored.training
                    assert restored.config.model_mode == mode
                    for name, original in model.state_dict().items():
                        torch.testing.assert_close(
                            restored.state_dict()[name], original, rtol=0, atol=0
                        )
                    projection = torch.nn.functional.one_hot(
                        torch.arange(n_states) % len(REGIMES), len(REGIMES)
                    ).to(dtype=dtype)
                    runtime = HSMMFilterRuntime(restored, regime_projection=projection)
                    tolerance = 2e-5 if dtype == torch.float32 else 2e-12
                    for step, row in enumerate(x):
                        state = runtime.step(row, timestamp=step + 1)
                        assert torch.isfinite(state.state_posterior).all()
                        torch.testing.assert_close(
                            state.state_posterior[0],
                            expected.state_posterior[0, step],
                            rtol=0,
                            atol=tolerance,
                        )
                    prediction = runtime.forecast()
                    assert prediction.labels == REGIMES
                    torch.testing.assert_close(
                        prediction.probabilities.sum(dim=-1),
                        torch.ones(1, dtype=dtype),
                        rtol=0,
                        atol=tolerance,
                    )
                    runtime.reset()
                    assert runtime.state is None
                    replay = runtime.step(x[0], timestamp=1)
                    torch.testing.assert_close(
                        replay.state_posterior[0],
                        expected.state_posterior[0, 0],
                        rtol=0,
                        atol=tolerance,
                    )
                cases.append(
                    {"mode": mode, "dtype": str(dtype), "duration_tail": tail, "n_states": n_states}
                )
    multivariate = NHSMM(
        ModelConfig.for_mode("hsmm", n_states=2, n_features=3, max_duration=3, seed=11),
        device="cpu",
    )
    multivariate.initialize_distributions()
    multivariate.eval()
    projection = torch.tensor([[1, 0, 0, 0, 0], [0, 0, 1, 0, 0]], dtype=torch.float32)
    runtime = HSMMFilterRuntime(multivariate, regime_projection=projection)
    runtime.step(torch.tensor([0.1, -0.2, 0.3]))
    prediction = runtime.forecast()
    assert prediction.probabilities.shape == (1, len(REGIMES))
    assert prediction.probabilities[0, 0] + prediction.probabilities[0, 2] > 0.999
    try:
        HSMMFilterRuntime(multivariate, regime_projection=torch.zeros(2, len(REGIMES)))
    except ValueError as exc:
        assert "sum to one" in str(exc)
    else:
        raise AssertionError("invalid regime projection was accepted")

    assert not hasattr(nhsmm, "ModelHealthThresholds")
    assert not hasattr(nhsmm, "evaluate_context_effect_replication")
    print(
        json.dumps(
            {
                "scope": "public package smoke; not canonical model qualification",
                "version": version("nhsmm"),
                "module": nhsmm.__file__,
                "torch": torch.__version__,
                "cases": cases,
            }
        )
    )


if __name__ == "__main__":
    main()
