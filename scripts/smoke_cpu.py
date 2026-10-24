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


def main():
    torch.set_num_threads(1)
    cases = []
    for dtype in (torch.float32, torch.float64):
        for tail in (False, True):
            config = ModelConfig(
                n_states=2,
                n_features=1,
                max_duration=3,
                duration_tail=tail,
                causal=True,
                dropout=0.0,
                seed=7,
            )
            assert config.with_overrides(max_iter=20).max_iter == 20
            model = NHSMM(config, device="cpu")
            model.initialize_distributions()
            model.to(dtype=dtype).eval()
            x = torch.tensor([[0.0], [0.1], [-0.2], [0.3]], dtype=dtype)
            with torch.inference_mode():
                expected = filter_model_sequence(model, x)
            with TemporaryDirectory(prefix="nhsmm-smoke-") as directory:
                artifact = Path(directory) / "model.pt"
                save_artifact(model, artifact)
                restored = load_artifact(artifact, require_causal=True)
                assert not restored.training
                for name, original in model.state_dict().items():
                    torch.testing.assert_close(
                        restored.state_dict()[name], original, rtol=0, atol=0
                    )
                runtime = HSMMFilterRuntime(restored)
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
                runtime.reset()
                assert runtime.state is None
                replay = runtime.step(x[0], timestamp=1)
                torch.testing.assert_close(
                    replay.state_posterior[0],
                    expected.state_posterior[0, 0],
                    rtol=0,
                    atol=tolerance,
                )
            cases.append({"dtype": str(dtype), "duration_tail": tail})
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
