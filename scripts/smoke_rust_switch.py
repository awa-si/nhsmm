"""Focused CPU smoke for the research Rust mode switch, without a native library."""

from unittest.mock import patch

import torch

from nhsmm import ModelConfig, NHSMM
from nhsmm.models.objective import observation_log_likelihood


def main() -> None:
    for mode in ("hmm", "hsmm", "nhsmm"):
        options = {"max_duration": 3} if mode != "hmm" else {}
        model = NHSMM(
            ModelConfig.for_mode(mode, n_states=2, n_features=1, seed=7, **options),
            device="cpu",
        )
        model.initialize_distributions()
        module = f"nhsmm.models.research_{'hmm' if mode == 'hmm' else 'hsmm'}_rust"
        native_function = object()
        result = torch.tensor([13.0])
        with (
            patch(f"{module}.load_kernel", return_value=(object(), native_function)) as loader,
            patch(
                f"{module}.research_{'hmm' if mode == 'hmm' else 'hsmm'}_likelihood",
                return_value=result,
            ) as likelihood,
        ):
            model.enable_research_rust("/unused/research.so")
            loader.assert_called_once_with("/unused/research.so")
            sequence = object()
            assert observation_log_likelihood(model, sequence) is result
            likelihood.assert_called_once_with(model, sequence, None, native_function)
        model.disable_research_rust()
        assert model._research_rust is None
        x = torch.zeros(3, 1)
        assert torch.isfinite(model.log_likelihood(x)).all()


if __name__ == "__main__":
    main()
