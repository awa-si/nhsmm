"""Focused backend-selection smoke, including automatic Python fallback."""

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
        module = f"nhsmm.models.native_{'hmm' if mode == 'hmm' else 'hsmm'}"
        native_function = object()
        result = torch.tensor([13.0])
        with (
            patch("nhsmm.native_backend.native_library_path", return_value="/unused/native.so"),
            patch(
                "nhsmm.native_backend.load_native_kernel",
                return_value=(object(), native_function),
            ) as loader,
            patch(
                f"{module}.native_{'hmm' if mode == 'hmm' else 'hsmm'}_likelihood",
                return_value=result,
            ) as likelihood,
        ):
            assert model.backend == "auto"
            sequence = object()
            assert observation_log_likelihood(model, sequence) is result
            loader.assert_called_once_with(mode, "/unused/native.so")
            likelihood.assert_called_once_with(model, sequence, None, native_function)

        model.set_backend("python")
        assert model._native_kernel() is None
        model.set_backend("auto", library_path="/missing/native.so")
        assert model._native_kernel() is None
        model.set_backend("rust", library_path="/missing/native.so")
        try:
            model._native_kernel()
        except RuntimeError as exc:
            assert "unavailable" in str(exc)
        else:
            raise AssertionError("strict Rust accepted a missing library")
        model.set_backend("python")
        assert torch.isfinite(model.log_likelihood(torch.zeros(3, 1))).all()


if __name__ == "__main__":
    main()
