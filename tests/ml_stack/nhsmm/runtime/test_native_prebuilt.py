"""Precompiled Rust kernel selection and fail-closed platform/identity gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from nhsmm import native_backend


@pytest.mark.parametrize("kind", ["hmm", "hsmm"])
def test_verified_packaged_prebuilt_is_selected(kind, monkeypatch):
    monkeypatch.delenv(f"NHSMM_NATIVE_{kind.upper()}_LIB", raising=False)
    monkeypatch.delenv("NHSMM_NATIVE_DISABLE_PREBUILT", raising=False)
    path = native_backend.native_library_path(kind)
    assert path is not None and path.is_file()
    assert "/nhsmm/native/prebuilt/linux_x86_64_glibc/" in str(path)
    _, kernel = native_backend.load_native_kernel(kind, str(path))
    assert kernel.restype is not None


def test_prebuilt_is_not_used_on_wrong_platform(monkeypatch):
    monkeypatch.setattr(native_backend.sys, "platform", "darwin")
    monkeypatch.setattr(native_backend.shutil, "which", lambda _: None)
    assert native_backend.native_library_path("hmm") is None


def test_prebuilt_is_not_used_on_old_glibc(monkeypatch):
    monkeypatch.setattr(native_backend.platform, "libc_ver", lambda: ("glibc", "2.31"))
    monkeypatch.setattr(native_backend.shutil, "which", lambda _: None)
    assert native_backend.native_library_path("hsmm") is None


def test_source_hash_mismatch_fails_closed(monkeypatch):
    base = Path(native_backend.__file__).parent / "native" / "prebuilt" / "linux_x86_64_glibc"
    manifest = json.loads((base / "manifest.json").read_text())
    assert set(manifest["kernels"]) == {"hmm", "hsmm"}
    for kind, item in manifest["kernels"].items():
        root = Path(native_backend.__file__).parent / "native"
        assert (
            hashlib.sha256((root / f"{kind}.rs").read_bytes()).hexdigest() == item["source_sha256"]
        )
        assert (
            hashlib.sha256((base / item["binary"]).read_bytes()).hexdigest()
            == item["binary_sha256"]
        )

    # This test removes any ambient build fallback and deliberately rejects
    # the trusted directory without mutating the shared binary.
    monkeypatch.setattr(native_backend, "_prebuilt_library", lambda _: None)
    monkeypatch.setattr(native_backend.shutil, "which", lambda _: None)
    assert native_backend.native_library_path("hmm") is None


def test_explicit_override_has_priority(monkeypatch, tmp_path):
    custom = tmp_path / "custom.so"
    monkeypatch.setenv("NHSMM_NATIVE_HMM_LIB", str(custom))
    assert native_backend.native_library_path("hmm") == custom


@pytest.mark.parametrize("mode", ["hmm", "hsmm"])
def test_default_prebuilt_likelihood_and_first_derivatives_match_python(mode):
    import copy

    import torch
    from nhsmm import NHSMM, ModelConfig

    kw = {"max_duration": 3} if mode == "hsmm" else {}
    cfg = ModelConfig.for_mode(mode, n_states=3, n_features=1, seed=29, **kw)
    template = NHSMM(cfg, device="cpu").double()
    template.initialize_distributions(jitter=0)
    native, python = copy.deepcopy(template), copy.deepcopy(template)
    python.set_backend("python")
    assert native.backend == "auto"
    x = torch.linspace(-1, 1, 13, dtype=torch.float64).unsqueeze(-1)
    result = []
    for model in (native, python):
        model.zero_grad(set_to_none=True)
        likelihood = model.log_likelihood(x, reduce=True)
        likelihood.backward()
        result.append(
            (
                likelihood.detach(),
                {
                    name: value.grad.clone()
                    for name, value in model.named_parameters()
                    if value.grad is not None
                },
            )
        )
    a, b = result
    torch.testing.assert_close(a[0], b[0], atol=1e-8, rtol=1e-8)
    assert a[1].keys() == b[1].keys()
    for name in a[1]:
        torch.testing.assert_close(a[1][name], b[1][name], atol=1e-8, rtol=1e-7)


def test_tampered_packaged_source_disables_prebuilt(monkeypatch):
    from pathlib import Path

    real_read_bytes = Path.read_bytes

    def modified(self):
        data = real_read_bytes(self)
        if self.name == "hmm.rs" and "nhsmm/native" in str(self):
            return data + b"untrusted-change"
        return data

    monkeypatch.setattr(Path, "read_bytes", modified)
    assert native_backend._prebuilt_library("hmm") is None


def test_native_hmm_structural_zero_uses_python_fallback():
    from unittest.mock import patch

    import torch
    from nhsmm import NHSMM, ModelConfig

    model = NHSMM(ModelConfig.for_mode("hmm", n_states=2, n_features=1), device="cpu")
    model.initialize_distributions(jitter=0)
    model.double()
    model.eval()
    x = torch.linspace(-0.2, 0.2, 5, dtype=torch.float64).unsqueeze(-1)
    pi = torch.tensor([[[0.0, float("-inf")]]], dtype=torch.float64)
    # Valid hard-zero state support, with one reachable initial state.
    with patch.object(model.dist.initial, "log_matrix", return_value=pi):
        model.set_backend("auto")
        score = model.log_likelihood(x, reduce=True)
        assert torch.isfinite(score)
        model.set_backend("rust")
        with pytest.raises(ValueError, match="hard-zero"):
            model.log_likelihood(x, reduce=True)
