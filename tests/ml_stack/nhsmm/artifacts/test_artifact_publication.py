"""Artifact publication must leave readers a complete old or complete new model."""

import os
from pathlib import Path

import nhsmm.artifact as artifact
import pytest
import torch
from nhsmm import NHSMM, ModelConfig, load_artifact, save_artifact


def model(dtype=torch.float64, emission="gaussian"):
    value = NHSMM(
        ModelConfig(
            n_states=2,
            n_features=1,
            max_duration=3,
            duration_tail=True,
            causal=True,
            emission_type=emission,
            dropout=0,
            seed=617,
        ),
        device="cpu",
    )
    value.initialize_distributions()
    return value.to(dtype=dtype).eval()


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("failure", ["serialize", "sync", "replace"])
def test_failed_publication_preserves_previous_artifact(tmp_path, monkeypatch, existing, failure):
    target = tmp_path / "core.pt"
    value = model()
    if existing:
        save_artifact(value, target)
    previous = target.read_bytes() if existing else None

    def partial_save(payload, destination):
        if hasattr(destination, "write"):
            destination.write(b"partial serialization")
        else:
            Path(destination).write_bytes(b"partial serialization")
        raise OSError("injected serialization failure")

    def fail(*args, **kwargs):
        raise OSError("injected " + failure + " failure")

    if failure == "serialize":
        monkeypatch.setattr(artifact.torch, "save", partial_save)
    else:
        monkeypatch.setattr(os, "fsync" if failure == "sync" else "replace", fail)
    with pytest.raises(OSError, match="injected"):
        save_artifact(value, target)
    assert target.exists() == existing
    if existing:
        assert target.read_bytes() == previous
        restored = load_artifact(target, require_causal=True)
        for name, tensor in value.state_dict().items():
            assert torch.equal(tensor, restored.state_dict()[name])
    assert sorted(p.name for p in tmp_path.iterdir()) == (["core.pt"] if existing else [])


@pytest.mark.parametrize("existing", [False, True])
def test_target_stays_unchanged_until_serialization_finishes(tmp_path, monkeypatch, existing):
    target = tmp_path / "core.pt"
    value = model()
    if existing:
        save_artifact(value, target)
    previous = target.read_bytes() if existing else None
    with torch.no_grad():
        value.dist.emission.mu.add_(0.5)
    original = artifact.torch.save

    def observe(payload, destination):
        original(payload, destination)
        assert (target.read_bytes() if target.exists() else None) == previous

    monkeypatch.setattr(artifact.torch, "save", observe)
    assert save_artifact(value, target) == target
    assert load_artifact(target, require_causal=True).config.to_dict() == value.config.to_dict()
    assert [p.name for p in tmp_path.iterdir()] == ["core.pt"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("emission", ["gaussian", "studentt"])
def test_atomic_artifact_roundtrip_and_overwrite(tmp_path, dtype, emission):
    value = model(dtype, emission)
    target = tmp_path / "nested" / "core.pt"
    for update in (0.0, 0.25):
        with torch.no_grad():
            (value.dist.emission.mu if emission == "gaussian" else value.dist.emission.loc).add_(
                update
            )
        save_artifact(value, target)
        restored = load_artifact(target, require_causal=True)
        for name, tensor in value.state_dict().items():
            assert torch.equal(tensor, restored.state_dict()[name])
        assert (
            restored.dist.emission.mu if emission == "gaussian" else restored.dist.emission.loc
        ).dtype == dtype
    assert [p.name for p in target.parent.iterdir()] == ["core.pt"]


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission contract")
def test_replacement_preserves_existing_permissions(tmp_path):
    value = model()
    target = tmp_path / "core.pt"
    save_artifact(value, target)
    target.chmod(0o640)
    save_artifact(value, target)
    assert target.stat().st_mode & 0o777 == 0o640
