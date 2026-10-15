"""Zero-update gradient audit of the frozen Trading HSMM; no GT or test rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from nhsmm import load_artifact
from scripts.diagnose_model_conform_duration import duration_mean

DATA_SHA = "370e6e9e87e52d779ea208111eeb2e03b0bf21f8b7e88859fcc8c19ac2739fb4"
MODEL_SHA = "925bc1d35721d031d5385e9fa52ffc381eb789e1abfe225d18468878f6b2130b"
FEATURES = ("log_return_12", "trend_48", "ewma_vol_12", "ewma_vol_48", "volume_z_48")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observations(data, protocol, split, smoke=False):
    if split not in ("train", "valid"):
        raise ValueError("only train and valid are admitted")
    bounds = {"train": (0, 36000), "valid": (36000, 48000)}[split]
    if tuple(protocol["ranges_rows_half_open"][split]) != bounds:
        raise ValueError("unexpected chronological split")
    if tuple(protocol["features"]) != FEATURES:
        raise ValueError("unexpected features")
    anchors = protocol["window_anchors"][split]
    width = protocol["window_width"]
    if width != 480 or not anchors or any(
        not isinstance(a, int) or a < bounds[0] or a + width > bounds[1]
        for a in anchors
    ):
        raise ValueError("invalid window bounds")
    if any(b - a < width for a, b in zip(anchors, anchors[1:])):
        raise ValueError("overlapping or unordered windows")
    if smoke:
        anchors, width = anchors[:1], 32
    raw = np.stack([
        np.column_stack([data[name][a:a + width] for name in FEATURES])
        for a in anchors
    ]).astype(np.float32)
    scaler = protocol["feature_scaler_train_only"]
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    if mean.shape != (5,) or std.shape != (5,) or not np.isfinite(mean).all():
        raise ValueError("invalid scaler")
    if not np.isfinite(std).all() or (std <= 0).any() or not np.isfinite(raw).all():
        raise ValueError("invalid observations or scale")
    return torch.tensor((raw - mean) / std)


@torch.no_grad()
def duration_law(model):
    pmf = model.dist.duration.expected_probs(soft_dmax=model.duration_logits_bias)
    hazard = model.dist.duration.tail_probability()
    if hazard is None:
        hazard = torch.ones(model.config.n_states, dtype=pmf.dtype, device=pmf.device)
    return pmf.cpu().numpy(), hazard.cpu().numpy()


def group(name):
    if name == "duration_logits_bias":
        return "duration_gate"
    for component in ("duration", "emission", "transition", "initial"):
        if name.startswith("dist." + component + "."):
            return "tail_hazard" if name.endswith("tail_end_probability") else component
    return "other"


def gradient_summary(model):
    rows = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or name.endswith(".log_temperature"):
            continue
        grad = parameter.grad
        row = rows.setdefault(group(name), {"parameters": [], "squared_norm": 0.0})
        norm2 = 0.0 if grad is None else float(grad.double().square().sum())
        if grad is not None and not torch.isfinite(grad).all():
            raise FloatingPointError("nonfinite gradient: " + name)
        row["parameters"].append({
            "name": name, "numel": parameter.numel(), "gradient_present": grad is not None,
            "gradient_l2": math.sqrt(norm2),
            "nonzero_elements": 0 if grad is None else int(torch.count_nonzero(grad)),
        })
        row["squared_norm"] += norm2
    total2 = sum(row["squared_norm"] for row in rows.values())
    scale = min(1.0, 5.0 / (math.sqrt(total2) + 1e-6))
    for row in rows.values():
        norm2 = row.pop("squared_norm")
        row["gradient_l2"] = math.sqrt(norm2)
        row["squared_norm_fraction"] = norm2 / total2 if total2 else 0.0
        row["post_clip_l2"] = math.sqrt(norm2) * scale
    return {"groups": rows, "global_l2": math.sqrt(total2), "clip_scale": scale}


def run(args):
    torch.set_num_threads(1)
    if digest(args.dataset) != DATA_SHA or digest(args.artifact) != MODEL_SHA:
        raise ValueError("frozen dataset/artifact identity mismatch")
    protocol = json.loads(args.protocol.read_text())
    if protocol["dataset_sha256"] != DATA_SHA or protocol["oracle_model_inputs"]:
        raise ValueError("unexpected baseline protocol")
    # Only these two observation splits are extracted. No labels are requested.
    with np.load(args.dataset, allow_pickle=False) as data:
        train = observations(data, protocol, "train", args.smoke)
        valid = observations(data, protocol, "valid", args.smoke)
    model = load_artifact(args.artifact, device="cpu", require_causal=True)
    model.eval()
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    cfg = model.config
    schedule = 0.3 + 0.7 / (1 + math.exp(10 / cfg.max_iter * (cfg.max_iter - 1 - cfg.max_iter / 2)))
    passes = {}
    for label, temperature in (("target", 1.0), ("last_scheduled", schedule)):
        model.zero_grad(set_to_none=True)
        ll, loss = model._backward_training_loss(
            train, context=None, batch_size=1, loss_bias=cfg.loss_bias,
            it=0, max_iter=1, t_min=temperature, t_max=temperature,
        )
        with torch.no_grad():
            valid_ll = model._target_training_score(valid, None, temperature, batch_size=1)
        passes[label] = {
            "temperature": temperature, "train_ll_per_row": float(ll) / train.shape[:2].numel(),
            "valid_ll_per_row": float(valid_ll) / valid.shape[:2].numel(),
            "loss_including_penalty": float(loss), **gradient_summary(model),
        }
    pmf, hazard = duration_law(model)
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
    payload = {
        "schema": "trading-gradient-flow-v1", "smoke": args.smoke,
        "dataset_sha256": DATA_SHA, "artifact_sha256": MODEL_SHA,
        "protocol_sha256": digest(args.protocol), "runner_sha256": digest(__file__),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "train_shape": list(train.shape), "valid_shape": list(valid.shape),
        "oracle": False, "test_rows_extracted": 0, "fitting_updates": 0,
        "state_dict_unchanged": True, "config": cfg.to_dict(),
        "duration_mean": [duration_mean(p, h) for p, h in zip(pmf, hazard)],
        "tail_hazard": hazard.tolist(), "passes": passes,
        "limits": [
            "Frozen artifact has no optimizer state or historical update trace.",
            "Raw gradient norm fractions do not equal Adam parameter update fractions.",
            "Nonzero local gradients do not establish convergence or statistical recovery.",
            "Scheduled-temperature scores and target-temperature scores are different objectives.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(payload, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "artifact", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
