"""Fixed one-step 4x300 resource smoke; no statistical evaluation/selection."""

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch

from scripts.compare_unlabelled_duration_em import aggregate, emission_scores
from scripts.verify_isolated_trading_hsmm import make_model, model_law, sequences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("existing profile output; refusing overwrite")
    torch.set_num_threads(1)
    x, _ = sequences(401918, count=4, steps=300, sigma=2.0)
    model = make_model(np.full((4, 260), 1 / 260), sigma=2.0)
    model.dist.duration.logits.requires_grad_(True)
    frozen = {name: p.clone() for name, p in model.named_parameters() if not p.requires_grad}
    cfg = model.config.with_overrides(
        n_init=1,
        max_iter=1,
        lr=0.01,
        optimizer_temperature_mode="constant",
        optimizer_target_temperature=1.0,
        optimizer_initialization="current",
        use_scheduler=False,
        convergence_stop=False,
        verbose=False,
        loss_bias=0.0,
    )
    start = time.perf_counter()
    model.optimize(torch.tensor(x), cfg=cfg)
    elapsed = time.perf_counter() - start
    if any(
        not torch.equal(frozen[name], p)
        for name, p in model.named_parameters()
        if not p.requires_grad
    ):
        raise ValueError("frozen nuisance changed")
    with torch.no_grad():
        public = float(model.log_likelihood(torch.tensor(x), reduce=True))
    ll, _ = aggregate(emission_scores(x), model_law(model).detach().numpy())
    error = max(abs(public - ll), abs(public - model._restart_scores[0]))
    if error > 1e-7:
        raise ValueError("terminal target/public/reference mismatch")
    payload = {
        "schema": "nhsmm-optimizer-resource-smoke-v1",
        "source_commit": args.source_commit,
        "source_sha256": {
            p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in (
                "nhsmm/config.py",
                "nhsmm/models/base.py",
                "scripts/profile_optimizer_bridge.py",
            )
        },
        "training_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
        "seed": 401918,
        "training_sequences": 4,
        "steps": 300,
        "actual_adam_updates": 1,
        "optimizer_seconds": elapsed,
        "process_peak_rss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "public_reference_target_error": error,
        "frozen_nuisance_preserved": True,
        "oracle": False,
        "outer_evaluation": False,
        "limits": "single-process whole-process peak memory and one-step wall time, not equal compute, statistical recovery or a full-size extrapolation",
    }
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
