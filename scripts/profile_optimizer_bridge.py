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
    parser.add_argument("--sequences", type=int, default=4)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int, default=401918)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("existing profile output; refusing overwrite")
    if args.sequences < 1 or (args.batch_size is not None and args.batch_size < 1):
        raise ValueError("positive sequence and batch sizes required")
    torch.set_num_threads(1)
    x, _ = sequences(args.seed, count=args.sequences, steps=300, sigma=2.0)
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
        optimizer_batch_size=args.batch_size,
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
        public = float(model._target_training_score(torch.tensor(x), None, 1.0, args.batch_size))
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
        "seed": args.seed,
        "training_sequences": args.sequences,
        "optimizer_batch_size": args.batch_size,
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
