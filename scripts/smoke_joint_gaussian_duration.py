"""Bounded actual joint EM smoke; no outer-data/statistical acceptance."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from scripts.block_mass_smoothing import marginal_objective
from scripts.joint_gaussian_duration import initialize_from_observations, joint_mstep, statistics
from scripts.verify_isolated_trading_hsmm import sequences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing smoke evidence overwrite")
    x, _ = sequences(401928, count=16, steps=300, sigma=2.0)
    means, variances = initialize_from_observations(x)
    p = np.full((4, 260), 1 / 260)
    history = []
    start = time.perf_counter()
    for update in range(4):
        s = statistics(x, p, means, variances, chunk=16)
        objective = marginal_objective(s["ll"], p, 0.0)
        if history and objective < history[-1]["objective"] - 1e-6:
            raise ValueError("joint augmented marginal objective decreased")
        history.append(
            {
                "completed_updates": update,
                "ll": s["ll"],
                "objective": objective,
                "means": means.tolist(),
                "variances": variances.tolist(),
            }
        )
        if update < 3:
            p, means, variances, _ = joint_mstep(s)
    result = {
        "schema": "nhsmm-joint-gaussian-duration-smoke-v1",
        "source_commit": args.source_commit,
        "source_sha256": {
            path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in (
                "scripts/joint_gaussian_duration.py",
                "scripts/smoke_joint_gaussian_duration.py",
            )
        },
        "training_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
        "seed": 401928,
        "sequences": 16,
        "steps": 300,
        "actual_joint_em_updates": 3,
        "history": history,
        "seconds": time.perf_counter() - start,
        "monotonic_objective": True,
        "training_observations_only_initialization": True,
        "oracle": False,
        "outer_evaluation": False,
        "limits": "research static transition/initial, scalar Gaussian/fixed-bin durations; execution smoke, not recovery/Trading/neural acceptance",
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
