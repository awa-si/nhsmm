"""Bounded all-five-parameter EM execution and public verification smoke."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from scripts.joint_gaussian_duration import initialize_from_observations
from scripts.compare_joint_gaussian_duration import SOURCES as PRIOR_SOURCES
from scripts.joint_static_chain import MASK, full_mstep, objective, statistics
from scripts.static_chain_check import numerical_check
from scripts.verify_isolated_trading_hsmm import sequences


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("preserve smoke evidence: output exists")
    torch.set_num_threads(1)
    x, _ = sequences(401937, count=16, steps=300, sigma=2.0)
    means, variances = initialize_from_observations(x)
    pmf, transition, initial = np.full((4, 260), 1 / 260), MASK / 3, np.full(4, 0.25)
    history = []
    start = time.perf_counter()
    for update in range(4):
        s = statistics(x, pmf, means, variances, transition, initial)
        value = objective(s["ll"], pmf, transition, initial)
        if history and value < history[-1]["objective"] - 1e-6:
            raise ValueError("augmented objective decreased")
        history.append(
            {
                "completed_updates": update,
                "ll": s["ll"],
                "objective": value,
                "pmf": pmf.tolist(),
                "means": means.tolist(),
                "variances": variances.tolist(),
                "transition": transition.tolist(),
                "initial": initial.tolist(),
            }
        )
        if update < 3:
            pmf, means, variances, transition, initial, certificate = full_mstep(s)
            if certificate["maximum_normalized_anchored_gradient"] > 1e-8:
                raise ValueError("duration solver certificate")
    checks = numerical_check(x, pmf, means, variances, transition, initial)
    if any(v > (1e-8 if "filter" in k else 1e-7) for k, v in checks.items()):
        raise ValueError("public all-parameter numerical smoke failed")
    sources = (
        *PRIOR_SOURCES,
        "scripts/validate_trading_regime_usefulness.py",
        "scripts/joint_static_chain.py",
        "scripts/static_chain_check.py",
        "scripts/smoke_joint_static_chain.py",
        "docs/validation/joint-static-chain-protocol.json",
    )
    result = {
        "schema": "nhsmm-joint-static-chain-smoke-v1",
        "source_commit": a.source_commit,
        "source_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "training_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
        "seed": 401937,
        "dimensions": {"sequences": 16, "steps": 300},
        "actual_full_em_updates": 3,
        "history": history,
        "numerical_check": checks,
        "seconds": time.perf_counter() - start,
        "oracle": False,
        "outer_evaluation": False,
        "statistical_acceptance": False,
        "limits": "execution only: scalar Gaussian, fixed duration bins, static learned transition/initial; not neural/Trading recovery",
    }
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"smoke_pass": True, "seconds": result["seconds"], "numerical_check": checks}))


if __name__ == "__main__":
    main()
