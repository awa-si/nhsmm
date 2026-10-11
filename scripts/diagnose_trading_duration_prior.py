"""Controlled duration-prior x iteration pilot; all interventions stay in this script."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch

from nhsmm import NHSMM
from scripts.diagnose_trading_decoder_duration import diagnose
from scripts.validate_trading_regime_usefulness import RunBudget, _config

SEEDS = (901, 902, 903)
ARMS = (("uniform", 4), ("uniform", 12), ("geometric128", 4), ("geometric128", 12))


def geometric_duration_mass(max_duration, mean):
    """Finite masses 1..D-1 plus the full remaining geometric D+ survival mass."""
    if max_duration < 1 or not np.isfinite(mean) or mean <= 1:
        raise ValueError("D must be positive and geometric mean must exceed one")
    hazard = 1.0 / mean
    p = hazard * (1 - hazard) ** np.arange(max_duration, dtype=float)
    p[-1] = (1 - hazard) ** (max_duration - 1)
    return p, hazard


class DurationPilotNHSMM(NHSMM):
    def _initialize_run_state(self, *args, **kwargs):
        super()._initialize_run_state(*args, **kwargs)
        if self.duration_pilot_init == "geometric128":
            p, h = geometric_duration_mass(self.config.max_duration, 128.0)
            with torch.no_grad():
                # Restart distributions exist now. Bias is constant and cancels in normalization.
                self.dist.duration.logits.copy_(
                    torch.as_tensor(
                        np.log(p), dtype=self.dist.duration.logits.dtype, device=self.device
                    ).expand_as(self.dist.duration.logits)
                )
                self.dist.duration.tail_end_probability.fill_(h)


def fit(seed, x, c, *, duration_init, **kwargs):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    model = DurationPilotNHSMM(_config(seed, **kwargs), device="cpu")
    model.duration_pilot_init = duration_init
    model.initialize_distributions()
    model.optimize(torch.tensor(x), context=torch.tensor(c))
    model.eval()
    return model


def run_seed(job):
    seed, path, directory = job
    rows = []
    for duration_init, iterations in ARMS:
        budget = RunBudget(
            window=240,
            train_windows=8,
            eval_windows=4,
            max_iter=iterations,
            transition_refine_steps=2,
        )
        row = diagnose(seed, path, budget, fit=partial(fit, duration_init=duration_init))
        row.update(duration_init=duration_init, max_iter=iterations)
        rows.append(row)
        # Preserve completed arms even if a later fit hits the outer runner time limit.
        target = directory / f"duration-prior-{seed}-{duration_init}-{iterations}.json"
        target.write_text(json.dumps(row, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "seed": seed,
                    "arm": [duration_init, iterations],
                    "filter": row["filter"],
                    "test_ll": row["test_ll_per_row"],
                }
            ),
            flush=True,
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.dataset.with_name("manifest.json").read_text())
    if manifest.get("transition_context", {}).get("conditioning_time") != "episode_end":
        parser.error("corrected episode_end data required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
        rows = [
            r
            for group in pool.map(run_seed, [(s, args.dataset, args.output.parent) for s in SEEDS])
            for r in group
        ]
    summary = {}
    for mode, iterations in ARMS:
        group = [r for r in rows if r["duration_init"] == mode and r["max_iter"] == iterations]
        values = {
            decoder: {
                metric: float(np.median([r[decoder][metric] for r in group]))
                for metric in (
                    "accuracy",
                    "ari",
                    "boundary_f1_tol5",
                    "run_duration_rel_error",
                    "switches",
                )
            }
            for decoder in ("filter", "viterbi")
        }
        for metric in ("train_ll_per_row", "test_ll_per_row"):
            values[metric] = float(np.median([r[metric] for r in group]))
        summary[f"{mode}-{iterations}"] = values
    payload = {
        "schema": "nhsmm-trading-duration-prior-iteration-pilot-v1",
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "dataset_schema": manifest["schema"],
        "seeds": list(SEEDS),
        "window": 240,
        "train_windows": 8,
        "eval_windows": 4,
        "transition_refine_steps": 2,
        "emission_init": "kmeans_cluster_variance",
        "arms": list(ARMS),
        "full_protocol": False,
        "runs": rows,
        "summary": summary,
        "contract": {
            "geometric128": "predeclared common mean 128; not estimated from labels",
            "iteration_axis": "existing annealing schedule also changes with max_iter",
            "training": "chronological first 60%, labels excluded",
            "evaluation": "fixed last 20%; no model selection",
            "viterbi": "retrospective diagnostic, not live causal output",
            "package_defaults": "unchanged; duration intervention is script-local",
        },
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
