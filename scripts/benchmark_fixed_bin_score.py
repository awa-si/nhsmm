"""Matched frozen-nuisance score/Adam timing, no recovery/model selection."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from scripts.compare_block_mass_smoothing import SOURCES
from scripts.compare_unlabelled_duration_em import aggregate, emission_scores
from scripts.fixed_bin_posterior_score import density, score_gradient
from scripts.verify_isolated_trading_hsmm import make_model, sequences


def benchmark():
    torch.set_num_threads(1)
    x, _ = sequences(401921, count=4, steps=300, sigma=2.0)
    observations = torch.tensor(x)
    trials = []
    for _ in range(3):
        native = torch.nn.Parameter(torch.zeros(4, 32, dtype=torch.float64))
        analytic = torch.nn.Parameter(native.detach().clone())
        p = density(native)
        model = make_model(p.detach().numpy(), sigma=2.0)
        del model.dist.duration.logits
        model.dist.duration.logits = p.log()
        native_optimizer = torch.optim.Adam([native], lr=0.01)
        analytic_optimizer = torch.optim.Adam([analytic], lr=0.01)
        start = time.perf_counter()
        objective = model.log_likelihood(observations, reduce=True) + 1e-6 * p.log().sum()
        (-objective).backward()
        reference_gradient = native.grad.clone()
        torch.nn.utils.clip_grad_norm_([native], 5.0)
        native_optimizer.step()
        native_seconds = time.perf_counter() - start
        start = time.perf_counter()
        initial = density(analytic).detach().numpy()
        ll, counts = aggregate(emission_scores(x), initial)
        analytic_objective = ll + 1e-6 * np.log(initial).sum()
        analytic.grad = -score_gradient(analytic, counts)
        gradient_error = float((analytic.grad - reference_gradient).abs().max())
        torch.nn.utils.clip_grad_norm_([analytic], 5.0)
        analytic_optimizer.step()
        analytic_seconds = time.perf_counter() - start
        objective_error = abs(float(objective.detach()) - analytic_objective)
        step_error = float((native.detach() - analytic.detach()).abs().max())
        if objective_error > 1e-7 or gradient_error > 1e-7 or step_error > 1e-10:
            raise ValueError("matched public score/Adam equivalence failed")
        trials.append(
            {
                "public_autograd_seconds": native_seconds,
                "analytic_score_seconds": analytic_seconds,
                "objective_error": objective_error,
                "gradient_error": gradient_error,
                "adam_step_error": step_error,
            }
        )
    paths = tuple(
        dict.fromkeys(
            (
                *SOURCES,
                "scripts/fixed_bin_posterior_score.py",
                "scripts/benchmark_fixed_bin_score.py",
            )
        )
    )
    return {
        "schema": "nhsmm-matched-fixed-bin-score-benchmark-v1",
        "seed": 401921,
        "sequences": 4,
        "steps": 300,
        "sigma": 2.0,
        "repetitions": 3,
        "same_data_initial_density_prior_clipping_adam": True,
        "gradient_target": "fixed width8 density, last bin anchored0, temperature1, prior1e-6 per age, no smoothing",
        "source_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "training_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
        "trials": trials,
        "median_public_autograd_seconds": float(
            np.median([t["public_autograd_seconds"] for t in trials])
        ),
        "median_analytic_score_seconds": float(
            np.median([t["analytic_score_seconds"] for t in trials])
        ),
        "oracle": False,
        "outer_evaluation": False,
        "limits": "local matched score/backward/clipped Adam timing; setup excluded; only known frozen nuisance and fixed density bins; not production/joint-neural acceleration or statistical recovery",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing benchmark overwrite")
    result = benchmark()
    result["source_commit"] = args.source_commit
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
