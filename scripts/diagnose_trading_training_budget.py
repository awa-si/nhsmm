"""Matched training-budget pilot; truth labels enter scoring only."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.diagnose_trading_decoder_duration import diagnose
from scripts.validate_trading_regime_usefulness import RunBudget

ITERATIONS = (4, 12, 30)
SEEDS = (901, 902, 903)


def run(job):
    seed, dataset, iterations = job
    rows = []
    for count in iterations:
        budget = RunBudget(
            window=240,
            train_windows=8,
            eval_windows=4,
            max_iter=count,
            transition_refine_steps=2,
        )
        row = diagnose(seed, dataset, budget)
        row["max_iter"] = count
        rows.append(row)
    return {"seed": seed, "arms": rows}


def summarize(rows, iterations):
    result = {}
    for count in iterations:
        group = [a for row in rows for a in row["arms"] if a["max_iter"] == count]
        summary = {
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
        summary.update(
            {
                key: float(np.median([r[key] for r in group]))
                for key in ("train_ll_per_row", "test_ll_per_row", "decoder_agreement")
            }
        )
        summary["all_states_covered"] = all(not r["filter"]["missing_states"] for r in group)
        result[str(count)] = summary
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-iters", default="4,12,30")
    parser.add_argument("--seeds", default="901,902,903")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    try:
        iterations = tuple(int(x) for x in args.max_iters.split(","))
        seeds = tuple(int(x) for x in args.seeds.split(","))
        if not iterations or min(iterations) < 1 or len(set(iterations)) != len(iterations):
            raise ValueError("positive distinct training budgets required")
        if not seeds or len(set(seeds)) != len(seeds) or args.workers < 1:
            raise ValueError("distinct seeds and positive workers required")
    except ValueError as exc:
        parser.error(str(exc))
    manifest = json.loads(args.dataset.with_name("manifest.json").read_text())
    if manifest.get("transition_context", {}).get("conditioning_time") != "episode_end":
        parser.error("corrected episode_end data required")
    jobs = [(seed, args.dataset, iterations) for seed in seeds]
    if args.workers == 1:
        rows = [run(job) for job in jobs]
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(args.workers, len(seeds))
        ) as pool:
            rows = list(pool.map(run, jobs))
    payload = {
        "schema": "nhsmm-trading-training-budget-pilot-v1",
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "seeds": list(seeds),
        "joint_iterations_predeclared": list(iterations),
        "window": 240,
        "train_windows": 8,
        "eval_windows": 4,
        "transition_refine_steps": 2,
        "full_protocol": False,
        "runs": rows,
        "summary": summarize(rows, iterations),
        "contract": {
            "changed": "joint training max_iter only; same seed reinitialized independently for each budget",
            "fixed": "full eight features, training-only standardization, variance-aware initializer, context, windows, optimizer, transition refinement",
            "labels": "scoring only; no training, stopping, budget selection or deployment selection",
            "filter": "causal marginal argmax; operational diagnostic",
            "viterbi": "retrospective full-window MAP; unavailable for live decisions",
            "duration": "state-run error differs from learned episode PMF",
            "protocol": "short training-data coverage; even 30 iterations is not full recovery acceptance",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"]))


if __name__ == "__main__":
    main()
