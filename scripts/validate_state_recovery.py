from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from nhsmm import ModelConfig, NHSMM
from nhsmm.filtering import filter_model_sequence

N_STATES = 3
N_FEATURES = 3
MAX_DURATION = 25
STEPS = 180
N_TRAIN = 8
N_EVAL = 5
DEFAULT_SEEDS = tuple(range(301, 316))


@dataclass(frozen=True)
class Scenario:
    name: str
    separation: float
    min_median_accuracy: float | None = None
    min_median_ari: float | None = None
    max_median_accuracy: float | None = None
    max_abs_median_ari: float | None = None
    min_noncollapsed: int = 14


SCENARIOS = {
    "strong": Scenario("strong", 4.0, min_median_accuracy=0.90, min_median_ari=0.80),
    "moderate": Scenario("moderate", 2.0, min_median_accuracy=0.70, min_median_ari=0.40),
    "null": Scenario(
        "null",
        0.0,
        max_median_accuracy=0.45,
        max_abs_median_ari=0.10,
        min_noncollapsed=0,
    ),
}


def _generate(seed: int, separation: float, n_sequences: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    observations = np.zeros((n_sequences, STEPS, N_FEATURES), dtype=np.float32)
    states = np.zeros((n_sequences, STEPS), dtype=np.int64)
    means = np.eye(N_STATES, dtype=np.float32) * float(separation)
    for batch in range(n_sequences):
        t = 0
        state = int(rng.integers(N_STATES))
        while t < STEPS:
            duration = int(np.clip(rng.poisson(9) + 1, 2, MAX_DURATION))
            end = min(STEPS, t + duration)
            observations[batch, t:end] = means[state] + rng.normal(
                0.0, 1.0, (end - t, N_FEATURES)
            ).astype(np.float32)
            states[batch, t:end] = state
            candidates = [candidate for candidate in range(N_STATES) if candidate != state]
            state = int(rng.choice(candidates))
            t = end
    return observations, states


def _adjusted_rand_index(true: np.ndarray, predicted: np.ndarray) -> float:
    contingency = np.zeros((N_STATES, N_STATES), dtype=np.int64)
    for expected, actual in zip(true, predicted):
        contingency[expected, actual] += 1

    def comb2(value: int) -> int:
        return value * (value - 1) // 2

    n = len(true)
    total = comb2(n)
    if total == 0:
        return 0.0
    sum_cells = sum(comb2(int(value)) for value in contingency.ravel())
    sum_rows = sum(comb2(int(value)) for value in contingency.sum(axis=1))
    sum_cols = sum(comb2(int(value)) for value in contingency.sum(axis=0))
    expected = sum_rows * sum_cols / total
    upper = 0.5 * (sum_rows + sum_cols)
    denominator = upper - expected
    return float((sum_cells - expected) / denominator) if abs(denominator) > 1e-12 else 0.0


def _matched_accuracy(true: np.ndarray, predicted: np.ndarray) -> float:
    best = 0.0
    for permutation in itertools.permutations(range(N_STATES)):
        mapped = np.fromiter(
            (permutation[state] for state in predicted), dtype=np.int64, count=len(predicted)
        )
        best = max(best, float(np.mean(mapped == true)))
    return best


def _run_one(seed: int, scenario: Scenario) -> dict[str, object]:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    train_x, _ = _generate(seed, scenario.separation, N_TRAIN)
    config = ModelConfig(
        n_states=N_STATES,
        n_features=N_FEATURES,
        max_duration=MAX_DURATION,
        causal=True,
        seed=seed,
        n_init=1,
        max_iter=40,
        use_scheduler=False,
        convergence_stop=False,
        verbose=False,
        dropout=0.0,
        emission_init_mode="kmeans",
    )
    model = NHSMM(config, device="cpu")
    model.initialize_distributions()
    model.optimize(torch.tensor(train_x))
    model.eval()

    eval_x, eval_y = _generate(seed + 10_000, scenario.separation, N_EVAL)
    with torch.inference_mode():
        eval_tensor = torch.tensor(eval_x)
        trace = filter_model_sequence(model, eval_tensor)
        posterior = trace.state_posterior.cpu().numpy()
        predicted = posterior.argmax(axis=-1).reshape(-1)
        expected = eval_y.reshape(-1)
        occupancy = posterior.reshape(-1, N_STATES).mean(axis=0)
        occupancy /= occupancy.sum()
        clipped = np.clip(occupancy, 1e-12, 1.0)
        effective_states = float(np.exp(-np.sum(clipped * np.log(clipped))))
        max_occupancy = float(occupancy.max())
        log_likelihood = float(model.log_likelihood(eval_tensor, reduce=True)) / (N_EVAL * STEPS)

    return {
        "seed": seed,
        "scenario": scenario.name,
        "matched_accuracy": _matched_accuracy(expected, predicted),
        "ari": _adjusted_rand_index(expected, predicted),
        "effective_states": effective_states,
        "max_occupancy": max_occupancy,
        "noncollapsed": effective_states >= 2.0 and max_occupancy <= 0.80,
        "ll_per_row": log_likelihood,
        "occupancy": occupancy.tolist(),
    }


def _summarize(scenario: Scenario, rows: list[dict[str, object]]) -> dict[str, object]:
    accuracy = np.asarray([row["matched_accuracy"] for row in rows], dtype=float)
    ari = np.asarray([row["ari"] for row in rows], dtype=float)
    effective = np.asarray([row["effective_states"] for row in rows], dtype=float)
    noncollapsed = sum(bool(row["noncollapsed"]) for row in rows)
    median_accuracy = float(np.median(accuracy))
    median_ari = float(np.median(ari))

    reasons: list[str] = []
    if scenario.min_median_accuracy is not None and median_accuracy < scenario.min_median_accuracy:
        reasons.append("median_accuracy")
    if scenario.min_median_ari is not None and median_ari < scenario.min_median_ari:
        reasons.append("median_ari")
    if scenario.max_median_accuracy is not None and median_accuracy > scenario.max_median_accuracy:
        reasons.append("median_accuracy_null")
    if scenario.max_abs_median_ari is not None and abs(median_ari) > scenario.max_abs_median_ari:
        reasons.append("median_ari_null")
    if noncollapsed < scenario.min_noncollapsed:
        reasons.append("noncollapsed")

    return {
        "scenario": scenario.name,
        "separation": scenario.separation,
        "n": len(rows),
        "median_accuracy": median_accuracy,
        "median_ari": median_ari,
        "median_effective_states": float(np.median(effective)),
        "noncollapsed": noncollapsed,
        "passed": not reasons,
        "failure_reasons": reasons,
    }


def _parse_seeds(raw: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate permutation-invariant latent-state recovery."
    )
    parser.add_argument("--scenario", choices=["all", *SCENARIOS], default="all")
    parser.add_argument("--seeds", type=_parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")

    selected: Iterable[Scenario] = (
        SCENARIOS.values() if args.scenario == "all" else (SCENARIOS[args.scenario],)
    )
    payload: dict[str, object] = {
        "schema": "nhsmm-state-recovery-v1",
        "emission_init_mode": "kmeans",
        "seeds": list(args.seeds),
        "scenarios": {},
    }
    all_passed = True
    for scenario in selected:
        jobs = [(seed, scenario) for seed in args.seeds]
        if args.workers == 1:
            rows = [_run_one(*job) for job in jobs]
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
                rows = list(executor.map(_worker, jobs))
        summary = _summarize(scenario, rows)
        payload["scenarios"][scenario.name] = {"summary": summary, "runs": rows}
        all_passed = all_passed and bool(summary["passed"])
        print(json.dumps(summary, sort_keys=True), flush=True)

    payload["passed"] = all_passed
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0 if all_passed else 1


def _worker(job: tuple[int, Scenario]) -> dict[str, object]:
    return _run_one(*job)


if __name__ == "__main__":
    raise SystemExit(main())
