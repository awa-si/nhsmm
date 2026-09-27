from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from nhsmm import ModelConfig, NHSMM
from nhsmm.filtering import filter_model_sequence

K = 3
F = 3
D = 24
T = 240
N_TRAIN = 10
N_EVAL = 6
SEEDS = tuple(range(501, 516))
MEANS = np.eye(K, dtype=np.float32) * 4.0


@dataclass(frozen=True)
class Scenario:
    name: str
    short_mean: float
    long_mean: float
    transition_preference: float
    min_acc: float | None = None
    min_ari: float | None = None
    min_duration_gap: float | None = None
    min_duration_direction: float | None = None
    max_transition_mae: float | None = None
    min_transition_corr: float | None = None
    min_transition_direction: float | None = None
    max_null_duration_gap: float | None = None
    max_null_transition_delta: float | None = None
    min_noncollapsed: int = 13


SCENARIOS = {
    "strong": Scenario(
        "strong", 6.0, 16.0, 0.85,
        min_acc=0.90, min_ari=0.80,
        min_duration_gap=2.0, min_duration_direction=2.0 / 3.0,
        max_transition_mae=0.20, min_transition_corr=0.60,
        min_transition_direction=0.75, min_noncollapsed=13,
    ),
    "moderate": Scenario(
        "moderate", 8.0, 14.0, 0.65,
        min_acc=0.80, min_ari=0.60,
        min_duration_gap=1.0, min_duration_direction=2.0 / 3.0,
        max_transition_mae=0.22, min_transition_corr=0.40,
        min_transition_direction=0.65, min_noncollapsed=13,
    ),
    "null": Scenario(
        "null", 11.0, 11.0, 0.50,
        max_null_duration_gap=1.0,
        max_null_transition_delta=0.10,
        min_noncollapsed=13,
    ),
}


def transition_matrices(scenario: Scenario) -> tuple[np.ndarray, np.ndarray]:
    p = scenario.transition_preference
    p0 = np.zeros((K, K), dtype=np.float64)
    p1 = np.zeros((K, K), dtype=np.float64)
    for state in range(K):
        first = (state + 1) % K
        second = (state + 2) % K
        p0[state, first], p0[state, second] = p, 1.0 - p
        p1[state, first], p1[state, second] = 1.0 - p, p
    return p0, p1


def generate(seed: int, scenario: Scenario, n_sequences: int):
    rng = np.random.default_rng(seed)
    p0, p1 = transition_matrices(scenario)
    observations = np.zeros((n_sequences, T, F), dtype=np.float32)
    states = np.zeros((n_sequences, T), dtype=np.int64)
    contexts = np.zeros((n_sequences, T, 1), dtype=np.float32)

    for batch in range(n_sequences):
        t = 0
        state = int(rng.integers(K))
        while t < T:
            context = int(rng.integers(2))
            mean_duration = scenario.long_mean if context else scenario.short_mean
            duration = int(np.clip(rng.poisson(max(1.0, mean_duration - 1.0)) + 1, 2, D))
            end = min(T, t + duration)
            observations[batch, t:end] = MEANS[state] + rng.normal(
                0.0, 0.7, (end - t, F)
            ).astype(np.float32)
            states[batch, t:end] = state
            contexts[batch, t:end, 0] = float(context)
            if end < T:
                state = int(rng.choice(K, p=(p1 if context else p0)[state]))
            t = end
    return observations, states, contexts


def adjusted_rand_index(true: np.ndarray, predicted: np.ndarray) -> float:
    contingency = np.zeros((K, K), dtype=np.int64)
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


def best_permutation(true: np.ndarray, predicted: np.ndarray):
    best = (-1.0, None)
    for permutation in itertools.permutations(range(K)):
        mapped = np.fromiter(
            (permutation[state] for state in predicted), dtype=np.int64, count=len(predicted)
        )
        accuracy = float(np.mean(mapped == true))
        if accuracy > best[0]:
            best = (accuracy, permutation)
    return best


def align_matrix(matrix: np.ndarray, permutation: tuple[int, ...]) -> np.ndarray:
    inverse = np.empty(K, dtype=np.int64)
    for learned, true in enumerate(permutation):
        inverse[true] = learned
    return matrix[np.ix_(inverse, inverse)]


def learned_transition_matrices(model: NHSMM) -> tuple[np.ndarray, np.ndarray]:
    matrices = []
    with torch.inference_mode():
        for value in (0.0, 1.0):
            context = torch.tensor([[[value]]], dtype=torch.float32)
            probs = model.dist.transition.expected_probs(context=context).detach().cpu().numpy()
            matrices.append(probs.reshape(-1, K, K)[0].astype(np.float64))
    return matrices[0], matrices[1]


def learned_duration_means(model: NHSMM) -> tuple[np.ndarray, np.ndarray]:
    means = []
    support = np.arange(1, D + 1, dtype=np.float64)
    with torch.inference_mode():
        for value in (0.0, 1.0):
            context = torch.tensor([[[value]]], dtype=torch.float32)
            probs = model.dist.duration.expected_probs(context=context).detach().cpu().numpy()
            pmf = probs.reshape(-1, K, D)[0].astype(np.float64)
            means.append((pmf * support[None, :]).sum(axis=1))
    return means[0], means[1]


def run(seed: int, scenario: Scenario) -> dict[str, object]:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    train_x, _, train_c = generate(seed, scenario, N_TRAIN)
    config = ModelConfig(
        n_states=K,
        n_features=F,
        max_duration=D,
        causal=True,
        context_dim=1,
        seed=seed,
        n_init=1,
        max_iter=40,
        use_scheduler=False,
        convergence_stop=False,
        verbose=False,
        dropout=0.0,
        emission_init_mode="kmeans",
        transition_type="semi",
        transition_context_max_delta=1.5,
        transition_refine_steps=20,
        transition_refine_lr=3e-2,
    )
    model = NHSMM(config, device="cpu")
    model.initialize_distributions()
    model.optimize(torch.tensor(train_x), context=torch.tensor(train_c))
    model.eval()

    eval_x, eval_z, eval_c = generate(seed + 10_000, scenario, N_EVAL)
    eval_tensor = torch.tensor(eval_x)
    context_tensor = torch.tensor(eval_c)
    with torch.inference_mode():
        trace = filter_model_sequence(model, eval_tensor, context=context_tensor)
        posterior = trace.state_posterior.cpu().numpy()
        predicted = posterior.argmax(axis=-1).reshape(-1)
        true = eval_z.reshape(-1)
        accuracy, permutation = best_permutation(true, predicted)
        occupancy = posterior.reshape(-1, K).mean(axis=0)
        occupancy /= occupancy.sum()
        clipped = np.clip(occupancy, 1e-12, 1.0)
        effective_states = float(np.exp(-np.sum(clipped * np.log(clipped))))
        max_occupancy = float(occupancy.max())
        ll_per_row = float(
            model.log_likelihood(eval_tensor, context=context_tensor, reduce=True)
        ) / float(N_EVAL * T)

    learned_p0, learned_p1 = learned_transition_matrices(model)
    learned_p0 = align_matrix(learned_p0, permutation)
    learned_p1 = align_matrix(learned_p1, permutation)
    true_p0, true_p1 = transition_matrices(scenario)
    transition_mae = float(
        (np.abs(learned_p0 - true_p0).mean() + np.abs(learned_p1 - true_p1).mean()) / 2.0
    )
    true_delta = (true_p1 - true_p0).reshape(-1)
    learned_delta = (learned_p1 - learned_p0).reshape(-1)
    active = np.abs(true_delta) > 1e-8
    transition_corr = (
        float(np.corrcoef(true_delta[active], learned_delta[active])[0, 1])
        if active.sum() >= 2 and np.std(learned_delta[active]) > 1e-12
        else 0.0
    )
    transition_direction = (
        float(np.mean(np.sign(true_delta[active]) == np.sign(learned_delta[active])))
        if active.any()
        else None
    )
    null_transition_delta = float(np.abs(learned_delta).mean())

    duration_short, duration_long = learned_duration_means(model)
    duration_gap_by_state = duration_long - duration_short
    duration_gap = float(np.mean(duration_gap_by_state))
    duration_direction = float(np.mean(duration_gap_by_state > 0.0))

    return {
        "seed": seed,
        "scenario": scenario.name,
        "matched_accuracy": accuracy,
        "ari": adjusted_rand_index(true, predicted),
        "effective_states": effective_states,
        "max_occupancy": max_occupancy,
        "noncollapsed": bool(effective_states >= 2.0 and max_occupancy <= 0.80),
        "duration_gap": duration_gap,
        "duration_direction": duration_direction,
        "duration_short_means": duration_short.tolist(),
        "duration_long_means": duration_long.tolist(),
        "transition_mae": transition_mae,
        "transition_delta_correlation": transition_corr,
        "transition_direction_accuracy": transition_direction,
        "null_transition_abs_delta": null_transition_delta,
        "ll_per_row": ll_per_row,
        "true_p0": true_p0.tolist(),
        "true_p1": true_p1.tolist(),
        "learned_p0": learned_p0.tolist(),
        "learned_p1": learned_p1.tolist(),
    }


def summarize(scenario: Scenario, rows: list[dict[str, object]]) -> dict[str, object]:
    def median(key: str) -> float:
        return float(np.median([float(row[key]) for row in rows]))

    noncollapsed = sum(bool(row["noncollapsed"]) for row in rows)
    directions = [
        float(row["transition_direction_accuracy"])
        for row in rows
        if row["transition_direction_accuracy"] is not None
    ]
    values = {
        "median_accuracy": median("matched_accuracy"),
        "median_ari": median("ari"),
        "median_duration_gap": median("duration_gap"),
        "median_duration_direction": median("duration_direction"),
        "median_transition_mae": median("transition_mae"),
        "median_transition_corr": median("transition_delta_correlation"),
        "median_transition_direction": float(np.median(directions)) if directions else None,
        "median_null_transition_abs_delta": median("null_transition_abs_delta"),
        "noncollapsed": noncollapsed,
    }
    reasons: list[str] = []
    if scenario.min_acc is not None and values["median_accuracy"] < scenario.min_acc:
        reasons.append("state_accuracy")
    if scenario.min_ari is not None and values["median_ari"] < scenario.min_ari:
        reasons.append("state_ari")
    if scenario.min_duration_gap is not None and values["median_duration_gap"] < scenario.min_duration_gap:
        reasons.append("duration_gap")
    if scenario.min_duration_direction is not None and values["median_duration_direction"] < scenario.min_duration_direction:
        reasons.append("duration_direction")
    if scenario.max_transition_mae is not None and values["median_transition_mae"] > scenario.max_transition_mae:
        reasons.append("transition_mae")
    if scenario.min_transition_corr is not None and values["median_transition_corr"] < scenario.min_transition_corr:
        reasons.append("transition_correlation")
    if scenario.min_transition_direction is not None and (
        values["median_transition_direction"] is None
        or values["median_transition_direction"] < scenario.min_transition_direction
    ):
        reasons.append("transition_direction")
    if scenario.max_null_duration_gap is not None and abs(values["median_duration_gap"]) > scenario.max_null_duration_gap:
        reasons.append("null_duration_gap")
    if scenario.max_null_transition_delta is not None and values["median_null_transition_abs_delta"] > scenario.max_null_transition_delta:
        reasons.append("null_transition_delta")
    if noncollapsed < scenario.min_noncollapsed:
        reasons.append("noncollapsed")

    return {
        "scenario": scenario.name,
        "n": len(rows),
        **values,
        "passed": not reasons,
        "failure_reasons": reasons,
    }


def parse_seeds(raw: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return values


def worker(job: tuple[int, Scenario]) -> dict[str, object]:
    return run(*job)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate joint state, duration-context, and transition-context identifiability."
    )
    parser.add_argument("--scenario", choices=["all", *SCENARIOS], default="all")
    parser.add_argument("--seeds", type=parse_seeds, default=SEEDS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")

    selected: Iterable[Scenario] = (
        SCENARIOS.values() if args.scenario == "all" else (SCENARIOS[args.scenario],)
    )
    payload: dict[str, object] = {
        "schema": "nhsmm-multicomponent-identifiability-v1",
        "seeds": list(args.seeds),
        "config": {
            "n_states": K,
            "n_features": F,
            "max_duration": D,
            "transition_type": "semi",
            "emission_init_mode": "kmeans",
            "transition_context_max_delta": 1.5,
            "transition_refine_steps": 20,
            "transition_refine_lr": 3e-2,
        },
        "scenarios": {},
    }
    passed = True
    for scenario in selected:
        jobs = [(seed, scenario) for seed in args.seeds]
        if args.workers == 1:
            rows = [worker(job) for job in jobs]
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
                rows = list(executor.map(worker, jobs))
        rows.sort(key=lambda row: int(row["seed"]))
        summary = summarize(scenario, rows)
        payload["scenarios"][scenario.name] = {"summary": summary, "runs": rows}
        passed = passed and bool(summary["passed"])
        print(json.dumps(summary, sort_keys=True), flush=True)

    payload["passed"] = passed
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
