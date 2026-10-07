from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from nhsmm import ModelConfig, NHSMM
from nhsmm.runtime import HSMMFilterRuntime

DEFAULT_SEEDS = tuple(range(321, 329))
MAX_DURATION = 8
HORIZONS = (1, 3, 5, 8)
SCENARIOS = ("true_tail", "finite_control")


def _generate(seed: int, *, n_sequences: int, steps: int, scenario: str):
    rng = np.random.default_rng(seed)
    sequences: list[np.ndarray] = []
    remaining_to_end: list[np.ndarray] = []
    durations: list[int] = []

    for sequence_index in range(n_sequences):
        x = np.zeros((steps, 2), dtype=np.float32)
        remaining = np.zeros(steps, dtype=np.int32)
        t = 0
        state = sequence_index % 2

        while t < steps:
            if scenario == "true_tail":
                if rng.random() < 0.30:
                    duration = int(rng.integers(2, MAX_DURATION))
                else:
                    duration = MAX_DURATION + int(rng.geometric(0.14) - 1)
            elif scenario == "finite_control":
                duration = int(rng.integers(2, MAX_DURATION + 1))
            else:
                raise ValueError(f"unknown scenario {scenario!r}")

            durations.append(duration)
            end = min(steps, t + duration)
            mean = -1.0 if state == 0 else 1.0
            x[t:end, 0] = mean + rng.normal(0.0, 0.55, end - t)
            x[t:end, 1] = rng.normal(0.0, 0.65, end - t)
            for index in range(t, end):
                remaining[index] = end - 1 - index
            t = end
            state = 1 - state

        sequences.append(x)
        remaining_to_end.append(remaining)

    return np.stack(sequences), remaining_to_end, np.asarray(durations)


def _model_config(seed: int, *, duration_tail: bool) -> ModelConfig:
    return ModelConfig(
        n_states=2,
        n_features=2,
        max_duration=MAX_DURATION,
        causal=True,
        duration_tail=duration_tail,
        duration_tail_init_probability=0.5,
        use_context_encoder=False,
        dropout=0.0,
        emission_init_mode="kmeans",
        duration_init_mode="uniform",
        transition_init_mode="uniform",
        initial_init_mode="uniform",
        n_init=1,
        max_iter=80,
        lr=8e-3,
        use_scheduler=False,
        convergence_stop=False,
        seed=seed,
        verbose=False,
    )


def _train(seed: int, *, duration_tail: bool, x: np.ndarray) -> NHSMM:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    model = NHSMM(_model_config(seed, duration_tail=duration_tail), device="cpu")
    model.initialize_distributions()
    model.optimize(torch.as_tensor(x, dtype=torch.float32))
    model.eval()
    return model


def _survival_calibration(
    model: NHSMM,
    x: np.ndarray,
    remaining_to_end: list[np.ndarray],
) -> dict[str, float]:
    predictions = {horizon: [] for horizon in HORIZONS}
    outcomes = {horizon: [] for horizon in HORIZONS}

    for sequence, remaining in zip(x, remaining_to_end):
        runtime = HSMMFilterRuntime(model)
        for t, row in enumerate(sequence):
            runtime.step(torch.as_tensor(row, dtype=torch.float32))
            if t < 10 or t >= len(sequence) - max(HORIZONS) - 1:
                continue

            forecast = runtime.forecast_survival(HORIZONS)
            values = forecast.end_within_probability[0].detach().cpu().numpy()
            for index, horizon in enumerate(HORIZONS):
                predictions[horizon].append(float(values[index]))
                outcomes[horizon].append(float(remaining[t] < horizon))

    mae = []
    brier = []
    for horizon in HORIZONS:
        predicted = np.asarray(predictions[horizon], dtype=np.float64)
        observed = np.asarray(outcomes[horizon], dtype=np.float64)
        mae.append(float(np.mean(np.abs(predicted - observed))))
        brier.append(float(np.mean((predicted - observed) ** 2)))

    return {
        "survival_mae": float(np.mean(mae)),
        "survival_brier": float(np.mean(brier)),
    }


def _run_seed(seed: int, *, scenario: str) -> dict[str, object]:
    torch.set_num_threads(1)
    train_x, _, _ = _generate(seed, n_sequences=12, steps=300, scenario=scenario)
    test_x, remaining, durations = _generate(
        seed + 10000,
        n_sequences=8,
        steps=300,
        scenario=scenario,
    )
    test_tensor = torch.as_tensor(test_x, dtype=torch.float32)

    result: dict[str, object] = {
        "seed": seed,
        "scenario": scenario,
        "true_tail_fraction": float(np.mean(durations > MAX_DURATION)),
        "true_mean_duration": float(np.mean(durations)),
    }

    for duration_tail in (False, True):
        model = _train(seed, duration_tail=duration_tail, x=train_x)
        with torch.inference_mode():
            ll_per_row = float(model.log_likelihood(test_tensor, reduce=True)) / float(
                test_x.shape[0] * test_x.shape[1]
            )
        calibration = _survival_calibration(model, test_x, remaining)
        prefix = "tail" if duration_tail else "finite"
        result[f"{prefix}_ll_per_row"] = ll_per_row
        result[f"{prefix}_survival_mae"] = calibration["survival_mae"]
        result[f"{prefix}_survival_brier"] = calibration["survival_brier"]
        if duration_tail:
            result["learned_h_tail"] = [
                float(value) for value in model.dist.duration.tail_probability().detach().cpu()
            ]

    result["ll_gain_tail_minus_finite"] = float(
        result["tail_ll_per_row"] - result["finite_ll_per_row"]
    )
    result["mae_gain_finite_minus_tail"] = float(
        result["finite_survival_mae"] - result["tail_survival_mae"]
    )
    result["brier_gain_finite_minus_tail"] = float(
        result["finite_survival_brier"] - result["tail_survival_brier"]
    )
    return result


def _run_seed_payload(payload: tuple[int, str]) -> dict[str, object]:
    seed, scenario = payload
    return _run_seed(seed, scenario=scenario)


def _summary(rows: list[dict[str, object]]) -> dict[str, float | int]:
    ll_gain = np.asarray([row["ll_gain_tail_minus_finite"] for row in rows], dtype=np.float64)
    mae_gain = np.asarray([row["mae_gain_finite_minus_tail"] for row in rows], dtype=np.float64)
    brier_gain = np.asarray([row["brier_gain_finite_minus_tail"] for row in rows], dtype=np.float64)
    tail_hazard = np.asarray(
        [value for row in rows for value in row["learned_h_tail"]],
        dtype=np.float64,
    )

    return {
        "n": len(rows),
        "positive_ll_gain": int(np.sum(ll_gain > 0.0)),
        "positive_mae_gain": int(np.sum(mae_gain > 0.0)),
        "positive_brier_gain": int(np.sum(brier_gain > 0.0)),
        "median_ll_gain": float(np.median(ll_gain)),
        "median_mae_gain": float(np.median(mae_gain)),
        "median_brier_gain": float(np.median(brier_gain)),
        "median_learned_h_tail": float(np.median(tail_hazard)),
        "min_learned_h_tail": float(np.min(tail_hazard)),
        "mean_true_tail_fraction": float(np.mean([row["true_tail_fraction"] for row in rows])),
        "mean_true_duration": float(np.mean([row["true_mean_duration"] for row in rows])),
    }


def _acceptance(
    true_tail: dict[str, float | int],
    finite_control: dict[str, float | int],
) -> dict[str, bool]:
    checks = {
        "true_tail_calibration": bool(
            true_tail["positive_brier_gain"] >= 7
            and true_tail["median_brier_gain"] >= 0.03
            and true_tail["positive_mae_gain"] >= 7
            and true_tail["median_mae_gain"] >= 0.01
        ),
        "true_tail_likelihood": bool(
            true_tail["positive_ll_gain"] >= 5 and true_tail["median_ll_gain"] >= 0.0
        ),
        "true_tail_hazard_activates": bool(true_tail["median_learned_h_tail"] <= 0.50),
        "finite_control_reduces_to_finite": bool(
            finite_control["min_learned_h_tail"] >= 0.99
            and abs(finite_control["median_ll_gain"]) <= 0.001
            and abs(finite_control["median_mae_gain"]) <= 0.001
            and abs(finite_control["median_brier_gain"]) <= 0.001
        ),
    }
    checks["passed"] = all(checks.values())
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate bounded D+ duration-tail usefulness and finite-support fallback."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Independent seed workers (default: min(4, CPU count)).",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")

    scenarios = []
    for scenario in SCENARIOS:
        payloads = [(seed, scenario) for seed in DEFAULT_SEEDS]
        if args.workers == 1:
            rows = [_run_seed_payload(payload) for payload in payloads]
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
                rows = list(executor.map(_run_seed_payload, payloads))

        for row in rows:
            print(json.dumps(row, sort_keys=True), flush=True)
        scenarios.append(
            {
                "scenario": scenario,
                "rows": rows,
                "summary": _summary(rows),
            }
        )

    by_name = {item["scenario"]: item["summary"] for item in scenarios}
    acceptance = _acceptance(by_name["true_tail"], by_name["finite_control"])
    payload = {
        "schema": "nhsmm-duration-tail-acceptance-v1",
        "max_duration": MAX_DURATION,
        "horizons": list(HORIZONS),
        "seeds": list(DEFAULT_SEEDS),
        "scenarios": scenarios,
        "acceptance": acceptance,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(
        json.dumps(
            {"acceptance": acceptance, "summaries": by_name},
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if acceptance["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
