from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from nhsmm import ModelConfig, NHSMM
from nhsmm.filtering import filter_model_sequence
from nhsmm.runtime import HSMMFilterRuntime

DEFAULT_SEEDS = tuple(range(201, 216))
SCENARIOS = (("strong", 6, 18), ("moderate", 9, 15), ("null", 12, 12))


def _generate(seed: int, *, n_sequences: int, steps: int, short_mean: int, long_mean: int):
    rng = np.random.default_rng(seed)
    sequences, contexts = [], []
    for sequence_index in range(n_sequences):
        x = np.zeros((steps, 3), dtype=np.float32)
        context = np.zeros(steps, dtype=np.float32)
        t = 0
        state = sequence_index % 2
        while t < steps:
            c = float(rng.choice((-1.0, 1.0)))
            mean_duration = short_mean if c < 0 else long_mean
            duration = int(np.clip(rng.poisson(mean_duration) + 1, 2, 30))
            end = min(steps, t + duration)
            x[t:end, 0] = c + rng.normal(0.0, 0.12, end - t)
            x[t:end, 1] = (-1.5 if state == 0 else 1.5) + rng.normal(0.0, 0.18, end - t)
            x[t:end, 2] = rng.normal(0.0, 0.35, end - t)
            context[t:end] = c
            t = end
            state = 1 - state
        sequences.append(x)
        contexts.append(context)
    return np.stack(sequences), contexts


def _train(seed: int, *, short_mean: int, long_mean: int) -> NHSMM:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    x, _ = _generate(seed, n_sequences=8, steps=180, short_mean=short_mean, long_mean=long_mean)
    cfg = ModelConfig(
        n_states=2,
        n_features=3,
        max_duration=30,
        causal=True,
        seed=seed,
        n_init=1,
        max_iter=40,
        use_scheduler=False,
        convergence_stop=False,
        verbose=False,
    )
    model = NHSMM(cfg, device="cpu")
    model.initialize_distributions()
    model.optimize(torch.as_tensor(x, dtype=torch.float32))
    model.eval()
    return model


def _evaluate(model: NHSMM, x: np.ndarray, contexts: list[np.ndarray]) -> dict[str, float]:
    short, long = [], []
    for sequence, context in zip(x, contexts):
        runtime = HSMMFilterRuntime(model)
        for t, row in enumerate(sequence):
            runtime.step(torch.as_tensor(row, dtype=torch.float32))
            if 20 <= t < len(sequence) - 5:
                value = float(runtime.forecast_survival((5,)).end_within_probability[0, 0])
                (short if context[t] < 0 else long).append(value)
    with torch.inference_mode():
        tensor = torch.as_tensor(x, dtype=torch.float32)
        trace = filter_model_sequence(model, tensor)
        occupancy = trace.state_posterior.cpu().numpy().reshape(-1, model.config.n_states).mean(axis=0)
        occupancy /= occupancy.sum()
        ll_per_row = float(model.log_likelihood(tensor, reduce=True)) / float(x.shape[0] * x.shape[1])
    entropy = -float(np.sum(np.clip(occupancy, 1e-12, 1.0) * np.log(np.clip(occupancy, 1e-12, 1.0))))
    return {
        "gap": float(np.mean(short) - np.mean(long)),
        "effective_states": float(np.exp(entropy)),
        "max_occupancy": float(occupancy.max()),
        "validation_log_likelihood_per_row": ll_per_row,
    }


def _summary(rows: list[dict[str, float]]) -> dict[str, float | int]:
    gaps = np.asarray([row["gap"] for row in rows], dtype=np.float64)
    return {
        "n": len(rows),
        "positive": int(np.sum(gaps > 0.0)),
        "material_ge_0_02": int(np.sum(gaps >= 0.02)),
        "median_gap": float(np.median(gaps)),
        "mean_gap": float(np.mean(gaps)),
        "std_gap": float(np.std(gaps)),
        "noncollapsed": int(sum(row["effective_states"] >= 1.5 and row["max_occupancy"] <= 0.85 for row in rows)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate context-conditioned duration recovery on controlled synthetic data.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    scenarios = []
    for name, short_mean, long_mean in SCENARIOS:
        rows = []
        for seed in DEFAULT_SEEDS:
            model = _train(seed, short_mean=short_mean, long_mean=long_mean)
            x, contexts = _generate(seed + 10000, n_sequences=5, steps=180, short_mean=short_mean, long_mean=long_mean)
            metrics = _evaluate(model, x, contexts)
            rows.append({"seed": seed, **metrics})
            print(json.dumps({"scenario": name, "seed": seed, **metrics}, sort_keys=True), flush=True)
        scenarios.append({"scenario": name, "short_mean": short_mean, "long_mean": long_mean, "rows": rows, "summary": _summary(rows)})

    by_name = {item["scenario"]: item["summary"] for item in scenarios}
    strong = by_name["strong"]
    moderate = by_name["moderate"]
    null = by_name["null"]
    acceptance = {
        "strong_context": bool(strong["positive"] >= 13 and strong["material_ge_0_02"] >= 10 and strong["median_gap"] >= 0.02 and strong["noncollapsed"] >= 13),
        "moderate_context": bool(moderate["positive"] >= 12 and moderate["median_gap"] >= 0.01 and moderate["noncollapsed"] >= 13),
        "null_context_no_spurious_separation": bool(abs(null["median_gap"]) <= 0.015 and abs(null["mean_gap"]) <= 0.015 and null["noncollapsed"] >= 13),
    }
    acceptance["passed"] = all(acceptance.values())
    payload = {
        "schema": "nhsmm-package-synthetic-acceptance-v1",
        "seeds": list(DEFAULT_SEEDS),
        "scenarios": scenarios,
        "acceptance": acceptance,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"acceptance": acceptance, "summaries": by_name}, sort_keys=True), flush=True)
    return 0 if acceptance["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
