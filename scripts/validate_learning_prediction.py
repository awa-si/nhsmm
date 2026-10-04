from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from nhsmm import (
    ModelConfig,
    NHSMM,
    compare_validation_snapshots,
    evaluate_model_health,
    evaluate_validation_snapshot,
)
from nhsmm.filtering import filter_model_sequence

N_STATES = 3
N_FEATURES = 3
MAX_DURATION = 25
STEPS = 180
N_TRAIN = 8
N_EVAL = 5
DEFAULT_SEEDS = tuple(range(401, 411))


@dataclass(frozen=True)
class Scenario:
    name: str
    separation: float
    min_median_oos_accuracy: float | None = None
    min_median_accuracy_gain: float | None = None
    min_median_ll_gain: float | None = None
    max_median_oos_accuracy: float | None = None
    max_abs_median_ari: float | None = None
    max_collapse_fraction: float | None = None
    min_median_boundary_f1: float | None = None
    max_median_run_length_rel_error: float | None = None
    max_median_transition_mae: float | None = None
    max_median_occupancy_l1: float | None = None
    max_median_ece: float | None = None
    expect_collapse: bool = False


SCENARIOS = {
    "strong": Scenario(
        "strong",
        4.0,
        min_median_oos_accuracy=0.95,
        min_median_ll_gain=1.0,
        max_collapse_fraction=0.0,
        min_median_boundary_f1=0.90,
        max_median_run_length_rel_error=0.12,
        max_median_transition_mae=0.10,
        max_median_occupancy_l1=0.10,
        max_median_ece=0.12,
    ),
    "moderate": Scenario(
        "moderate",
        2.0,
        min_median_oos_accuracy=0.90,
        min_median_accuracy_gain=0.02,
        min_median_ll_gain=0.50,
        max_collapse_fraction=0.0,
        min_median_boundary_f1=0.70,
        max_median_run_length_rel_error=0.25,
        max_median_transition_mae=0.18,
        max_median_occupancy_l1=0.16,
        max_median_ece=0.18,
    ),
    "weak": Scenario(
        "weak",
        1.0,
        min_median_oos_accuracy=0.55,
        min_median_ll_gain=0.10,
        max_collapse_fraction=0.40,
        min_median_boundary_f1=None,
        max_median_run_length_rel_error=0.50,
        max_median_transition_mae=0.30,
        max_median_occupancy_l1=0.30,
        max_median_ece=0.25,
    ),
    "null": Scenario(
        "null",
        0.0,
        max_median_oos_accuracy=0.45,
        max_abs_median_ari=0.10,
        expect_collapse=False,
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


def _flatten_predictions(
    decoded: torch.Tensor | list[torch.Tensor],
) -> np.ndarray:
    if isinstance(decoded, list):
        if not decoded:
            return np.empty(0, dtype=np.int64)
        return np.concatenate([item.detach().cpu().numpy().reshape(-1) for item in decoded])
    return decoded.detach().cpu().numpy().reshape(-1)


def _best_state_permutation(
    true: np.ndarray,
    predicted: np.ndarray,
) -> tuple[tuple[int, ...], float]:
    best_permutation = tuple(range(N_STATES))
    best_accuracy = -1.0
    for permutation in itertools.permutations(range(N_STATES)):
        mapped = np.fromiter(
            (permutation[int(state)] for state in predicted),
            dtype=np.int64,
            count=len(predicted),
        )
        accuracy = float(np.mean(mapped == true))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_permutation = permutation
    return best_permutation, best_accuracy


def _matched_accuracy(true: np.ndarray, predicted: np.ndarray) -> float:
    return _best_state_permutation(true, predicted)[1]


def _apply_permutation(values: np.ndarray, permutation: tuple[int, ...]) -> np.ndarray:
    mapping = np.asarray(permutation, dtype=np.int64)
    return mapping[np.asarray(values, dtype=np.int64)]


def _boundary_f1(
    true: np.ndarray,
    predicted: np.ndarray,
    *,
    tolerance: int = 0,
) -> float:
    true = np.asarray(true, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    if true.shape != predicted.shape or true.ndim != 2:
        raise ValueError("boundary F1 expects equal [B,T] arrays")
    if tolerance < 0:
        raise ValueError("boundary tolerance must be >= 0")

    tp = 0
    fp = 0
    fn = 0
    for true_sequence, predicted_sequence in zip(true, predicted):
        true_positions = list(np.flatnonzero(true_sequence[1:] != true_sequence[:-1]) + 1)
        pred_positions = list(np.flatnonzero(predicted_sequence[1:] != predicted_sequence[:-1]) + 1)
        matched_true: set[int] = set()
        matched_pred: set[int] = set()

        candidates: list[tuple[int, int, int]] = []
        for pred_index, pred_position in enumerate(pred_positions):
            for true_index, true_position in enumerate(true_positions):
                distance = abs(int(pred_position) - int(true_position))
                if distance <= tolerance:
                    candidates.append((distance, pred_index, true_index))

        for _, pred_index, true_index in sorted(candidates):
            if pred_index in matched_pred or true_index in matched_true:
                continue
            matched_pred.add(pred_index)
            matched_true.add(true_index)

        tp += len(matched_true)
        fp += len(pred_positions) - len(matched_pred)
        fn += len(true_positions) - len(matched_true)

    denominator = 2 * tp + fp + fn
    return 1.0 if denominator == 0 else float(2 * tp / denominator)


def _run_lengths(states: np.ndarray) -> np.ndarray:
    runs: list[int] = []
    for sequence in np.asarray(states, dtype=np.int64):
        if sequence.size == 0:
            continue
        starts = np.flatnonzero(np.concatenate(([True], sequence[1:] != sequence[:-1])))
        ends = np.concatenate((starts[1:], [sequence.size]))
        runs.extend((ends - starts).tolist())
    return np.asarray(runs, dtype=np.float64)


def _empirical_transition_matrix(states: np.ndarray) -> np.ndarray:
    counts = np.zeros((N_STATES, N_STATES), dtype=np.float64)
    for sequence in np.asarray(states, dtype=np.int64):
        for source, target in zip(sequence[:-1], sequence[1:]):
            if source != target:
                counts[int(source), int(target)] += 1.0
    rows = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, rows, out=np.zeros_like(counts), where=rows > 0)


def _occupancy(states: np.ndarray) -> np.ndarray:
    counts = np.bincount(np.asarray(states, dtype=np.int64).reshape(-1), minlength=N_STATES)
    return counts.astype(np.float64) / counts.sum()


def _expected_calibration_error(
    confidence: np.ndarray,
    correct: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(confidence)
    ece = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        count = int(mask.sum())
        if count:
            ece += count / total * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(ece)


def _adjusted_rand_index(true: np.ndarray, predicted: np.ndarray) -> float:
    contingency = np.zeros((N_STATES, N_STATES), dtype=np.int64)
    for expected, actual in zip(true, predicted):
        contingency[int(expected), int(actual)] += 1

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


def _parameter_l1_change(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> float:
    total = 0.0
    for name, value in before.items():
        current = after[name]
        if value.is_floating_point():
            total += float((current - value).abs().sum().item())
    return total


def _model(seed: int, *, max_iter: int) -> NHSMM:
    config = ModelConfig(
        n_states=N_STATES,
        n_features=N_FEATURES,
        max_duration=MAX_DURATION,
        causal=True,
        seed=seed,
        n_init=1,
        max_iter=max_iter,
        use_scheduler=False,
        convergence_stop=False,
        verbose=False,
        dropout=0.0,
        emission_init_mode="kmeans",
    )
    model = NHSMM(config, device="cpu")
    model.initialize_distributions()
    return model


def _run_one(
    seed: int,
    scenario: Scenario,
    *,
    max_iter: int,
) -> dict[str, object]:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    train_x, train_y = _generate(seed, scenario.separation, N_TRAIN)
    oos_x, oos_y = _generate(seed + 10_000, scenario.separation, N_EVAL)
    train_tensor = torch.tensor(train_x)
    oos_tensor = torch.tensor(oos_x)

    model = _model(seed, max_iter=max_iter)
    model.eval()

    with torch.inference_mode():
        ll_before = float(model.log_likelihood(oos_tensor, reduce=True)) / (N_EVAL * STEPS)
        pred_before = _flatten_predictions(model.predict(oos_tensor, mode="viterbi", verbose=False))
        accuracy_before = _matched_accuracy(oos_y.reshape(-1), pred_before)
        ari_before = _adjusted_rand_index(oos_y.reshape(-1), pred_before)

    state_before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if isinstance(value, torch.Tensor)
    }

    model.train()
    model.optimize(train_tensor)
    model.eval()

    with torch.inference_mode():
        ll_after = float(model.log_likelihood(oos_tensor, reduce=True)) / (N_EVAL * STEPS)
        decoded_after = model.predict(oos_tensor, mode="viterbi", verbose=False)
        pred_after = _flatten_predictions(decoded_after)
        permutation, oos_accuracy = _best_state_permutation(oos_y.reshape(-1), pred_after)
        aligned_pred = _apply_permutation(pred_after, permutation).reshape(N_EVAL, STEPS)
        oos_ari = _adjusted_rand_index(oos_y.reshape(-1), pred_after)

        trace = filter_model_sequence(model, oos_tensor)
        posterior = trace.state_posterior.detach().cpu().numpy()
        posterior_aligned = np.empty_like(posterior)
        for predicted_state, true_state in enumerate(permutation):
            posterior_aligned[..., true_state] = posterior[..., predicted_state]
        posterior_confidence = posterior_aligned.max(axis=-1).reshape(-1)
        posterior_prediction = posterior_aligned.argmax(axis=-1)
        posterior_correct = (posterior_prediction == oos_y).reshape(-1)

        boundary_f1 = _boundary_f1(oos_y, aligned_pred, tolerance=0)
        boundary_f1_tolerance_1 = _boundary_f1(oos_y, aligned_pred, tolerance=1)
        true_runs = _run_lengths(oos_y)
        predicted_runs = _run_lengths(aligned_pred)
        true_mean_run = float(true_runs.mean())
        predicted_mean_run = float(predicted_runs.mean())
        run_length_rel_error = abs(predicted_mean_run - true_mean_run) / true_mean_run

        true_transition = _empirical_transition_matrix(oos_y)
        predicted_transition = _empirical_transition_matrix(aligned_pred)
        active_rows = true_transition.sum(axis=1) > 0
        transition_mae = float(
            np.abs(predicted_transition[active_rows] - true_transition[active_rows]).mean()
        )
        occupancy_l1 = float(np.abs(_occupancy(aligned_pred) - _occupancy(oos_y)).sum())
        posterior_ece = _expected_calibration_error(posterior_confidence, posterior_correct)
        posterior_brier = float(
            np.square(posterior_aligned - np.eye(N_STATES, dtype=np.float32)[oos_y])
            .sum(axis=-1)
            .mean()
        )

        train_pred = _flatten_predictions(
            model.predict(train_tensor, mode="viterbi", verbose=False)
        )
        train_accuracy = _matched_accuracy(train_y.reshape(-1), train_pred)

    parameter_change = _parameter_l1_change(state_before, model.state_dict())
    train_health = evaluate_model_health(model, train_tensor)
    oos_health = evaluate_model_health(model, oos_tensor)
    train_snapshot = evaluate_validation_snapshot(model, train_tensor, label="train")
    oos_snapshot = evaluate_validation_snapshot(model, oos_tensor, label="oos")
    comparison = compare_validation_snapshots(train_snapshot, oos_snapshot)

    variable = [torch.tensor(oos_x[0, :53]), torch.tensor(oos_x[1, :101])]
    variable_pred = model.predict(variable, mode="viterbi", verbose=False)
    variable_lengths = [int(item.numel()) for item in variable_pred]

    return {
        "seed": seed,
        "scenario": scenario.name,
        "separation": scenario.separation,
        "parameter_l1_change": parameter_change,
        "ll_before": ll_before,
        "ll_after": ll_after,
        "ll_gain": ll_after - ll_before,
        "accuracy_before": accuracy_before,
        "oos_accuracy": oos_accuracy,
        "accuracy_gain": oos_accuracy - accuracy_before,
        "ari_before": ari_before,
        "oos_ari": oos_ari,
        "train_accuracy": train_accuracy,
        "generalization_gap": train_accuracy - oos_accuracy,
        "boundary_f1": boundary_f1,
        "boundary_f1_tolerance_1": boundary_f1_tolerance_1,
        "true_mean_run_length": true_mean_run,
        "predicted_mean_run_length": predicted_mean_run,
        "run_length_rel_error": run_length_rel_error,
        "transition_mae": transition_mae,
        "occupancy_l1": occupancy_l1,
        "posterior_ece": posterior_ece,
        "posterior_brier": posterior_brier,
        "posterior_mean_confidence": float(posterior_confidence.mean()),
        "train_health": train_health.as_dict(),
        "oos_health": oos_health.as_dict(),
        "validation_comparison": comparison.as_dict(),
        "same_model_fingerprint": train_snapshot.model_fingerprint
        == oos_snapshot.model_fingerprint,
        "different_data_fingerprint": train_snapshot.data_fingerprint
        != oos_snapshot.data_fingerprint,
        "variable_prediction_lengths": variable_lengths,
    }


def _median(rows: list[dict[str, object]], key: str) -> float:
    return float(np.median([float(row[key]) for row in rows]))


def _summarize(scenario: Scenario, rows: list[dict[str, object]]) -> dict[str, object]:
    median_accuracy = _median(rows, "oos_accuracy")
    median_accuracy_gain = _median(rows, "accuracy_gain")
    median_ll_gain = _median(rows, "ll_gain")
    median_ari = _median(rows, "oos_ari")
    median_boundary_f1 = _median(rows, "boundary_f1")
    median_boundary_f1_tolerance_1 = _median(rows, "boundary_f1_tolerance_1")
    median_run_length_rel_error = _median(rows, "run_length_rel_error")
    median_transition_mae = _median(rows, "transition_mae")
    median_occupancy_l1 = _median(rows, "occupancy_l1")
    median_ece = _median(rows, "posterior_ece")
    collapsed = sum(bool(row["oos_health"]["state_collapsed"]) for row in rows)
    parameters_changed = sum(float(row["parameter_l1_change"]) > 0.0 for row in rows)
    snapshot_contract = sum(
        bool(row["same_model_fingerprint"]) and bool(row["different_data_fingerprint"])
        for row in rows
    )
    variable_contract = sum(row["variable_prediction_lengths"] == [53, 101] for row in rows)

    learning_reasons: list[str] = []
    usefulness_reasons: list[str] = []
    if parameters_changed != len(rows):
        learning_reasons.append("parameters_did_not_change")
    if snapshot_contract != len(rows):
        learning_reasons.append("snapshot_contract")
    if variable_contract != len(rows):
        learning_reasons.append("variable_length_predict_contract")
    if (
        scenario.min_median_oos_accuracy is not None
        and median_accuracy < scenario.min_median_oos_accuracy
    ):
        learning_reasons.append("median_oos_accuracy")
    if (
        scenario.min_median_accuracy_gain is not None
        and median_accuracy_gain < scenario.min_median_accuracy_gain
    ):
        learning_reasons.append("median_accuracy_gain")
    if scenario.min_median_ll_gain is not None and median_ll_gain < scenario.min_median_ll_gain:
        learning_reasons.append("median_ll_gain")
    if (
        scenario.max_median_oos_accuracy is not None
        and median_accuracy > scenario.max_median_oos_accuracy
    ):
        learning_reasons.append("null_median_oos_accuracy")
    if scenario.max_abs_median_ari is not None and abs(median_ari) > scenario.max_abs_median_ari:
        learning_reasons.append("null_median_ari")
    if (
        scenario.min_median_boundary_f1 is not None
        and median_boundary_f1 < scenario.min_median_boundary_f1
    ):
        usefulness_reasons.append("median_boundary_f1")
    if scenario.name == "weak" and median_boundary_f1_tolerance_1 < 0.45:
        usefulness_reasons.append("median_boundary_f1_tolerance_1")
    if (
        scenario.max_median_run_length_rel_error is not None
        and median_run_length_rel_error > scenario.max_median_run_length_rel_error
    ):
        usefulness_reasons.append("median_run_length_rel_error")
    if (
        scenario.max_median_transition_mae is not None
        and median_transition_mae > scenario.max_median_transition_mae
    ):
        usefulness_reasons.append("median_transition_mae")
    if (
        scenario.max_median_occupancy_l1 is not None
        and median_occupancy_l1 > scenario.max_median_occupancy_l1
    ):
        usefulness_reasons.append("median_occupancy_l1")
    if scenario.max_median_ece is not None and median_ece > scenario.max_median_ece:
        usefulness_reasons.append("median_posterior_ece")
    if scenario.expect_collapse:
        if collapsed != len(rows):
            usefulness_reasons.append("expected_state_collapse")
    elif scenario.max_collapse_fraction is not None:
        if collapsed / len(rows) > scenario.max_collapse_fraction:
            usefulness_reasons.append("unexpected_state_collapse")

    learning_pass = not learning_reasons
    usefulness_applicable = scenario.separation > 0.0
    usefulness_pass = usefulness_applicable and not usefulness_reasons
    passed = learning_pass and (usefulness_pass if usefulness_applicable else True)

    return {
        "scenario": scenario.name,
        "separation": scenario.separation,
        "n_runs": len(rows),
        "median_ll_before": _median(rows, "ll_before"),
        "median_ll_after": _median(rows, "ll_after"),
        "median_ll_gain": median_ll_gain,
        "median_accuracy_before": _median(rows, "accuracy_before"),
        "median_oos_accuracy": median_accuracy,
        "median_accuracy_gain": median_accuracy_gain,
        "median_oos_ari": median_ari,
        "median_train_accuracy": _median(rows, "train_accuracy"),
        "median_generalization_gap": _median(rows, "generalization_gap"),
        "median_boundary_f1": median_boundary_f1,
        "median_boundary_f1_tolerance_1": median_boundary_f1_tolerance_1,
        "median_run_length_rel_error": median_run_length_rel_error,
        "median_transition_mae": median_transition_mae,
        "median_occupancy_l1": median_occupancy_l1,
        "median_posterior_ece": median_ece,
        "median_posterior_brier": _median(rows, "posterior_brier"),
        "median_posterior_confidence": _median(rows, "posterior_mean_confidence"),
        "median_effective_states": float(
            np.median([float(row["oos_health"]["effective_states"]) for row in rows])
        ),
        "median_max_occupancy": float(
            np.median([float(row["oos_health"]["max_occupancy"]) for row in rows])
        ),
        "state_collapsed_runs": collapsed,
        "parameters_changed_runs": parameters_changed,
        "snapshot_contract_runs": snapshot_contract,
        "variable_length_contract_runs": variable_contract,
        "learning_pass": learning_pass,
        "learning_failure_reasons": learning_reasons,
        "usefulness_applicable": usefulness_applicable,
        "usefulness_pass": usefulness_pass,
        "usefulness_failure_reasons": usefulness_reasons,
        "passed": passed,
        "failure_reasons": [*learning_reasons, *usefulness_reasons],
    }


def _parse_seeds(raw: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return values


def _worker(job: tuple[int, Scenario, int]) -> dict[str, object]:
    seed, scenario, max_iter = job
    return _run_one(seed, scenario, max_iter=max_iter)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prove NHSMM learning and held-out prediction behavior."
    )
    parser.add_argument("--scenario", choices=["all", *SCENARIOS], default="all")
    parser.add_argument("--seeds", type=_parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.max_iter < 1:
        parser.error("--max-iter must be >= 1")
    if args.workers < 1:
        parser.error("--workers must be >= 1")

    selected: Iterable[Scenario] = (
        SCENARIOS.values() if args.scenario == "all" else (SCENARIOS[args.scenario],)
    )
    payload: dict[str, object] = {
        "schema": "nhsmm-learning-prediction-v1",
        "config": {
            "n_states": N_STATES,
            "n_features": N_FEATURES,
            "max_duration": MAX_DURATION,
            "steps": STEPS,
            "n_train": N_TRAIN,
            "n_eval": N_EVAL,
            "max_iter": args.max_iter,
            "emission_init_mode": "kmeans",
        },
        "seeds": list(args.seeds),
        "scenarios": {},
    }

    all_passed = True
    for scenario in selected:
        jobs = [(seed, scenario, args.max_iter) for seed in args.seeds]
        if args.workers == 1:
            rows = [_worker(job) for job in jobs]
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
                rows = list(executor.map(_worker, jobs))

        summary = _summarize(scenario, rows)
        payload["scenarios"][scenario.name] = {
            "scenario": asdict(scenario),
            "summary": summary,
            "runs": rows,
        }
        all_passed = all_passed and bool(summary["passed"])
        print(json.dumps(summary, sort_keys=True), flush=True)

    payload["passed"] = all_passed
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
