"""Fixed-size tail-exposure and finite-versus-D+ run-support controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch

from scripts.diagnose_duration_resets import censored_duration_mle, fit, measure
from scripts.diagnose_model_conform_duration import (
    D,
    PMF,
    TAIL,
    MEANS,
    SIGMA,
    generate,
    complete_duration_nll,
)

TRANSITION = np.array([[0.02, 0.98], [0.98, 0.02]])


def tail_exposure(ledger):
    rows = []
    for state in range(2):
        episodes = [e for row in ledger for e in row if e["state"] == state and e["observed"] >= D]
        ends = sum(not e["censored"] for e in episodes)
        failures = sum(e["observed"] - D for e in episodes)
        trials = ends + failures
        rows.append(
            {
                "state": state,
                "completed_tail_episodes": ends,
                "censored_tail_episodes": sum(e["censored"] for e in episodes),
                "tail_survival_steps": failures,
                "tail_trials": trials,
                "hazard_mle": ends / trials if trials else None,
            }
        )
    return rows


def require_coverage(ledger, minimum_ends=30, minimum_trials=60):
    rows = tail_exposure(ledger)
    if any(
        r["completed_tail_episodes"] < minimum_ends or r["tail_trials"] < minimum_trials
        for r in rows
    ):
        raise ValueError(f"fixed dataset fails preregistered tail exposure gate: {rows}")
    return rows


def run_lengths(labels):
    lengths = []
    for row in labels:
        edges = np.r_[0, np.flatnonzero(row[1:] != row[:-1]) + 1, len(row)]
        lengths.extend(np.diff(edges).tolist())
    return {"max_observed_run": max(lengths), "observed_runs_above_D": sum(v > D for v in lengths)}


def run(seed, iterations, train_count=128, test_count=32):
    torch.set_num_threads(1)
    result = {"seed": seed, "scenarios": {}}
    # Freeze and check all generated datasets before any fitting.
    datasets = {}
    for tail in (True, False):
        train, train_labels, ledger = generate(
            seed + 20000, tail=tail, count=train_count, transition=TRANSITION
        )
        test, labels, test_ledger = generate(
            seed + 30000, tail=tail, count=test_count, transition=TRANSITION
        )
        datasets[tail] = train, train_labels, ledger, test, labels, test_ledger
    require_coverage(datasets[True][2])
    for tail in (True, False):
        train, train_labels, ledger, test, labels, test_ledger = datasets[tail]
        arms = (
            (("known_transition", True, False), ("hazard_only_known_pmf", True, True))
            if tail
            else (("finite_no_resets", False, False), ("D_plus_no_resets", True, False))
        )
        metrics = {}
        for name, model_tail, known_pmf in arms:
            arm = "known_transition" if tail else "no_resets"
            model = fit(
                seed, model_tail, train, TRANSITION, arm, iterations, known_duration_pmf=known_pmf
            )
            metrics[name] = measure(model, test, labels, test_ledger, tail, TRANSITION)
            with torch.no_grad():
                metrics[name]["final_train_ll"] = float(
                    model.log_likelihood(torch.tensor(train, dtype=torch.float64), reduce=True)
                )
            if tail:
                metrics[name]["tail_absolute_error"] = np.abs(
                    np.array(metrics[name]["tail_hazard"]) - TAIL
                ).tolist()
            else:
                # Run durations on reset data have a different law from latent
                # episodes. Do not label differences from episode truth as errors.
                for key in (
                    "duration_tv_mean",
                    "duration_tv_states",
                    "complete_episode_nll",
                    "true_complete_episode_nll",
                    "true_mean",
                ):
                    metrics[name].pop(key)
                metrics[name]["modeled_run_mean"] = metrics[name].pop("learned_mean")
            metrics[name]["duration_interpretation"] = (
                "latent episode" if tail else "observable run; compare run-TV, not episode-TV"
            )
            print(f"seed={seed} generator_tail={tail} arm={name} complete", flush=True)
        pmf, h, history = censored_duration_mle(ledger, tail=tail, iterations=iterations)
        small = generate(seed + 20000, tail=tail, count=6, transition=TRANSITION)
        if not np.array_equal(train[:6], small[0]):
            raise AssertionError(
                "larger training set did not preserve original six-sequence prefix"
            )
        result["scenarios"]["tail" if tail else "finite"] = {
            "train_sha256": hashlib.sha256(train.tobytes()).hexdigest(),
            "test_sha256": hashlib.sha256(test.tobytes()).hexdigest(),
            "small_train_sha256": hashlib.sha256(small[0].tobytes()).hexdigest(),
            "train_exposure": tail_exposure(ledger),
            "test_exposure": tail_exposure(test_ledger),
            "small_train_exposure": tail_exposure(small[2]),
            "train_runs": run_lengths(train_labels),
            "test_runs": run_lengths(labels),
            "arms": metrics,
            "labeled_censored_mle": {
                "pmf": pmf.tolist(),
                "tail_hazard": h.tolist(),
                "train_nll_history": history,
                "test_complete_episode_nll": complete_duration_nll(pmf, h, test_ledger),
                "scope": "privileged episode ledger, not an unsupervised comparator",
            },
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--base-commit", required=True)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    payload = {
        "schema": "nhsmm-tail-exposure-support-diagnostic-v1",
        "execution": {
            "base_commit": args.base_commit,
            "runtime": "openai-native-workspace",
            "workers": 1,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "source_sha256": {
                p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in [
                    __file__,
                    "scripts/diagnose_duration_resets.py",
                    "scripts/diagnose_model_conform_duration.py",
                    "scripts/diagnose_model_identifiability.py",
                ]
            },
        },
        "generative_law": {
            "pmf": PMF.tolist(),
            "tail_hazard": TAIL.tolist(),
            "means": MEANS.tolist(),
            "sigma": SIGMA,
            "transition": TRANSITION.tolist(),
            "initial": [0.5, 0.5],
        },
        "contract": {
            "train_sequences": 128,
            "test_sequences": 32,
            "steps": 128,
            "minimum_completed_tail_episodes_per_state": 30,
            "minimum_tail_trials_per_state": 60,
            "dataset_selection": "fixed size/seeds; fail insufficient exposure, never regenerate or select by fit/test metrics",
            "training": "known Gaussian emissions and initial; gate fixed 1; temperature 1; Float64; Adam .008; clip 5",
            "iterations": args.iterations,
            "censoring": "terminal episode always right-censored",
            "limits": "three synthetic seeds; known nuisance parameters; no general acceptance, production default or independent unsupervised EM claim",
        },
        "run": run(args.seed, args.iterations),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                s: {
                    a: {
                        k: r[k]
                        for k in ("tail_hazard", "observable_run_tv_mean", "test_ll_per_row")
                    }
                    for a, r in v["arms"].items()
                }
                for s, v in payload["run"]["scenarios"].items()
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
