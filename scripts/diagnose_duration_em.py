"""Independent unlabeled duration EM versus fixed-budget gradient controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch

from scripts.diagnose_duration_resets import fit, measure
from scripts.diagnose_model_conform_duration import D, MEANS, SIGMA, generate, make_model
from scripts.diagnose_tail_coverage import TRANSITION, require_coverage


def expanded_chain(pmf, hazard, transition):
    """Hidden state = (state, drawn duration bucket, elapsed age within bucket)."""
    width = pmf.shape[1]
    nodes = [(k, d, a) for k in range(2) for d in range(width) for a in range(d + 1)]
    index = {node: i for i, node in enumerate(nodes)}
    starts = np.array([[index[k, d, 0] for d in range(width)] for k in range(2)])
    ends = np.array([[index[k, d, d] for d in range(width)] for k in range(2)])
    matrix = np.zeros((len(nodes), len(nodes)))
    for i, (k, d, age) in enumerate(nodes):
        if age < d:
            matrix[i, index[k, d, age + 1]] = 1
        else:
            end_probability = hazard[k] if d == width - 1 else 1
            matrix[i, starts.reshape(-1)] += (
                end_probability * transition[k, :, None] * pmf
            ).reshape(-1)
            if d == width - 1:
                matrix[i, i] += 1 - hazard[k]
    initial = np.zeros(len(nodes))
    initial[starts.reshape(-1)] = (pmf * 0.5).reshape(-1)
    return matrix, initial, starts, ends, np.array([n[0] for n in nodes])


def expectation(emissions, pmf, hazard, transition):
    """Scaled forward/backward counts; terminal beta=1 means right censoring."""
    matrix, initial, starts, ends, states = expanded_chain(pmf, hazard, transition)
    batch, steps, _ = emissions.shape
    emission = emissions[:, :, states].transpose(1, 0, 2)
    alpha = np.empty_like(emission)
    scales = np.empty((steps, batch))
    frontier = np.broadcast_to(initial, (batch, len(initial))).copy()
    for t in range(steps):
        if t:
            frontier = frontier @ matrix
        frontier *= emission[t]
        scales[t] = frontier.sum(1)
        if not np.all(scales[t] > 0):
            raise ValueError("independent EM has zero evidence")
        frontier /= scales[t, :, None]
        alpha[t] = frontier
    beta = np.ones_like(frontier)
    bucket_counts = np.zeros_like(pmf)
    tail_ends, tail_stays = np.zeros(2), np.zeros(2)
    end_probabilities = np.ones_like(pmf)
    end_probabilities[:, -1] = hazard
    for t in range(steps - 2, -1, -1):
        weighted_next = emission[t + 1] * beta / scales[t + 1, :, None]
        source_end = alpha[t][:, ends] * end_probabilities[None]
        incoming_state = source_end.sum(2) @ transition
        destination = weighted_next[:, starts] * pmf[None]
        bucket_counts += (incoming_state[:, :, None] * destination).sum(0)
        continuation = weighted_next[:, ends[:, -1]]
        tail_frontier = alpha[t][:, ends[:, -1]]
        tail_ends += (tail_frontier * hazard * (destination.sum(2) @ transition.T)).sum(0)
        tail_stays += (tail_frontier * (1 - hazard) * continuation).sum(0)
        beta = weighted_next @ matrix.T
    bucket_counts += (alpha[0] * beta)[:, starts].sum(0)
    return {
        "ll": float(np.log(scales).sum()),
        "bucket_counts": bucket_counts,
        "tail_ends": tail_ends,
        "tail_stays": tail_stays,
    }


def gaussian_emissions(x):
    return np.exp(-0.5 * ((x[:, :, None] - MEANS) / SIGMA) ** 2).prod(-1) / (2 * math.pi * SIGMA**2)


def em_fit(x, transition, iterations):
    pmf = np.full((2, D), 1 / D)
    hazard = np.full(2, 0.5)
    emissions = gaussian_emissions(x.astype(np.float64))
    history = []
    for _ in range(iterations):
        counts = expectation(emissions, pmf, hazard, transition)
        history.append(counts["ll"])
        pmf = counts["bucket_counts"] / counts["bucket_counts"].sum(1, keepdims=True)
        trials = counts["tail_ends"] + counts["tail_stays"]
        hazard = np.divide(counts["tail_ends"], trials, out=hazard.copy(), where=trials > 0)
    history.append(expectation(emissions, pmf, hazard, transition)["ll"])
    if np.min(np.diff(history)) < -1e-7:
        raise AssertionError("independent EM likelihood decreased")
    return pmf, hazard, history


def model_from_law(seed, pmf, hazard):
    """Evaluate NumPy EM parameters through the public model; not used in EM fitting."""
    model = make_model(seed, True, 80, fixed=True).double()
    with torch.no_grad():
        model.dist.initial.logits.fill_(0)
        model.dist.emission.mu.copy_(torch.tensor(MEANS))
        model.dist.emission.log_var.fill_(math.log(math.expm1(SIGMA**2)))
        model.duration_logits_bias.fill_(1)
        model.dist.duration.logits.copy_(torch.tensor(np.log(pmf)))
        model.dist.duration.tail_end_probability.copy_(torch.tensor(hazard))
        model.dist.transition.logits.copy_(
            torch.tensor(np.log(TRANSITION))[:, None, :].expand(2, D, 2)
        )
    model.eval()
    return model


def run(seed, base, prior_path):
    train, _, ledger = generate(seed + 20000, tail=True, count=128, transition=TRANSITION)
    test, labels, test_ledger = generate(seed + 30000, tail=True, count=32, transition=TRANSITION)
    coverage = require_coverage(ledger)
    prior_bytes = prior_path.read_bytes()
    prior = json.loads(prior_bytes)["run"]["scenarios"]["tail"]
    for name, data in (("train", train), ("test", test)):
        if hashlib.sha256(data.tobytes()).hexdigest() != prior[name + "_sha256"]:
            raise AssertionError("frozen Step 3 dataset mismatch")
    started = time.perf_counter()
    pmf, hazard, history = em_fit(train, TRANSITION, 80)
    em_seconds = time.perf_counter() - started
    model = model_from_law(seed, pmf, hazard)
    model._diagnostic_completed_iterations = 80
    model._diagnostic_initial_train_ll = history[0]
    arms = {"numpy_em_80": measure(model, test, labels, test_ledger, True, TRANSITION)}
    arms["numpy_em_80"].update(
        train_ll_history=history, final_train_ll=history[-1], fit_seconds=em_seconds
    )
    with torch.no_grad():
        train_ll = float(
            model.log_likelihood(torch.tensor(train, dtype=torch.float64), reduce=True)
        )
    if abs(train_ll - history[-1]) > 1e-7:
        raise AssertionError("EM/model training likelihood mismatch")
    print(f"seed={seed} arm=numpy_em_80 complete", flush=True)
    for iterations in (80, 400):
        started = time.perf_counter()
        model = fit(seed, True, train, TRANSITION, "known_transition", iterations)
        seconds = time.perf_counter() - started
        name = f"adam_{iterations}"
        arms[name] = measure(model, test, labels, test_ledger, True, TRANSITION)
        with torch.no_grad():
            arms[name]["final_train_ll"] = float(
                model.log_likelihood(torch.tensor(train, dtype=torch.float64), reduce=True)
            )
        arms[name]["fit_seconds"] = seconds
        if abs(arms[name]["initial_train_ll"] - history[0]) > 1e-7:
            raise AssertionError("EM/Adam initial likelihood mismatch")
        if (
            iterations == 80
            and abs(
                arms[name]["test_ll_per_row"] - prior["arms"]["known_transition"]["test_ll_per_row"]
            )
            > 1e-12
        ):
            raise AssertionError("Adam 80 did not reproduce frozen Step 3 result")
        print(f"seed={seed} arm={name} complete", flush=True)
    return {
        "seed": seed,
        "base_commit": base,
        "train_sha256": prior["train_sha256"],
        "test_sha256": prior["test_sha256"],
        "prior_report_sha256": hashlib.sha256(prior_bytes).hexdigest(),
        "coverage": coverage,
        "arms": arms,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-commit", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    prior = Path(
        f"docs/validation/results/tail-coverage-step3-seed{args.seed}-native-20261012.json"
    )
    payload = {
        "schema": "nhsmm-independent-duration-em-diagnostic-v1",
        "execution": {
            "runtime": "openai-native-workspace",
            "base_commit": args.base_commit,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "workers": 1,
            "source_sha256": {
                p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in [
                    __file__,
                    "scripts/diagnose_duration_resets.py",
                    "scripts/diagnose_model_conform_duration.py",
                    "scripts/diagnose_tail_coverage.py",
                    "scripts/diagnose_model_identifiability.py",
                ]
            },
        },
        "contract": {
            "data": "frozen Step 3 D+ datasets: 128 train, 32 independent test sequences, 128 rows",
            "initialization": "uniform duration PMF, hazard .5; matched initial marginal likelihood asserted",
            "known_nuisance": "Gaussian emissions, initial .5/.5, true .02 reset matrix; gate 1, temperature 1",
            "labels": "neither EM nor Adam sees labels or episode ledger; coverage gate and evaluation only",
            "em": "NumPy scaled expanded-bucket forward/backward; closed-form PMF and hazard M-steps; terminal beta=1",
            "budgets": "80 EM versus 80 Adam updates; additional predeclared Adam 400; equal iterations are not equal compute",
            "adam": "Float64, lr .008, gradient clip 5; same script helper as Step 3",
            "limits": "known nuisance duration-learning experiment; not full unsupervised model recovery or production optimizer bug proof",
        },
        "run": run(args.seed, args.base_commit, prior),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                a: {
                    k: v[k]
                    for k in (
                        "observable_run_tv_mean",
                        "duration_tv_mean",
                        "test_ll_per_row",
                        "final_train_ll",
                        "fit_seconds",
                    )
                }
                for a, v in payload["run"]["arms"].items()
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
