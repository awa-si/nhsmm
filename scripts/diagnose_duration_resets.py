"""Controlled encoderless reset-capacity experiment; production defaults unchanged."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
import torch

from scripts.diagnose_model_conform_duration import (
    D,
    MEANS,
    PMF,
    SIGMA,
    TAIL,
    complete_duration_nll,
    evaluate,
    generate,
    law,
    ledger_coverage,
    make_model,
)
from scripts.diagnose_model_identifiability import observable_run_law

ARMS = ("age_specific_resets", "tied_resets", "no_resets", "known_transition")


def censored_duration_mle(ledger, *, tail, iterations):
    """Independent labeled EM: censored bucket allocation plus geometric exposure."""
    pmf = np.full((2, D), 1 / D)
    hazards = np.full(2, 0.5) if tail else np.ones(2)
    history = [complete_duration_nll(pmf, hazards, ledger)]
    for _ in range(iterations):
        counts = np.zeros_like(pmf)
        ends, failures = np.zeros(2), np.zeros(2)
        for row in ledger:
            for e in row:
                state, length = e["state"], e["observed"]
                if e["censored"] and length < D:
                    weights = pmf[state, length - 1 :]
                    counts[state, length - 1 :] += weights / weights.sum()
                else:
                    counts[state, min(length, D) - 1] += 1
                if tail and length >= D:
                    ends[state] += not e["censored"]
                    failures[state] += length - D
        for state in range(2):
            if counts[state].sum() > 0:
                pmf[state] = counts[state] / counts[state].sum()
            if tail and ends[state] + failures[state] > 0:
                hazards[state] = ends[state] / (ends[state] + failures[state])
        history.append(complete_duration_nll(pmf, hazards, ledger))
    if np.max(np.diff(history)) > 1e-8:
        raise AssertionError("labeled censored likelihood decreased")
    return pmf, hazards, history


def independent_log_likelihood(x, pmf, hazards, transition):
    """NumPy probability-space age-chain likelihood with per-row scaling."""
    end = pmf / np.cumsum(pmf[:, ::-1], axis=1)[:, ::-1]
    end[:, -1] = hazards
    total = 0.0
    for sequence in x:
        frontier = np.zeros((2, D))
        frontier[:, 0] = 0.5
        for t, observation in enumerate(sequence):
            if t:
                following = np.zeros_like(frontier)
                following[:, 1:] += frontier[:, :-1] * (1 - end[:, :-1])
                following[:, -1] += frontier[:, -1] * (1 - end[:, -1])
                following[:, 0] += np.einsum("ka,ka,kaj->j", frontier, end, transition)
                frontier = following
            density = np.exp(-0.5 * ((observation - MEANS) / SIGMA) ** 2).prod(1)
            density /= (math.sqrt(2 * math.pi) * SIGMA) ** 2
            frontier *= density[:, None]
            scale = frontier.sum()
            if not scale > 0:
                raise AssertionError("zero independent likelihood mass")
            total += math.log(scale)
            frontier /= scale
    return total


def fit(seed, tail, x, transition, arm, iterations, *, known_duration_pmf=False):
    if arm not in ARMS or iterations < 1:
        raise ValueError("supported arm and positive budget required")
    model = make_model(seed, tail, iterations, fixed=True).double()
    with torch.no_grad():
        model.dist.initial.logits.fill_(0)
        model.dist.emission.mu.copy_(torch.tensor(MEANS))
        model.dist.emission.log_var.fill_(math.log(math.expm1(SIGMA**2)))
        model.duration_logits_bias.fill_(1)
        model.dist.duration.initialize(mode="uniform", jitter=0)
        if known_duration_pmf:
            model.dist.duration.logits.copy_(torch.tensor(np.log(PMF)))
        model.dist.transition.logits.fill_(0)
        if arm == "known_transition":
            model.dist.transition.logits.copy_(
                torch.tensor(np.log(np.maximum(transition, 1e-30)))[:, None, :].expand(2, D, 2)
            )
        if arm == "no_resets" or (arm == "known_transition" and np.trace(transition) == 0):
            model.dist.transition._structural_constraint &= ~torch.eye(2, dtype=torch.bool)[:, None]
    for p in model.parameters():
        p.requires_grad_(False)
    params = [] if known_duration_pmf else [model.dist.duration.logits]
    if tail:
        params.append(model.dist.duration.tail_end_probability)
    if arm in ("age_specific_resets", "tied_resets"):
        params.append(model.dist.transition.logits)
    if not params:
        raise ValueError("fit requires at least one trainable parameter")
    for p in params:
        p.requires_grad_(True)
    if arm == "tied_resets":
        model.dist.transition.logits.register_hook(lambda g: g.sum(1, keepdim=True).expand_as(g))
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.Adam(params, lr=0.008)
    tx = torch.tensor(x, dtype=torch.float64)
    history = []
    for _ in range(iterations):
        optimizer.zero_grad()
        ll = model.log_likelihood(tx, reduce=True)
        history.append(float(ll.detach()))
        (-ll).backward()
        torch.nn.utils.clip_grad_norm_(params, 5)
        optimizer.step()
        model.dist.duration._project_tail_parameter()
    for n, p in model.named_parameters():
        if n in frozen and not torch.equal(p, frozen[n]):
            raise AssertionError("frozen nuisance parameter changed")
    if arm == "tied_resets":
        logits = model.dist.transition.logits.detach()
        if not torch.equal(logits, logits[:, :1].expand_as(logits)):
            raise AssertionError("tied transitions diverged across age")
    model.eval()
    model._diagnostic_completed_iterations = len(history)
    model._diagnostic_initial_train_ll = history[0]
    return model


def measure(model, x, labels, ledger, tail, transition):
    # Existing metrics require input/model dtype agreement.
    row = evaluate(model, x.astype(np.float64), labels, ledger, tail, align=False)
    pmf, hazards = law(model)
    with torch.no_grad():
        trans = model.dist.transition.log_matrix(T=1).exp().reshape(2, D, 2).numpy()
    reference = independent_log_likelihood(x, pmf, hazards, trans) / (x.shape[0] * x.shape[1])
    error = abs(reference - row["test_ll_per_row"])
    if error > 2e-6:
        raise AssertionError(f"independent likelihood mismatch: {error}")
    actual_h = TAIL if tail else np.ones(2)
    fitted = [observable_run_law(pmf[k], hazards[k], trans[k, :, k]) for k in range(2)]
    truth = [
        observable_run_law(PMF[k], actual_h[k], np.repeat(transition[k, k], D)) for k in range(2)
    ]
    row.update(
        independent_ll_error_per_row=error,
        observable_run_tv_mean=float(
            np.mean([np.abs(a[0] - b[0]).sum() / 2 for a, b in zip(fitted, truth)])
        ),
        observable_run_means=[a[1] for a in fitted],
        true_observable_run_means=[a[1] for a in truth],
        completed_iterations=model._diagnostic_completed_iterations,
        initial_train_ll=model._diagnostic_initial_train_ll,
    )
    return row


def run(seed, tail, reset, iterations):
    transition = np.array([[reset, 1 - reset], [1 - reset, reset]])
    train, _, train_ledger = generate(seed + 20000, tail=tail, transition=transition)
    test, labels, ledger = generate(seed + 30000, tail=tail, count=4, transition=transition)
    arms = {}
    for arm in ARMS:
        model = fit(seed, tail, train, transition, arm, iterations)
        arms[arm] = measure(model, test, labels, ledger, tail, transition)
        with torch.no_grad():
            arms[arm]["final_train_ll"] = float(
                model.log_likelihood(torch.tensor(train, dtype=torch.float64), reduce=True)
            )
        print(f"seed={seed} tail={tail} reset={reset} arm={arm} complete", flush=True)
    p, h, history = censored_duration_mle(train_ledger, tail=tail, iterations=iterations)
    fitted = [observable_run_law(p[k], h[k], np.repeat(reset, D)) for k in range(2)]
    actual_h = TAIL if tail else np.ones(2)
    truth = [observable_run_law(PMF[k], actual_h[k], np.repeat(reset, D)) for k in range(2)]
    return {
        "seed": seed,
        "scenario": "tail" if tail else "finite",
        "generator_reset": reset,
        "train_sha256": hashlib.sha256(train.tobytes()).hexdigest(),
        "test_sha256": hashlib.sha256(test.tobytes()).hexdigest(),
        "train_coverage": ledger_coverage(train_ledger),
        "test_coverage": ledger_coverage(ledger),
        "arms": arms,
        "labeled_censored_mle": {
            "pmf": p.tolist(),
            "tail_hazard": h.tolist(),
            "completed_iterations": iterations,
            "train_nll_history": history,
            "test_complete_episode_nll": complete_duration_nll(p, h, ledger),
            "observable_run_tv_mean": float(
                np.mean([np.abs(a[0] - b[0]).sum() / 2 for a, b in zip(fitted, truth)])
            ),
            "scope": "privileged episode ledger, known transitions; not an unsupervised baseline",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", default="901,902,903")
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--base-commit", required=True)
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    if len(set(seeds)) != len(seeds) or args.iterations < 1:
        parser.error("distinct seeds and positive iterations required")
    torch.set_num_threads(1)
    rows = [
        run(seed, tail, reset, args.iterations)
        for seed in seeds
        for tail in (False, True)
        for reset in (0.0, 0.02)
    ]
    metrics = (
        "duration_tv_mean",
        "observable_run_tv_mean",
        "test_ll_per_row",
        "self_transition_mean",
        "accuracy",
    )
    summary = {
        f"{scenario}/reset={reset}": {
            arm: {
                m: float(
                    np.median(
                        [
                            r["arms"][arm][m]
                            for r in rows
                            if r["scenario"] == scenario and r["generator_reset"] == reset
                        ]
                    )
                )
                for m in metrics
            }
            for arm in ARMS
        }
        for scenario in ("finite", "tail")
        for reset in (0.0, 0.02)
    }
    payload = {
        "schema": "nhsmm-duration-reset-diagnostic-v1",
        "execution": {
            "runtime": "openai-native-workspace",
            "base_commit": args.base_commit,
            "diagnostic_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "workers": 1,
        },
        "iterations": args.iterations,
        "seeds": seeds,
        "runs": rows,
        "summary": summary,
        "contract": {
            "data": "6 train, 4 independent test episode-start sequences of 128 rows; final episode right-censored",
            "known_nuisance": "initial 0.5/0.5 and true Gaussian emissions fixed in every arm",
            "learning": "Float64 marginal likelihood, temperature 1, gate fixed 1, Adam lr .008, gradient clip 5; equal update budgets",
            "resets": "age-specific free, age-tied free, structurally zero, or known true matrix; script-only masks/hooks",
            "labels": "excluded from marginal fits; privileged independent labeled censored EM reported separately",
            "independent_control": "NumPy probability-space age-chain likelihood at each fitted law; not independent unsupervised EM",
            "limits": "small synthetic mechanism experiment; sparse state-1 tail exposure; no production defaults, trading or general statistical acceptance",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
