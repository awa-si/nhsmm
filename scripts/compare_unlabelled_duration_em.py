"""Independent observation-only HSMM EM versus production Adam and sample size."""

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

from scripts.generate_trading_regime_dataset import BASE_TRANSITION
from scripts.verify_isolated_trading_hsmm import (
    D,
    MEANS,
    check_recursion,
    lse,
    make_model,
    model_law,
    sequences,
    true_law,
)
from scripts.verify_unlabelled_trading_hsmm import SIGMA, exposure, independent_ll

ITERATIONS = 30
LARGE_COUNT = 1024
STEPS = 300
CHUNK = 64


def emission_scores(x):
    return -0.5 * (
        math.log(2 * math.pi * SIGMA**2) + (x[:, :, 0, None].astype(float) - MEANS) ** 2 / SIGMA**2
    )


def expectation(scores, pmf, transition, initial):
    """Independent probability forward/log backward; endpoint is right censored.

    Completed-event posteriors count age at boundaries strictly before T.
    Terminal state-age posteriors allocate latent lengths conditional on survival.
    No observed states, segment boundaries or hard-decoded paths enter this code.
    """
    batch, steps, states = scores.shape
    width = pmf.shape[1]
    survival = np.cumsum(pmf[:, ::-1], axis=1)[:, ::-1]
    hazard = np.divide(pmf, survival, out=np.zeros_like(pmf), where=survival > 0)
    continuation = np.divide(
        np.pad(survival[:, 1:], ((0, 0), (0, 1))),
        survival,
        out=np.zeros_like(pmf),
        where=survival > 0,
    )
    with np.errstate(divide="ignore"):
        log_end = np.log(hazard)
        log_continue = np.log(continuation)
        log_transition = np.log(transition)
    # Relative emissions avoid underflow from absolute Gaussian densities.
    offset = scores.max(2)
    relative = np.exp(scores - offset[:, :, None])
    alpha = np.zeros((batch, steps, states, width))
    log_z = np.empty((batch, steps))
    frontier = np.zeros((batch, states, width))
    frontier[:, :, 0] = initial
    for t in range(steps):
        if t:
            predicted = np.zeros_like(frontier)
            predicted[:, :, 0] = (frontier * hazard).sum(2) @ transition
            predicted[:, :, 1:] = frontier[:, :, :-1] * continuation[:, :-1]
            frontier = predicted
        frontier = frontier * relative[:, t, :, None]
        normalizer = frontier.sum((1, 2))
        if np.any(normalizer <= 0) or not np.isfinite(normalizer).all():
            raise AssertionError("impossible or non-finite forward evidence")
        log_z[:, t] = np.log(normalizer) + offset[:, t]
        frontier /= normalizer[:, None, None]
        alpha[:, t] = frontier
    complete = np.zeros_like(pmf)
    censored = alpha[:, -1].sum(0)
    beta = np.zeros((batch, states, width))  # log backward likelihood / suffix evidence
    for t in range(steps - 2, -1, -1):
        future = scores[:, t + 1] + beta[:, :, 0]
        boundary = lse(log_transition[None] + future[:, None, :], 2)
        with np.errstate(divide="ignore"):
            log_alpha = np.log(alpha[:, t])
        events = log_alpha + log_end[None] + boundary[:, :, None] - log_z[:, t + 1, None, None]
        complete += np.exp(events).sum(0)
        continuing = np.full_like(beta, -np.inf)
        continuing[:, :, :-1] = (
            log_continue[None, :, :-1] + scores[:, t + 1, :, None] + beta[:, :, 1:]
        )
        beta = np.logaddexp(log_end[None] + boundary[:, :, None], continuing)
        beta -= log_z[:, t + 1, None, None]
    ratios = np.divide(censored, survival, out=np.zeros_like(censored), where=survival > 0)
    expected = complete + pmf * np.cumsum(ratios, axis=1)
    ll = log_z.sum(1)
    with np.errstate(divide="ignore"):
        backward_ll = lse(np.log(initial)[None] + scores[:, 0] + beta[:, :, 0], 1)
    # beta is suffix likelihood divided by suffix evidence: initial evidence remains.
    np.testing.assert_allclose(backward_ll, log_z[:, 0], rtol=0, atol=1e-8)
    if not np.isfinite(expected).all() or not np.isfinite(ll).all():
        raise AssertionError("non-finite expectation")
    np.testing.assert_allclose(censored.sum(), batch, rtol=0, atol=1e-8)
    np.testing.assert_allclose(expected.sum(), complete.sum() + batch, rtol=0, atol=1e-8)
    return ll, expected, complete, censored


def aggregate(scores, pmf, chunk=CHUNK):
    expected = np.zeros_like(pmf)
    likelihood = 0.0
    for a in range(0, len(scores), chunk):
        ll, counts, _, _ = expectation(
            scores[a : a + chunk], pmf, BASE_TRANSITION, np.full(4, 0.25)
        )
        likelihood += ll.sum()
        expected += counts
    return float(likelihood), expected


def fit_em(x, initial_pmf, *, iterations=ITERATIONS, label="EM"):
    scores = emission_scores(x)
    p = initial_pmf.copy()
    history = []
    start = time.monotonic()
    # Every history item scores the distribution at its stated completed-update count.
    for update in range(iterations + 1):
        ll, counts = aggregate(scores, p)
        score = counts - p * counts.sum(1, keepdims=True)
        history.append(
            {
                "completed_updates": update,
                "ll": ll,
                "maximum_softmax_gradient_per_sequence": float(np.abs(score).max() / len(x)),
                "elapsed_seconds": time.monotonic() - start,
            }
        )
        if update and ll < history[-2]["ll"] - 1e-6:
            raise AssertionError("EM marginal objective decreased")
        if update == 0 or update % 5 == 0:
            print(
                f"{label} EM={update}/{iterations} seconds={time.monotonic() - start:.1f}",
                flush=True,
            )
        if update < iterations:
            if np.any(counts.sum(1) <= 0):
                raise AssertionError("state has no posterior episode exposure")
            p = counts / counts.sum(1, keepdims=True)
    return p, history


def posterior_gradient_check(x, pmf, *, support=None):
    # Existing callers retain their oracle control; new callers can pass generic support.
    carrier = true_law() if support is None else support / support.sum(1, keepdims=True)
    model = make_model(carrier, sigma=SIGMA)
    with torch.no_grad():
        model.dist.duration.logits.copy_(torch.tensor(np.log(np.maximum(pmf, 1e-300))))
    actual_p = model_law(model).detach().numpy()
    model.dist.duration.logits.requires_grad_(True)
    public_ll = model.log_likelihood(torch.tensor(x), reduce=True)
    public_ll.backward()
    ll, counts, _, _ = expectation(emission_scores(x), actual_p, BASE_TRANSITION, np.full(4, 0.25))
    gradient = counts - actual_p * counts.sum(1, keepdims=True)
    error = float(np.max(np.abs(gradient - model.dist.duration.logits.grad.numpy())))
    independent_error = abs(float(ll.sum()) - independent_ll(x, actual_p))
    public_error = abs(float(public_ll.detach()) - float(ll.sum()))
    if max(independent_error, public_error) > 1e-7 or error > 1e-7:
        raise AssertionError(
            f"posterior-score mismatch {error}, LL {independent_error}/{public_error}"
        )
    return {
        "maximum_all_logit_gradient_error": error,
        "public_ll_error": public_error,
        "segment_reference_ll_error": independent_error,
    }


def metrics(p, truth):
    mean = p @ np.arange(1, D + 1)
    return {
        "pmf_tv": (np.abs(p - truth).sum(1) / 2).tolist(),
        "mean": mean.tolist(),
        "mean_rmse": float(np.sqrt(np.mean((mean - truth @ np.arange(1, D + 1)) ** 2))),
        "fine_pmf_gate": bool(np.max(np.abs(p - truth).sum(1) / 2) <= 0.12),
    }


def run(prior):
    seed = prior["seed"]
    truth = true_law()
    initial = (truth > 0) / (truth > 0).sum(1, keepdims=True)
    x, _ = sequences(seed + 400000, count=128, steps=STEPS, sigma=SIGMA)
    if hashlib.sha256(x.tobytes()).hexdigest() != prior["training_observation_sha256"]:
        raise AssertionError("Step13 training replay hash mismatch")
    heldout, heldout_labels = sequences(seed + 500000, count=24, steps=STEPS, sigma=SIGMA)
    if hashlib.sha256(heldout.tobytes()).hexdigest() != prior["heldout_observation_sha256"]:
        raise AssertionError("Step13 heldout replay hash mismatch")
    derivative = posterior_gradient_check(x[:1], initial)
    small, small_history = fit_em(x, initial, label=f"seed={seed} small")
    adam = np.array(prior["fitted_pmf"])
    adam_ll, _ = aggregate(emission_scores(x), adam)
    if abs(adam_ll - prior["train_ll_final"]) > 1e-7:
        raise AssertionError("Step13 Adam objective replay mismatch")
    # Paired size control: the first128 observation sequences are exactly the small arm.
    large_x, evaluation_labels = sequences(
        seed + 400000, count=LARGE_COUNT, steps=STEPS, sigma=SIGMA
    )
    np.testing.assert_array_equal(large_x[:128], x)
    large, large_history = fit_em(large_x, initial, label=f"seed={seed} large")
    test_scores = emission_scores(heldout)
    initial_test, _ = aggregate(test_scores, initial)
    small_test, _ = aggregate(test_scores, small)
    large_test, _ = aggregate(test_scores, large)
    model = make_model(truth, sigma=SIGMA)
    with torch.no_grad():
        model.dist.duration.logits.copy_(torch.tensor(np.log(np.maximum(large, 1e-300))))
    recursion = check_recursion(model, heldout[:2], heldout_labels[:2], sigma=SIGMA)
    return {
        "seed": seed,
        "labels_used_in_learning": False,
        "small_observation_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
        "large_observation_sha256": hashlib.sha256(large_x.tobytes()).hexdigest(),
        "paired_first128_identical": True,
        "gradient_check": derivative,
        "small_em_history": small_history,
        "large_em_history": large_history,
        "small_adam_ll": adam_ll,
        "small_em_ll": small_history[-1]["ll"],
        "small_em_minus_adam_nats_per_row": (small_history[-1]["ll"] - adam_ll)
        / x.shape[0]
        / STEPS,
        "adam_metrics": metrics(adam, truth),
        "small_em_metrics": metrics(small, truth),
        "large_em_metrics": metrics(large, truth),
        "initial_heldout_ll": initial_test,
        "small_em_heldout_ll": small_test,
        "large_em_heldout_ll": large_test,
        "small_heldout_improvement_nats_per_row": (small_test - initial_test) / heldout_labels.size,
        "large_heldout_improvement_nats_per_row": (large_test - initial_test) / heldout_labels.size,
        "large_exposure_evaluation_only": exposure(evaluation_labels),
        "large_law_public_recursion": recursion,
        "small_fitted_pmf": small.tolist(),
        "large_fitted_pmf": large.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    prior_path = Path("docs/validation/results/unlabelled-hsmm-step13-native-20261012.json")
    prior = json.loads(prior_path.read_text())
    paths = [
        "scripts/compare_unlabelled_duration_em.py",
        "scripts/verify_unlabelled_trading_hsmm.py",
        "scripts/verify_isolated_trading_hsmm.py",
        "scripts/generate_trading_regime_dataset.py",
        "nhsmm/models/base.py",
        "nhsmm/distributions/default.py",
        "nhsmm/filtering.py",
    ]
    payload = {
        "schema": "nhsmm-unlabelled-em-size-v1",
        "execution": {
            "runtime": "openai-native-workspace",
            "base_commit": args.base_commit,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "workers": 1,
            "source_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
            "step13_report_sha256": hashlib.sha256(prior_path.read_bytes()).hexdigest(),
        },
        "contract": {
            "training": "observation-only independent HSMM forward/backward EM; uniform-support initial PMF;30 exact M-steps; labels evaluation only",
            "sample": "paired128 versus1024 sequences/seed,300 rows; first128 identical to Step13; unchanged24 heldout sequences; three seeds901/902/903",
            "nuisance": "Step13 known Gaussian sigma2.0, transition/no resets, initial state, D260 and support; fresh starts and terminal right censoring",
            "gates": "EM objective monotonic within1e-6; all1040-coordinate public gradient match<=1e-7; LL reference<=1e-7; public state probabilities<=1e-8; report separately heldout improvement and each-TV<=0.12",
            "limits": "bounded EM reference, no global MLE or convergence guarantee; separate sample size and optimizer comparisons, no Trading features, joint optimize, regularization, runtime/artifact mask or live acceptance",
        },
        "runs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for r in prior["runs"]:
        payload["runs"].append(run(r))
        args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    k: payload["runs"][-1][k]
                    for k in [
                        "seed",
                        "small_em_minus_adam_nats_per_row",
                        "small_em_metrics",
                        "large_em_metrics",
                        "small_heldout_improvement_nats_per_row",
                        "large_heldout_improvement_nats_per_row",
                    ]
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
