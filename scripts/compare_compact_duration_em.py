"""Training-selected compact duration histograms versus unrestricted unlabelled EM."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from scripts.compare_unlabelled_duration_em import (
    aggregate,
    emission_scores,
    metrics,
    posterior_gradient_check,
)
from scripts.verify_isolated_trading_hsmm import (
    check_recursion,
    make_model,
    sequences,
    true_law,
)
from scripts.verify_unlabelled_trading_hsmm import SIGMA

WIDTHS = (4, 8, 16)
INNER_UPDATES = 20
OUTER_UPDATES = 30
STEPS = 300
LARGE_COUNT = 1024


def partitions(support, width):
    """Disjoint constant-density interior blocks; clipped endpoints stay free."""
    support = np.asarray(support, dtype=bool)
    if support.ndim != 2 or not isinstance(width, int) or width < 1:
        raise ValueError("2D support and positive integer width required")
    groups = []
    for row in support:
        indices = np.flatnonzero(row)
        if not len(indices) or not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
            raise ValueError("support must be nonempty and contiguous for each state")
        lo, hi = int(indices[0]), int(indices[-1])
        state_groups = [np.array([lo])]
        for a in range(lo + 1, hi, width):
            state_groups.append(np.arange(a, min(a + width, hi)))
        if hi != lo:
            state_groups.append(np.array([hi]))
        groups.append(state_groups)
    return groups


def compact_mstep(counts, groups):
    """Exact Q maximizer: block mass is its total count; density divides by size."""
    counts = np.asarray(counts, dtype=float)
    if (
        counts.ndim != 2
        or len(groups) != len(counts)
        or not np.isfinite(counts).all()
        or np.any(counts < 0)
    ):
        raise ValueError("finite nonnegative state-duration counts required")
    p = np.zeros_like(counts)
    for k, state_groups in enumerate(groups):
        indices = np.concatenate(state_groups)
        if len(np.unique(indices)) != len(indices):
            raise ValueError("overlapping blocks")
        outside = np.ones(counts.shape[1], dtype=bool)
        outside[indices] = False
        if np.any(counts[k, outside] != 0) or counts[k].sum() <= 0:
            raise ValueError("counts outside support or state without exposure")
        for block in state_groups:
            p[k, block] = counts[k, block].sum() / counts[k].sum() / len(block)
    return p


def group_score(counts, p, groups):
    return max(
        abs(float(counts[k, block].sum() - p[k, block].sum() * counts[k].sum()))
        for k, state_groups in enumerate(groups)
        for block in state_groups
    )


def fit_compact(x, support, width, *, iterations=OUTER_UPDATES, tag="compact"):
    groups = partitions(support, width)
    p = support / support.sum(1, keepdims=True)
    scores = emission_scores(x)
    history = []
    start = time.monotonic()
    for update in range(iterations + 1):
        ll, counts = aggregate(scores, p)
        history.append(
            {
                "completed_updates": update,
                "ll": ll,
                "maximum_group_gradient_per_sequence": group_score(counts, p, groups) / len(x),
                "elapsed_seconds": time.monotonic() - start,
            }
        )
        if update and ll < history[-2]["ll"] - 1e-6:
            raise AssertionError("constrained EM objective decreased")
        if update == 0 or update % 5 == 0:
            print(
                f"{tag} width={width} EM={update}/{iterations} seconds={time.monotonic() - start:.1f}",
                flush=True,
            )
        if update < iterations:
            p = compact_mstep(counts, groups)
    return p, history


def select_width(x, support, *, seed, iterations=INNER_UPDATES, widths=WIDTHS):
    """API contains training observations only; no outer test data or true PMF."""
    indices = np.random.default_rng(seed + 600000).permutation(len(x))
    cutoff = 3 * len(x) // 4
    train, validation = indices[:cutoff], indices[cutoff:]
    if len(train) < 1 or len(validation) < 1:
        raise ValueError("inner selection needs training and validation sequences")
    candidates = []
    val_scores = emission_scores(x[validation])
    for width in widths:
        p, history = fit_compact(
            x[train], support, width, iterations=iterations, tag=f"seed={seed} inner"
        )
        ll, _ = aggregate(val_scores, p)
        candidates.append(
            {
                "width": width,
                "validation_ll": ll,
                "validation_nll_per_row": -ll / x[validation].shape[0] / x.shape[1],
                "train_history": history,
                "free_parameters": sum(len(g) - 1 for g in partitions(support, width)),
            }
        )
    chosen = max(candidates, key=lambda c: (c["validation_ll"], -c["width"]))["width"]
    return chosen, {
        "selected_width": chosen,
        "train_indices": train.tolist(),
        "validation_indices": validation.tolist(),
        "candidates": candidates,
        "inner_training_sha256": hashlib.sha256(x[train].tobytes()).hexdigest(),
        "inner_validation_sha256": hashlib.sha256(x[validation].tobytes()).hexdigest(),
    }


def public_check(p, x, labels):
    model = make_model(true_law(), sigma=SIGMA)
    with torch.no_grad():
        model.dist.duration.logits.copy_(torch.tensor(np.log(np.maximum(p, 1e-300))))
    return check_recursion(model, x[:2], labels[:2], sigma=SIGMA)


def run(old, previous):
    seed = old["seed"]
    truth = true_law()
    support = truth > 0
    x, _ = sequences(seed + 400000, count=128, steps=STEPS, sigma=SIGMA)
    if hashlib.sha256(x.tobytes()).hexdigest() != old["small_observation_sha256"]:
        raise AssertionError("small replay mismatch")
    # Inner selection completes before this function constructs or scores outer test data.
    width, selection = select_width(x, support, seed=seed)
    small, small_history = fit_compact(x, support, width, tag=f"seed={seed} small")
    large_x, _ = sequences(seed + 400000, count=LARGE_COUNT, steps=STEPS, sigma=SIGMA)
    if hashlib.sha256(large_x.tobytes()).hexdigest() != old["large_observation_sha256"]:
        raise AssertionError("large replay mismatch")
    np.testing.assert_array_equal(x, large_x[:128])
    large, large_history = fit_compact(large_x, support, width, tag=f"seed={seed} large")
    heldout, labels = sequences(seed + 500000, count=24, steps=STEPS, sigma=SIGMA)
    if hashlib.sha256(heldout.tobytes()).hexdigest() != previous["heldout_observation_sha256"]:
        raise AssertionError("heldout replay mismatch")
    scores = emission_scores(heldout)
    small_test, _ = aggregate(scores, small)
    large_test, _ = aggregate(scores, large)
    small_gain = (small_test - old["initial_heldout_ll"]) / labels.size
    large_gain = (large_test - old["initial_heldout_ll"]) / labels.size
    small_metrics = metrics(small, truth)
    large_metrics = metrics(large, truth)
    return {
        "seed": seed,
        "selected_width": width,
        "selection": selection,
        "labels_used_in_learning_or_selection": False,
        "small_observation_sha256": old["small_observation_sha256"],
        "large_observation_sha256": old["large_observation_sha256"],
        "heldout_observation_sha256": previous["heldout_observation_sha256"],
        "paired_first128_identical": True,
        "free_parameters_by_state": [len(g) - 1 for g in partitions(support, width)],
        "unrestricted_free_parameters_by_state": (support.sum(1) - 1).tolist(),
        "small_history": small_history,
        "large_history": large_history,
        "small_metrics": small_metrics,
        "large_metrics": large_metrics,
        "small_heldout_ll": small_test,
        "large_heldout_ll": large_test,
        "small_heldout_gain_nats_per_row": small_gain,
        "large_heldout_gain_nats_per_row": large_gain,
        "small_heldout_gain_over_unrestricted_nats_per_row": (
            small_test - old["small_em_heldout_ll"]
        )
        / labels.size,
        "large_heldout_gain_over_unrestricted_nats_per_row": (
            large_test - old["large_em_heldout_ll"]
        )
        / labels.size,
        "small_training_ll_difference_to_unrestricted": small_history[-1]["ll"]
        - old["small_em_ll"],
        "large_training_ll_difference_to_unrestricted": large_history[-1]["ll"]
        - old["large_em_history"][-1]["ll"],
        "small_full_gate": bool(small_gain > 0 and small_metrics["fine_pmf_gate"]),
        "large_full_gate": bool(large_gain > 0 and large_metrics["fine_pmf_gate"]),
        "gradient_check": posterior_gradient_check(x[:1], small),
        "small_public_recursion": public_check(small, heldout, labels),
        "large_public_recursion": public_check(large, heldout, labels),
        "small_fitted_pmf": small.tolist(),
        "large_fitted_pmf": large.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    p13 = Path("docs/validation/results/unlabelled-hsmm-step13-native-20261012.json")
    p14 = Path("docs/validation/results/unlabelled-em-step14-native-20261012.json")
    previous = json.loads(p13.read_text())
    old = json.loads(p14.read_text())
    sources = [
        "scripts/compare_compact_duration_em.py",
        "scripts/compare_unlabelled_duration_em.py",
        "scripts/verify_unlabelled_trading_hsmm.py",
        "scripts/verify_isolated_trading_hsmm.py",
        "scripts/diagnose_trading_ground_truth.py",
        "scripts/generate_trading_regime_dataset.py",
        "nhsmm/models/base.py",
        "nhsmm/filtering.py",
        "nhsmm/distributions/default.py",
    ]
    payload = {
        "schema": "nhsmm-compact-duration-inner-selection-v1",
        "execution": {
            "runtime": "openai-native-workspace",
            "base_commit": args.base_commit,
            "date": "2026-10-13",
            "timezone": "Europe/Berlin",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "workers": 1,
            "source_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
            "input_report_sha256": {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (p13, p14)
            },
        },
        "contract": {
            "family": "uniform-density interior blocks at widths4/8/16; first/last known supported lengths separate freely learned atoms; zero outside support",
            "training": "observation-only constrained EM, exact grouped-count Q M-step; uniform-support initial PMF; known Step14 nuisance parameters and censor/start semantics",
            "selection": "seeded96/32 split of the128 small training sequences;20 EM updates per candidate; highest validation marginal LL, smaller width on exact tie; no labels, true PMF values or heldout observations used",
            "evaluation": "refit selected width on128 and1024 paired sequences with30 updates each; unchanged24 heldout sequences; three seeds901/902/903; larger arm uses width selected only from small training observations",
            "gates": "monotonic within1e-6; public all-logit gradient and LL<=1e-7; state probability error<=1e-8; joint heldout gain>0 and each-PMF-TV<=0.12 separately per arm",
            "limits": "oracle known support and nuisance parameters, research-only constraints, piecewise-uniform family approximation bias, finite inner-validation sample and bounded EM; no joint Trading features, production defaults, runtime/artifact masks or live acceptance",
        },
        "runs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for r, prior in zip(old["runs"], previous["runs"], strict=True):
        if r["seed"] != prior["seed"]:
            raise AssertionError("seed alignment")
        result = run(r, prior)
        payload["runs"].append(result)
        args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in [
                        "seed",
                        "selected_width",
                        "small_metrics",
                        "large_metrics",
                        "small_heldout_gain_nats_per_row",
                        "large_heldout_gain_nats_per_row",
                        "small_full_gate",
                        "large_full_gate",
                    ]
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
