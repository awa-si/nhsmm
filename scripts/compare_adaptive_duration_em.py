"""Research-only, full-support duration blocks learned by penalized exact EM."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from scripts.compare_compact_duration_em import fit_compact, public_check
from scripts.compare_unlabelled_duration_em import (
    aggregate,
    emission_scores,
    metrics,
    posterior_gradient_check,
)
from scripts.verify_isolated_trading_hsmm import D, sequences, true_law

PSEUDOCOUNT = 1e-6
PENALTY_FACTORS = (0.5, 1.0, 2.0)
TRAIN_COUNT = 64
HELDOUT_COUNT = 16
STEPS = 300
INNER_UPDATES = 6
OUTER_UPDATES = 10


def adaptive_mstep(counts, penalty, *, pseudocount=PSEUDOCOUNT):
    """Globally maximize count log-density minus penalty per additional block.

    Each state covers every configured age. Prefix dynamic programming chooses
    contiguous blocks; their optimal densities are augmented block mass/length.
    A positive symmetric Dirichlet term prevents irreversible zero-support fits.
    The omitted row-normalization term is constant across all partitions.
    """
    counts = np.asarray(counts, dtype=float)
    if (
        counts.ndim != 2
        or not counts.size
        or not np.isfinite(counts).all()
        or np.any(counts < 0)
        or not np.isfinite(penalty)
        or penalty < 0
        or not np.isfinite(pseudocount)
        or pseudocount <= 0
    ):
        raise ValueError("finite nonnegative counts/penalty and positive pseudocount required")
    pmf = np.empty_like(counts)
    partitions = []
    for k, row in enumerate(counts + pseudocount):
        prefix = np.r_[0.0, np.cumsum(row)]
        best = np.full(len(prefix), -np.inf)
        parent = np.zeros(len(prefix), dtype=int)
        best[0] = penalty  # B blocks incur penalty*(B-1), including B=1.
        for end in range(1, len(prefix)):
            starts = np.arange(end)
            masses = prefix[end] - prefix[starts]
            # Tiny augmented masses can round to zero in prefix subtraction.
            if np.any(masses <= 0):
                raise FloatingPointError("augmented prefix mass lost precision")
            scores = best[:end] + masses * np.log(masses / (end - starts)) - penalty
            parent[end] = int(np.argmax(scores))
            best[end] = scores[parent[end]]
        blocks = []
        end = len(row)
        while end:
            start = int(parent[end])
            blocks.append((start, end))
            pmf[k, start:end] = row[start:end].sum() / row.sum() / (end - start)
            end = start
        partitions.append(list(reversed(blocks)))
    return pmf, partitions


def objective(ll, pmf, blocks, penalty, *, pseudocount=PSEUDOCOUNT):
    return float(ll + pseudocount * np.log(pmf).sum() - penalty * sum(len(b) - 1 for b in blocks))


def fit_adaptive(x, penalty, *, iterations=OUTER_UPDATES, tag="adaptive"):
    if not isinstance(iterations, int) or iterations < 0:
        raise ValueError("nonnegative integer iterations required")
    p = np.full((4, D), 1.0 / D)
    blocks = [[(0, D)] for _ in range(4)]
    scores = emission_scores(x)
    history = []
    start = time.monotonic()
    for update in range(iterations + 1):
        ll, counts = aggregate(scores, p)
        score = objective(ll, p, blocks, penalty)
        history.append(
            {
                "completed_updates": update,
                "ll": ll,
                "penalized_objective": score,
                "blocks_by_state": [len(b) for b in blocks],
                "elapsed_seconds": time.monotonic() - start,
            }
        )
        if update and score < history[-2]["penalized_objective"] - 1e-6:
            raise AssertionError("penalized marginal objective decreased")
        if update == 0 or update % 5 == 0 or update == iterations:
            print(
                f"{tag} EM={update}/{iterations} blocks={history[-1]['blocks_by_state']}",
                flush=True,
            )
        if update < iterations:
            p, blocks = adaptive_mstep(counts, penalty)
    return p, blocks, history


def select_penalty(x, *, seed, iterations=INNER_UPDATES, factors=PENALTY_FACTORS):
    """Only training observations enter partition fitting and inner selection."""
    indices = np.random.default_rng(seed + 600000).permutation(len(x))
    train, validation = indices[: 3 * len(x) // 4], indices[3 * len(x) // 4 :]
    if not len(train) or not len(validation) or not factors:
        raise ValueError("nonempty inner split and penalty candidates required")
    candidates = []
    validation_scores = emission_scores(x[validation])
    for factor in factors:
        penalty = float(factor * np.log(len(train)))
        p, blocks, history = fit_adaptive(x[train], penalty, iterations=iterations, tag="inner")
        ll, _ = aggregate(validation_scores, p)
        candidates.append(
            {
                "factor": factor,
                "penalty": penalty,
                "validation_ll": ll,
                "blocks": blocks,
                "history": history,
            }
        )
    # A likelihood tie prefers the stronger complexity penalty.
    chosen = max(candidates, key=lambda c: (c["validation_ll"], c["factor"]))["factor"]
    return chosen, {
        "train_indices": train.tolist(),
        "validation_indices": validation.tolist(),
        "training_sha256": hashlib.sha256(x[train].tobytes()).hexdigest(),
        "validation_sha256": hashlib.sha256(x[validation].tobytes()).hexdigest(),
        "candidates": candidates,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--seed", type=int, default=1901)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a new output path to preserve earlier evidence")
    torch.set_num_threads(1)
    x, _ = sequences(args.seed + 400000, count=TRAIN_COUNT, steps=STEPS, sigma=2.0)
    factor, selection = select_penalty(x, seed=args.seed)
    penalty = float(factor * np.log(len(x)))
    p, blocks, history = fit_adaptive(x, penalty)
    support = np.ones((4, D), dtype=bool)
    control, control_history = fit_compact(x, support, 16, iterations=OUTER_UPDATES)
    # Construct outer evaluation only after selection and both fits are complete.
    y, labels = sequences(args.seed + 500000, count=HELDOUT_COUNT, steps=STEPS, sigma=2.0)
    baseline, _ = aggregate(emission_scores(y), support / D)
    truth = true_law()  # Evaluation only; no support mask is derived from truth.
    arms = {}
    for name, fitted in (("adaptive", p), ("fixed_width16", control)):
        ll, _ = aggregate(emission_scores(y), fitted)
        m = metrics(fitted, truth)
        gain = (ll - baseline) / labels.size
        arms[name] = {
            "metrics": m,
            "heldout_ll": ll,
            "heldout_gain_nats_per_row": gain,
            "full_gate": bool(gain > 0 and m["fine_pmf_gate"]),
            "pmf": fitted.tolist(),
            "gradient_check": posterior_gradient_check(x[:1], fitted, support=support),
            "public_recursion": public_check(fitted, y, labels, support=support),
        }
    source_paths = [
        "scripts/compare_adaptive_duration_em.py",
        "scripts/compare_compact_duration_em.py",
        "scripts/compare_unlabelled_duration_em.py",
        "scripts/verify_isolated_trading_hsmm.py",
        "scripts/verify_unlabelled_trading_hsmm.py",
        "scripts/diagnose_trading_ground_truth.py",
        "scripts/generate_trading_regime_dataset.py",
        "nhsmm/models/base.py",
        "nhsmm/filtering.py",
        "nhsmm/distributions/default.py",
    ]
    report = {
        "schema": "nhsmm-adaptive-duration-pilot-v1",
        "execution": {
            "runtime": "openai-native-workspace",
            "base_commit": args.base_commit,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "source_sha256": {
                s: hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in source_paths
            },
        },
        "contract": {
            "support": "configured ages1..260 for every state; oracle support disabled",
            "objective": "LL + pseudocount*sum(log PMF) - penalty*sum(block_count-1)",
            "pseudocount": PSEUDOCOUNT,
            "penalty_factors": PENALTY_FACTORS,
            "selection": "training-only48/16 inner split;6 updates/candidate; likelihood selection",
            "refit_penalty": "selected factor times log(refit sequence count)",
            "train_sequences": TRAIN_COUNT,
            "heldout_sequences": HELDOUT_COUNT,
            "steps": STEPS,
            "inner_updates": INNER_UPDATES,
            "outer_updates": OUTER_UPDATES,
            "limits": "single fresh seed, short pilot; known nuisance parameters and fresh starts; not three-seed/1024-sequence/production/Trading acceptance",
            "tv_threshold": 0.12,
        },
        "seed": args.seed,
        "labels_used_in_learning_or_selection": False,
        "training_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
        "heldout_sha256": hashlib.sha256(y.tobytes()).hexdigest(),
        "selected_factor": factor,
        "refit_penalty": penalty,
        "selection": selection,
        "blocks": blocks,
        "adaptive_history": history,
        "fixed_width16_history": control_history,
        "uniform_heldout_ll": baseline,
        "arms": arms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps({name: {k: a[k] for k in ("metrics", "full_gate")} for name, a in arms.items()})
    )


if __name__ == "__main__":
    main()
