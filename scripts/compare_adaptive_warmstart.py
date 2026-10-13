"""Fresh-data paired penalty/warm-start pilot with matched EM path budgets."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch

from scripts.compare_adaptive_duration_em import PSEUDOCOUNT, fit_adaptive
from scripts.compare_compact_duration_em import fit_compact, public_check
from scripts.compare_unlabelled_duration_em import (
    aggregate,
    emission_scores,
    metrics,
    posterior_gradient_check,
)
from scripts.verify_isolated_trading_hsmm import D, sequences, true_law

FACTORS = (0.125, 0.5, 2.0)
WARM_UPDATES = 3
INNER_TOTAL = 9
OUTER_TOTAL = 12
SEED = 1902


def paired_selection(x, *, seed=SEED):
    indices = np.random.default_rng(seed + 600000).permutation(len(x))
    train, validation = indices[: 3 * len(x) // 4], indices[3 * len(x) // 4 :]
    if not len(train) or not len(validation):
        raise ValueError("nonempty inner split required")
    warm_p, warm_blocks, warm_history = fit_adaptive(
        x[train], 0.0, iterations=WARM_UPDATES, tag="shared inner warm"
    )
    candidates = []
    validation_scores = emission_scores(x[validation])
    selected = {}
    for mode in ("uniform", "warm"):
        for factor in FACTORS:
            penalty = float(factor * np.log(len(train)))
            p, blocks, history = fit_adaptive(
                x[train],
                penalty,
                iterations=INNER_TOTAL - (WARM_UPDATES if mode == "warm" else 0),
                initial_state=(warm_p, warm_blocks) if mode == "warm" else None,
                tag=f"inner {mode} factor={factor}",
            )
            ll, _ = aggregate(validation_scores, p)
            candidates.append(
                {
                    "mode": mode,
                    "factor": factor,
                    "penalty": penalty,
                    "validation_ll": ll,
                    "blocks": blocks,
                    "history": history,
                }
            )
        selected[mode] = max(
            (c for c in candidates if c["mode"] == mode),
            key=lambda c: (c["validation_ll"], c["factor"]),
        )["factor"]
    return {
        "train_indices": train.tolist(),
        "validation_indices": validation.tolist(),
        "train_sha256": hashlib.sha256(x[train].tobytes()).hexdigest(),
        "validation_sha256": hashlib.sha256(x[validation].tobytes()).hexdigest(),
        "shared_warm_pmf": warm_p.tolist(),
        "shared_warm_blocks": warm_blocks,
        "shared_warm_history": warm_history,
        "selected_factor_by_mode": selected,
        "candidates": candidates,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a new path to preserve earlier evidence")
    torch.set_num_threads(1)
    x, _ = sequences(SEED + 400000, count=64, steps=300, sigma=2.0)
    selection = paired_selection(x)
    warm_p, warm_blocks, warm_history = fit_adaptive(
        x, 0.0, iterations=WARM_UPDATES, tag="refit warm"
    )
    fits = {}
    for mode in ("uniform", "warm"):
        factor = selection["selected_factor_by_mode"][mode]
        penalty = float(factor * np.log(len(x)))
        p, blocks, history = fit_adaptive(
            x,
            penalty,
            iterations=OUTER_TOTAL - (WARM_UPDATES if mode == "warm" else 0),
            initial_state=(warm_p, warm_blocks) if mode == "warm" else None,
            tag=f"refit {mode}",
        )
        fits[mode] = {"pmf": p, "blocks": blocks, "penalty": penalty, "history": history}
    support = np.ones((4, D), dtype=bool)
    p, history = fit_compact(x, support, 16, iterations=OUTER_TOTAL)
    fits["fixed_width16"] = {"pmf": p, "history": history}
    # All selection and fitting finish before outer observations or truth exist here.
    y, labels = sequences(SEED + 500000, count=16, steps=300, sigma=2.0)
    scores = emission_scores(y)
    baseline, _ = aggregate(scores, support / D)
    truth = true_law()
    for arm in fits.values():
        p = arm["pmf"]
        ll, _ = aggregate(scores, p)
        m = metrics(p, truth)
        gain = (ll - baseline) / labels.size
        arm.update(
            {
                "metrics": m,
                "heldout_ll": ll,
                "heldout_gain_nats_per_row": gain,
                "full_gate": bool(gain > 0 and m["fine_pmf_gate"]),
                "gradient_check": posterior_gradient_check(x[:1], p, support=support),
                "public_recursion": public_check(p, y, labels, support=support),
                "pmf": p.tolist(),
            }
        )
    sources = [
        "scripts/compare_adaptive_warmstart.py",
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
        "schema": "nhsmm-adaptive-warmstart-pilot-v1",
        "execution": {
            "runtime": "openai-native-workspace",
            "base_commit": args.base_commit,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "source_sha256": {s: hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in sources},
        },
        "contract": {
            "seed": SEED,
            "support": "configured ages1..260 every state; oracle disabled",
            "pseudocount": PSEUDOCOUNT,
            "factors": FACTORS,
            "train_sequences": 64,
            "heldout_sequences": 16,
            "steps": 300,
            "inner_total_updates_per_path": INNER_TOTAL,
            "outer_total_updates_per_path": OUTER_TOTAL,
            "warm_updates": WARM_UPDATES,
            "actual_msteps_with_shared_inner_warm": 84,
            "selection": "same48/16 observations; best marginal validation LL separately per initialization; stronger factor on tie; no outer data or truth",
            "warm_start": "three unpenalized EM updates with the same positive pseudocount; learn only from each fit's own training observations; reuse inner warm state across factors",
            "gates": "unchanged positive heldout gain and each-state PMF TV<=0.12; numerical checks separate",
            "limits": "short single fresh seed, known nuisance parameters; no large-sample/multi-seed/production/Trading acceptance",
        },
        "labels_used_in_learning_or_selection": False,
        "training_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
        "heldout_sha256": hashlib.sha256(y.tobytes()).hexdigest(),
        "selection": selection,
        "refit_shared_warm_pmf": warm_p.tolist(),
        "refit_shared_warm_blocks": warm_blocks,
        "refit_shared_warm_history": warm_history,
        "uniform_heldout_ll": baseline,
        "arms": fits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                name: {k: arm[k] for k in ("metrics", "heldout_gain_nats_per_row", "full_gate")}
                for name, arm in fits.items()
            }
        )
    )


if __name__ == "__main__":
    main()
