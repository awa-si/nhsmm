"""Checkpointed paired128/1024 observation-only adaptive duration experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch

from scripts.compare_adaptive_duration_em import PSEUDOCOUNT, fit_adaptive
from scripts.compare_compact_duration_em import public_check
from scripts.compare_unlabelled_duration_em import (
    aggregate,
    emission_scores,
    metrics,
    posterior_gradient_check,
)
from scripts.verify_isolated_trading_hsmm import D, sequences, true_law

SEED = 1903
COUNTS = (128, 1024)
WARM_UPDATES = 3
FIT_UPDATES = 27
FACTOR = 2.0


def save(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_checkpoint(path, expected):
    saved = json.loads(path.read_text())
    for key in ("schema", "execution", "contract"):
        if saved.get(key) != expected[key]:
            raise ValueError(f"resume {key} differs from frozen experiment")
    counts = [r["train_count"] for r in saved["runs"]]
    if counts != list(COUNTS[: len(counts)]):
        raise ValueError("completed arms must be a unique ordered prefix")
    for r in saved["runs"]:
        for key, updates in (("warm_history", WARM_UPDATES), ("history", FIT_UPDATES)):
            if [h["completed_updates"] for h in r[key]] != list(range(updates + 1)):
                raise ValueError("completed arm history is incomplete")
    active = saved.get("active")
    if active is not None:
        if len(counts) == len(COUNTS) or active["train_count"] != COUNTS[len(counts)]:
            raise ValueError("active arm differs from next incomplete count")
        phase = active["phase"]
        if phase not in ("warm", "fit"):
            raise ValueError("invalid active phase")
        h = active["history"]
        limit = WARM_UPDATES if phase == "warm" else FIT_UPDATES
        if (
            not h
            or len(h) > limit + 1
            or [e["completed_updates"] for e in h] != list(range(len(h)))
        ):
            raise ValueError("invalid active history")
        if phase == "fit" and [e["completed_updates"] for e in active["warm_history"]] != list(
            range(WARM_UPDATES + 1)
        ):
            raise ValueError("fit phase requires complete warm history")
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    sources = [
        "scripts/compare_adaptive_sample_size.py",
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
    payload = {
        "schema": "nhsmm-adaptive-sample-size-v1",
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
            "train_counts": list(COUNTS),
            "heldout_count": 24,
            "steps": 300,
            "factor": FACTOR,
            "penalty": "fixed factor2*log(sequence count); frozen from prior inner selection",
            "pseudocount": PSEUDOCOUNT,
            "warm_updates": WARM_UPDATES,
            "fit_updates": FIT_UPDATES,
            "support": "all configured ages1..260; oracle disabled",
            "limits": "one fresh seed, known nuisance parameters, valid starts/censoring; fixed method replication, no renewed inner selection or joint production/Trading acceptance",
            "gates": "positive heldout gain and each-PMF-TV<=0.12; numeric gates unchanged",
        },
        "runs": [],
        "active": None,
    }
    if args.resume:
        payload = load_checkpoint(args.output, payload)
    elif args.output.exists():
        parser.error("existing evidence: use --resume or a new path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = {r["train_count"] for r in payload["runs"]}
    for count in COUNTS:
        if count in completed:
            print(f"count={count} resumed complete", flush=True)
            continue
        x, _ = sequences(SEED + 400000, count=count, steps=300, sigma=2.0)
        train_hash = hashlib.sha256(x.tobytes()).hexdigest()
        active = payload["active"]
        if active is not None and active["training_sha256"] != train_hash:
            raise ValueError("active training replay differs")
        warm_history = None
        if active is None or active["phase"] == "warm":

            def warm_checkpoint(p, blocks, history):
                payload["active"] = {
                    "train_count": count,
                    "phase": "warm",
                    "training_sha256": train_hash,
                    "pmf": p.tolist(),
                    "blocks": blocks,
                    "history": history,
                }
                save(args.output, payload)

            p, blocks, warm_history = fit_adaptive(
                x,
                0.0,
                iterations=WARM_UPDATES,
                tag=f"count={count} warm",
                initial_state=(active["pmf"], active["blocks"]) if active else None,
                resume_history=active["history"] if active else None,
                checkpoint=warm_checkpoint,
            )
            active = None
        else:
            p, blocks = np.asarray(active["pmf"]), active["blocks"]
            warm_history = active["warm_history"]
        penalty = float(FACTOR * np.log(count))

        def fit_checkpoint(p, blocks, history):
            payload["active"] = {
                "train_count": count,
                "phase": "fit",
                "training_sha256": train_hash,
                "warm_history": warm_history,
                "pmf": p.tolist(),
                "blocks": blocks,
                "history": history,
            }
            save(args.output, payload)

        p, blocks, history = fit_adaptive(
            x,
            penalty,
            iterations=FIT_UPDATES,
            initial_state=(p, blocks),
            resume_history=active["history"] if active else None,
            checkpoint=fit_checkpoint,
            tag=f"count={count} adaptive",
        )
        # Ground truth and outer observations are used only after this fit completes.
        y, labels = sequences(SEED + 500000, count=24, steps=300, sigma=2.0)
        support = np.ones((4, D), dtype=bool)
        scores = emission_scores(y)
        baseline, _ = aggregate(scores, support / D)
        ll, _ = aggregate(scores, p)
        m = metrics(p, true_law())
        gain = (ll - baseline) / labels.size
        prefix_hash = hashlib.sha256(x[:128].tobytes()).hexdigest()
        if payload["runs"] and prefix_hash != payload["runs"][0]["training_sha256"]:
            raise AssertionError("paired128-prefix differs")
        result = {
            "train_count": count,
            "training_sha256": train_hash,
            "first128_sha256": prefix_hash,
            "heldout_sha256": hashlib.sha256(y.tobytes()).hexdigest(),
            "labels_used_in_learning": False,
            "penalty": penalty,
            "warm_history": warm_history,
            "history": history,
            "pmf": p.tolist(),
            "blocks": blocks,
            "metrics": m,
            "uniform_heldout_ll": baseline,
            "heldout_ll": ll,
            "heldout_gain_nats_per_row": gain,
            "full_gate": bool(gain > 0 and m["fine_pmf_gate"]),
            "gradient_check": posterior_gradient_check(x[:1], p, support=support),
            "public_recursion": public_check(p, y, labels, support=support),
        }
        payload["runs"].append(result)
        payload["active"] = None
        save(args.output, payload)
        print(
            json.dumps({k: result[k] for k in ("train_count", "metrics", "full_gate")}), flush=True
        )


if __name__ == "__main__":
    main()
