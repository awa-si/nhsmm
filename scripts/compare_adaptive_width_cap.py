"""Checkpointed matched width-cap pilot; research, oracle OFF."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch

from scripts.adaptive_width_cap import fit_capped, refine_state, validate_state
from scripts.compare_adaptive_duration_em import fit_adaptive
from scripts.compare_adaptive_large_selection import SOURCES as PRIOR_SOURCES
from scripts.compare_adaptive_large_selection import digest
from scripts.compare_adaptive_sample_size import save
from scripts.compare_compact_duration_em import public_check
from scripts.compare_unlabelled_duration_em import (
    aggregate,
    emission_scores,
    metrics,
    posterior_gradient_check,
)
from scripts.verify_isolated_trading_hsmm import D, sequences, true_law

PROTOCOL = Path("docs/validation/adaptive-width-cap-protocol.json")
SOURCES = tuple(
    dict.fromkeys(
        (
            *PRIOR_SOURCES,
            "scripts/adaptive_width_cap.py",
            "scripts/compare_adaptive_width_cap.py",
            str(PROTOCOL),
        )
    )
)


def expected_payload(seed, source_commit):
    contract = json.loads(PROTOCOL.read_text())
    if seed not in contract["seeds"]:
        raise ValueError("seed outside frozen protocol")
    return {
        "schema": "nhsmm-adaptive-width-cap-v1",
        "seed": seed,
        "execution": {
            "source_commit": source_commit,
            "runtime": "openai-native-workspace",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "source_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in SOURCES},
        },
        "contract": contract,
        "phases": {},
        "active": None,
        "evaluation": None,
    }


def load_checkpoint(path, expected):
    r = json.loads(path.read_text())
    for key in ("schema", "seed", "execution", "contract"):
        if r.get(key) != expected[key]:
            raise ValueError(f"resume {key} differs")
    order = ("warm", "unrestricted", "width8")
    if list(r["phases"]) != list(order[: len(r["phases"])]):
        raise ValueError("invalid completed phase prefix")
    if r["active"] is not None and (
        len(r["phases"]) == len(order) or r["active"]["key"] != order[len(r["phases"])]
    ):
        raise ValueError("invalid active phase")
    return r


def train(r, x, path, max_phases=None):
    """Only training observations enter this function; no labels or true PMFs."""
    c = r["contract"]
    m = c["method"]
    data = c["data"]
    if x.shape != (data["training_sequences"], data["steps"], 1):
        raise ValueError("training dimensions differ")
    sha = digest(x)
    if "training_sha256" in r and r["training_sha256"] != sha:
        raise ValueError("training replay differs")
    r["training_sha256"] = sha
    used = 0
    warm = None
    for key in ("warm", "unrestricted", "width8"):
        completed = key in r["phases"]
        if not completed and max_phases is not None and used >= max_phases:
            save(path, r)
            return False
        updates = m["warm_updates"] if key == "warm" else m["penalized_updates"]
        penalty = 0.0 if key == "warm" else float(m["factor"] * np.log(len(x)))
        width = m["maximum_block_width"] if key == "width8" else D
        identity = {
            "updates": updates,
            "penalty": penalty,
            "max_width": width,
            "training_sha256": sha,
        }
        record = r["phases"].get(key) if completed else r["active"]
        if record is not None:
            if any(record.get(k) != v for k, v in identity.items()) or (
                not completed and record.get("key") != key
            ):
                raise ValueError("phase identity differs")
            validate_state(record["pmf"], record["blocks"], width)
            if completed and [h["completed_updates"] for h in record["history"]] != list(
                range(updates + 1)
            ):
                raise ValueError("incomplete saved phase")
            history = record["history"]
            if (
                not history
                or [h["completed_updates"] for h in history] != list(range(len(history)))
                or len(history) > updates + 1
                or not all(
                    np.isfinite(h[k])
                    for h in history
                    for k in ("ll", "penalized_objective", "elapsed_seconds")
                )
                or any(
                    b["penalized_objective"] < a["penalized_objective"] - 1e-6
                    for a, b in zip(history, history[1:])
                )
                or history[-1]["blocks_by_state"] != [len(b) for b in record["blocks"]]
            ):
                raise ValueError("invalid saved phase history")
        initial = (
            (record["pmf"], record["blocks"])
            if record
            else (refine_state(*warm, width) if key == "width8" else warm)
        )

        def checkpoint(p, blocks, history):
            r["active"] = {
                "key": key,
                **identity,
                "pmf": p.tolist(),
                "blocks": blocks,
                "history": history,
            }
            save(path, r)

        kwargs = {
            "iterations": updates,
            "initial_state": initial,
            "resume_history": record["history"] if record else None,
            "checkpoint": None if completed else checkpoint,
            "tag": key,
        }
        p, blocks, history = (
            fit_capped(x, penalty, width, **kwargs)
            if key == "width8"
            else fit_adaptive(x, penalty, **kwargs)
        )
        if not completed:
            r["phases"][key] = {**identity, "pmf": p.tolist(), "blocks": blocks, "history": history}
            r["active"] = None
            save(path, r)
            used += 1
        if key == "warm":
            warm = (p, blocks)
    return True


def evaluate(r, x):
    if r["active"] is not None or list(r["phases"]) != ["warm", "unrestricted", "width8"]:
        raise ValueError("outer evaluation requires both completed fits")
    data = r["contract"]["data"]
    y, labels = sequences(
        r["seed"] + data["outer_seed_offset"],
        count=data["outer_sequences"],
        steps=data["steps"],
        sigma=data["sigma"],
    )
    scores = emission_scores(y)
    truth = true_law()
    support = np.ones((4, D), dtype=bool)
    baseline, _ = aggregate(scores, support / D)
    arms = {}
    for key in ("unrestricted", "width8"):
        record = r["phases"][key]
        p = np.asarray(record["pmf"])
        ll, _ = aggregate(scores, p)
        m = metrics(p, truth)
        gain = (ll - baseline) / labels.size
        projection = np.zeros_like(p)
        lower = []
        for k, blocks in enumerate(record["blocks"]):
            lower.append(
                float(
                    sum(np.abs(truth[k, a:b] - np.median(truth[k, a:b])).sum() for a, b in blocks)
                    / 2
                )
            )
            for a, b in blocks:
                projection[k, a:b] = truth[k, a:b].sum() / (b - a)
        arms[key] = {
            "metrics": m,
            "heldout_ll": ll,
            "heldout_gain_nats_per_row": gain,
            "full_gate": bool(gain > 0 and m["fine_pmf_gate"]),
            "blocks": [len(b) for b in record["blocks"]],
            "learned_partition_tv_lower_bound": lower,
            "forward_kl_projection_tv": (np.abs(projection - truth).sum(1) / 2).tolist(),
            "gradient_check": posterior_gradient_check(x[:1], p, support=support),
            "public_recursion": public_check(p, y, labels, support=support),
        }
    return {
        "heldout_sha256": digest(y),
        "uniform_heldout_ll": baseline,
        "arms": arms,
        "width8_minus_unrestricted_heldout_ll": arms["width8"]["heldout_ll"]
        - arms["unrestricted"]["heldout_ll"],
        "actual_msteps": sum(z["updates"] for z in r["phases"].values()),
        "labels_used_in_learning_or_selection": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--seed", required=True, type=int, choices=(1909, 1910, 1911))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-phases", type=int)
    args = parser.parse_args()
    if args.max_phases is not None and args.max_phases < 1:
        parser.error("positive max-phases required")
    r = expected_payload(args.seed, args.source_commit)
    if args.resume:
        r = load_checkpoint(args.output, r)
    elif args.output.exists():
        parser.error("preserve existing evidence: use --resume")
    if r["evaluation"] is not None:
        print("completed experiment; no new fitting or evaluation", flush=True)
        return
    torch.set_num_threads(1)
    data = r["contract"]["data"]
    x, _ = sequences(
        args.seed + data["training_seed_offset"],
        count=data["training_sequences"],
        steps=data["steps"],
        sigma=data["sigma"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if train(r, x, args.output, args.max_phases):
        r["evaluation"] = evaluate(r, x)
        save(args.output, r)
        print(json.dumps(r["evaluation"]), flush=True)
    else:
        print(f"phase boundary: completed {list(r['phases'])}", flush=True)


if __name__ == "__main__":
    main()
