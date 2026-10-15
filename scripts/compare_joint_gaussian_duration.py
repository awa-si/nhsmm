"""Frozen three-arm scalar Gaussian/duration EM study; atomic exact resumes."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from scripts.adaptive_width_cap import validate_state
from scripts.block_mass_smoothing import geometry, marginal_objective, smooth_mstep
from scripts.compare_adaptive_sample_size import save
from scripts.compare_fixed_bin_optimizers import SOURCES as PRIOR_SOURCES
from scripts.joint_gaussian_duration import initialize_from_observations, joint_mstep, statistics
from scripts.verify_isolated_trading_hsmm import MEANS, sequences

PROTOCOL = Path("docs/validation/joint-gaussian-duration-protocol.json")
ORDER = ("known_emission_control", "estimated_emission_frozen", "estimated_emission_joint")
SOURCES = tuple(
    dict.fromkeys(
        (
            *PRIOR_SOURCES,
            "scripts/joint_gaussian_duration.py",
            "scripts/compare_joint_gaussian_duration.py",
            "scripts/verify_joint_gaussian_duration.py",
            "scripts/joint_gaussian_evaluation.py",
            "nhsmm/__init__.py",
            "nhsmm/__version__.py",
            "nhsmm/context.py",
            "nhsmm/survival.py",
            "nhsmm/models/__init__.py",
            "nhsmm/transitions.py",
            "nhsmm/inference.py",
            "nhsmm/distributions/__init__.py",
            "nhsmm/artifact.py",
            "nhsmm/convergence.py",
            "nhsmm/diagnostics.py",
            "nhsmm/runtime.py",
            "nhsmm/encoder.py",
            "nhsmm/validation/split_fit.py",
            "nhsmm/validation/alignment.py",
            "nhsmm/validation/context_evidence.py",
            "nhsmm/validation/snapshot.py",
            "nhsmm/validation/__init__.py",
            "nhsmm/validation/api.py",
            "nhsmm/validation/nhsmm.py",
            "nhsmm/validation/state_effect.py",
            "scripts/verify_fixed_bin_optimizers.py",
            str(PROTOCOL),
        )
    )
)


def digest(x):
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def seal(record):
    record["state_sha256"] = hashlib.sha256(
        json.dumps(
            {k: v for k, v in record.items() if k != "state_sha256"},
            sort_keys=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    return record


def expected_payload(seed, source):
    contract = json.loads(PROTOCOL.read_text())
    if seed not in contract["seeds"]:
        raise ValueError("seed outside frozen protocol")
    return {
        "schema": "nhsmm-joint-gaussian-duration-v1",
        "seed": seed,
        "execution": {
            "source_commit": source,
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


def check_record(record, updates, *, complete=False):
    if seal({**record})["state_sha256"] != record.get("state_sha256"):
        raise ValueError("checkpoint state digest")
    h = record["history"]
    if not h or [v["completed_updates"] for v in h] != list(range(len(h))) or len(h) > updates + 1:
        raise ValueError("checkpoint history/budget")
    if complete and len(h) != updates + 1:
        raise ValueError("incomplete history")
    values = np.array([[v["ll"], v["objective"], v["seconds"]] for v in h])
    if (
        not np.isfinite(values).all()
        or (values[:, 2] < 0).any()
        or (np.diff(values[:, 1]) < -1e-6).any()
    ):
        raise ValueError("nonfinite/decreasing EM history")
    blocks = [[list(b) for b in geometry(260, 8)[0]] for _ in range(4)]
    validate_state(record["pmf"], blocks, 8)
    for key in ("means", "variances"):
        a = np.asarray(record[key])
        if a.shape != (4,) or not np.isfinite(a).all() or (key == "variances" and (a < 1e-6).any()):
            raise ValueError("invalid Gaussian state")
        if record[key] != h[-1][key]:
            raise ValueError("terminal Gaussian/history mismatch")
    if record["key"] != "estimated_emission_joint" and any(
        v[k] != record["initial"][k] for v in h for k in ("means", "variances")
    ):
        raise ValueError("frozen emissions changed")


def load_checkpoint(path, expected):
    r = json.loads(path.read_text())
    if any(r.get(k) != expected[k] for k in ("schema", "seed", "execution", "contract")):
        raise ValueError("source/software/seed/contract drift")
    if list(r["phases"]) != list(ORDER[: len(r["phases"])]):
        raise ValueError("invalid completed prefix")
    for key, record in r["phases"].items():
        if record["key"] != key:
            raise ValueError("phase key mismatch")
        check_record(record, r["contract"]["fit"]["em_updates_per_arm"], complete=True)
    if r["active"] is not None:
        if len(r["phases"]) == 3 or r["active"]["key"] != ORDER[len(r["phases"])]:
            raise ValueError("invalid active phase")
        check_record(r["active"], r["contract"]["fit"]["em_updates_per_arm"])
    if r["evaluation"] is not None and (r["active"] is not None or len(r["phases"]) != 3):
        raise ValueError("premature outer evaluation")
    return r


def initial_states(x, sigma=2.0):
    means, variances = initialize_from_observations(x)
    p = np.full((4, 260), 1 / 260).tolist()
    return {
        key: {
            "pmf": p,
            "means": (MEANS if key == ORDER[0] else means).tolist(),
            "variances": (np.full(4, sigma**2) if key == ORDER[0] else variances).tolist(),
        }
        for key in ORDER
    }


def fit_arm(x, key, initial, updates, *, record=None, max_new=None, checkpoint=None):
    """Exact terminal resume; no M-step when the requested budget is complete."""
    if max_new is not None and (
        isinstance(max_new, bool) or not isinstance(max_new, int) or max_new < 1
    ):
        raise ValueError("positive max-new budget required")
    if record is not None:
        check_record(record, updates)
        if record["key"] != key or record["initial"] != initial:
            raise ValueError("initialization identity")
    state = record or {"key": key, "initial": initial, **initial, "history": []}
    p, means, variances = (
        np.asarray(state[k], dtype=float).copy() for k in ("pmf", "means", "variances")
    )
    history = list(state["history"])
    start, used = time.perf_counter(), 0
    s = statistics(x, p, means, variances, chunk=16)
    objective = marginal_objective(s["ll"], p, 0.0)
    if history:
        if (
            abs(s["ll"] - history[-1]["ll"]) > 1e-7
            or abs(objective - history[-1]["objective"]) > 1e-7
        ):
            raise ValueError("terminal likelihood replay mismatch")
    else:
        history.append(
            {
                "completed_updates": 0,
                "ll": s["ll"],
                "objective": objective,
                "means": means.tolist(),
                "variances": variances.tolist(),
                "seconds": 0.0,
            }
        )

    def pack():
        return seal(
            {
                "key": key,
                "initial": initial,
                "pmf": p.tolist(),
                "means": means.tolist(),
                "variances": variances.tolist(),
                "history": history,
            }
        )

    if checkpoint:
        checkpoint(pack())
    while len(history) - 1 < updates and (max_new is None or used < max_new):
        if key == ORDER[2]:
            p, means, variances, certificate = joint_mstep(s)
        else:
            p, certificate = smooth_mstep(s["duration_counts"], 0.0, width=8)
        if certificate["maximum_normalized_anchored_gradient"] > 1e-8:
            raise ValueError("duration M-step certificate")
        s = statistics(x, p, means, variances, chunk=16)
        objective = marginal_objective(s["ll"], p, 0.0)
        if objective < history[-1]["objective"] - 1e-6:
            raise ValueError("joint augmented objective decreased")
        history.append(
            {
                "completed_updates": len(history),
                "ll": s["ll"],
                "objective": objective,
                "means": means.tolist(),
                "variances": variances.tolist(),
                "seconds": (
                    state["history"][-1]["seconds"] + time.perf_counter() - start
                    if state["history"]
                    else time.perf_counter() - start
                ),
            }
        )
        used += 1
        if checkpoint and ((len(history) - 1) % 5 == 0 or len(history) == updates + 1):
            checkpoint(pack())
    return pack(), used


def train(r, x, path, *, max_new=None):
    sha = digest(x)
    if r.get("training_sha256", sha) != sha:
        raise ValueError("training identity")
    r["training_sha256"] = sha
    initials = initial_states(x, r["contract"]["data"]["sigma"])
    updates = r["contract"]["fit"]["em_updates_per_arm"]
    used = 0
    for key in ORDER:
        if key in r["phases"]:
            if r["phases"][key]["initial"] != initials[key]:
                raise ValueError("completed initialization drift")
            continue
        if max_new is not None and used >= max_new:
            save(path, r)
            return False

        def checkpoint(record):
            r["active"] = record
            save(path, r)
            print(
                json.dumps(
                    {"seed": r["seed"], "phase": key, "updates": len(record["history"]) - 1}
                ),
                flush=True,
            )

        record, added = fit_arm(
            x,
            key,
            initials[key],
            updates,
            record=r["active"],
            max_new=None if max_new is None else max_new - used,
            checkpoint=checkpoint,
        )
        used += added
        if len(record["history"]) != updates + 1:
            r["active"] = record
            save(path, r)
            return False
        r["phases"][key], r["active"] = record, None
        save(path, r)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--seed", type=int, required=True, choices=(1930, 1931, 1932))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-updates", type=int)
    a = parser.parse_args()
    if a.max_new_updates is not None and a.max_new_updates < 1:
        parser.error("positive max-new-updates required")
    r = expected_payload(a.seed, a.source_commit)
    if a.resume:
        r = load_checkpoint(a.output, r)
    elif a.output.exists():
        parser.error("preserve evidence; use resume")
    if r["evaluation"] is not None:
        print("completed experiment; no new fitting or evaluation")
        return
    torch.set_num_threads(1)
    d = r["contract"]["data"]
    x, _ = sequences(
        a.seed + d["training_seed_offset"],
        count=d["training_sequences"],
        steps=d["steps"],
        sigma=d["sigma"],
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if train(r, x, a.output, max_new=a.max_new_updates):
        from scripts.joint_gaussian_evaluation import evaluate

        r["evaluation"] = evaluate(r, x)
        save(a.output, r)
        print(json.dumps(r["evaluation"]), flush=True)


if __name__ == "__main__":
    main()
