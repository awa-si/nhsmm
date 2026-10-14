"""Checkpointed matched EM/Adam recovery study; known nuisance, oracle OFF."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from scripts.adaptive_width_cap import validate_state
from scripts.block_mass_smoothing import geometry, marginal_objective
from scripts.compare_adaptive_sample_size import save
from scripts.compare_adaptive_width_cap import evaluate as paired_evaluate
from scripts.compare_block_mass_smoothing import SOURCES as PRIOR_SOURCES
from scripts.compare_block_mass_smoothing import check_history, digest, fit_fixed
from scripts.compare_unlabelled_duration_em import aggregate, emission_scores
from scripts.fixed_bin_posterior_score import density, score_gradient
from scripts.verify_isolated_trading_hsmm import D, sequences

PROTOCOL = Path("docs/validation/fixed-bin-optimizer-recovery-protocol.json")
ORDER = ("warm", "fixed_bin_em", "fixed_bin_adam")
SOURCES = tuple(
    dict.fromkeys(
        (
            *PRIOR_SOURCES,
            "nhsmm/config.py",
            "scripts/fixed_bin_posterior_score.py",
            "scripts/compare_fixed_bin_optimizers.py",
            str(PROTOCOL),
        )
    )
)


def expected_payload(seed, source):
    contract = json.loads(PROTOCOL.read_text())
    if seed not in contract["seeds"]:
        raise ValueError("seed outside frozen contract")
    return {
        "schema": "nhsmm-fixed-bin-optimizer-recovery-v1",
        "seed": seed,
        "execution": {
            "source_commit": source,
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
    if any(r.get(k) != expected[k] for k in ("schema", "seed", "execution", "contract")):
        raise ValueError("source/software/seed/contract drift")
    if list(r["phases"]) != list(ORDER[: len(r["phases"])]):
        raise ValueError("invalid completed prefix")
    if r["active"] is not None and (
        len(r["phases"]) == 3 or r["active"].get("key") != ORDER[len(r["phases"])]
    ):
        raise ValueError("invalid active phase")
    if r["evaluation"] is not None and (
        r["active"] is not None or list(r["phases"]) != list(ORDER)
    ):
        raise ValueError("premature outer evaluation")
    return r


def check_adam(record, updates, complete=False):
    h = record["history"]
    if (
        not h
        or [v["completed_updates"] for v in h] != list(range(len(h)))
        or len(h) > updates + 1
        or (complete and len(h) != updates + 1)
    ):
        raise ValueError("invalid Adam history/budget")
    if any(
        not np.isfinite(v[k])
        for v in h
        for k in ("ll", "penalized_objective", "gradient_max", "elapsed_seconds")
    ):
        raise ValueError("nonfinite Adam history")
    step = len(h) - 1
    state = record["adam_state"]
    z = np.asarray(record["logits"], dtype=float)
    if (
        z.shape != (4, 32)
        or not np.isfinite(z).all()
        or isinstance(state["step"], bool)
        or not isinstance(state["step"], int)
        or state["step"] != step
    ):
        raise ValueError("Adam logits/step identity")
    for key in ("exp_avg", "exp_avg_sq"):
        a = np.asarray(state[key], dtype=float)
        if (
            a.shape != z.shape
            or not np.isfinite(a).all()
            or (key == "exp_avg_sq" and (a < 0).any())
        ):
            raise ValueError("invalid Adam moment")
        if step == 0 and (a != 0).any():
            raise ValueError("nonzero initial Adam moment")
    restored = density(torch.tensor(z, dtype=torch.float64)).numpy()
    if not np.allclose(restored, record["pmf"], atol=1e-15, rtol=0):
        raise ValueError("Adam logits do not reproduce PMF")


def fit_adam(x, initial, updates, method, *, record=None, checkpoint=None):
    blocks, _, _ = geometry(D, 8)
    parts = [[list(b) for b in blocks] for _ in range(4)]
    initial = np.asarray(initial, dtype=float)
    validate_state(initial, parts, 8)
    z = np.log(initial[:, [a for a, _ in blocks]])
    z = z[:, :-1] - z[:, -1:]
    history = []
    offset = 0.0
    if record is not None:
        check_adam(record, updates)
        z = np.asarray(record["logits"])
        history = [dict(h) for h in record["history"][:-1]]
        offset = record["history"][-1]["elapsed_seconds"]
    parameter = torch.nn.Parameter(torch.tensor(z, dtype=torch.float64))
    optimizer = torch.optim.Adam(
        [parameter], lr=method["adam_lr"], betas=tuple(method["adam_betas"]), eps=method["adam_eps"]
    )
    first = len(record["history"]) - 1 if record else 0
    if record is not None and first:
        optimizer.state[parameter] = {
            "step": torch.tensor(float(first)),
            "exp_avg": torch.tensor(record["adam_state"]["exp_avg"], dtype=torch.float64),
            "exp_avg_sq": torch.tensor(record["adam_state"]["exp_avg_sq"], dtype=torch.float64),
        }
    scores = emission_scores(x)
    start = time.monotonic()
    for update in range(first, updates + 1):
        p = density(parameter).detach().numpy()
        ll, counts = aggregate(scores, p)
        objective = marginal_objective(ll, p, 0.0)
        gradient = score_gradient(parameter, counts, pseudocount=method["pseudocount_per_age"])
        if (
            record is not None
            and update == first
            and (
                abs(ll - record["history"][-1]["ll"]) > 1e-7
                or abs(objective - record["history"][-1]["penalized_objective"]) > 1e-7
            )
        ):
            raise ValueError("Adam terminal evidence replay mismatch")
        history.append(
            {
                "completed_updates": update,
                "ll": ll,
                "penalized_objective": objective,
                "gradient_max": float(gradient.abs().max()),
                "elapsed_seconds": offset + time.monotonic() - start,
            }
        )
        state = optimizer.state.get(parameter)
        payload = {
            "pmf": p.tolist(),
            "logits": parameter.detach().tolist(),
            "history": history,
            "adam_state": {
                "step": update,
                "exp_avg": state["exp_avg"].tolist() if state else np.zeros((4, 32)).tolist(),
                "exp_avg_sq": state["exp_avg_sq"].tolist() if state else np.zeros((4, 32)).tolist(),
            },
        }
        if update == first or update % 5 == 0 or update == updates:
            print(f"Adam={update}/{updates}", flush=True)
            if checkpoint is not None:
                checkpoint(payload)
        if update < updates:
            optimizer.zero_grad()
            parameter.grad = -gradient
            torch.nn.utils.clip_grad_norm_([parameter], method["adam_gradient_clip_norm"])
            optimizer.step()
    return payload


def train(r, x, path, max_phases=None):
    d, m = r["contract"]["data"], r["contract"]["method"]
    if (
        x.shape != (d["training_sequences"], d["steps"], 1)
        or m["pseudocount_per_age"] != 1e-6
        or m["smoothing_strength"] != 0.0
    ):
        raise ValueError("training shape/target mismatch")
    sha = digest(x)
    if r.get("training_sha256", sha) != sha:
        raise ValueError("training hash drift")
    r["training_sha256"] = sha
    blocks = [[list(b) for b in geometry(D, 8)[0]] for _ in range(4)]
    used, warm = 0, None
    for key in ORDER:
        completed = key in r["phases"]
        if not completed and max_phases is not None and used >= max_phases:
            save(path, r)
            return False
        updates = m["shared_warm_em_updates"] if key == "warm" else m["updates_per_arm"]
        identity = {"updates": updates, "strength": 0.0, "blocks": blocks, "training_sha256": sha}
        record = r["phases"].get(key) if completed else r["active"]
        if record is not None:
            if any(record.get(k) != v for k, v in identity.items()) or (
                not completed and record["key"] != key
            ):
                raise ValueError("phase identity mismatch")
            validate_state(record["pmf"], blocks, 8)
            if key == "fixed_bin_adam":
                check_adam(record, updates, completed)
            else:
                check_history(record["history"], updates, completed)
        initial = record["pmf"] if record else warm

        def checkpoint(payload):
            r["active"] = {"key": key, **identity, **payload}
            save(path, r)

        if key == "fixed_bin_adam":
            result = fit_adam(
                x, initial, updates, m, record=record, checkpoint=None if completed else checkpoint
            )
        else:
            p, h = fit_fixed(
                x,
                0.0,
                iterations=updates,
                solver_controls={
                    "width": 8,
                    "pseudocount": 1e-6,
                    "max_iterations": 1000,
                    "max_evaluations": 2000,
                    "gradient_tolerance": 1e-8,
                },
                initial_pmf=initial,
                resume_history=record["history"] if record else None,
                checkpoint=(
                    None
                    if completed
                    else lambda p, h: checkpoint({"pmf": p.tolist(), "history": h})
                ),
                tag=key,
            )
            result = {"pmf": p.tolist(), "history": h}
        if not completed:
            r["phases"][key] = {**identity, **result}
            r["active"] = None
            used += 1
            save(path, r)
        if key == "warm":
            warm = np.asarray(result["pmf"])
    return True


def evaluate(r, x):
    if r["active"] is not None or list(r["phases"]) != list(ORDER):
        raise ValueError("outer evaluation before all fits")
    view = {
        **r,
        "phases": {
            "warm": r["phases"]["warm"],
            "unrestricted": r["phases"]["fixed_bin_em"],
            "width8": r["phases"]["fixed_bin_adam"],
        },
    }
    e = paired_evaluate(view, x)
    data = r["contract"]["data"]
    replay, labels = sequences(
        r["seed"] + data["training_seed_offset"],
        count=data["training_sequences"],
        steps=data["steps"],
        sigma=data["sigma"],
    )
    if digest(replay) != r["training_sha256"]:
        raise ValueError("post-fit training exposure data drift")
    completed, censored = np.zeros(4, dtype=int), np.zeros(4, dtype=int)
    for row in labels:
        ends = np.flatnonzero(row[1:] != row[:-1]) + 1
        for end in ends:
            completed[row[end - 1]] += 1
        censored[row[-1]] += 1
    return {
        "heldout_sha256": e["heldout_sha256"],
        "uniform_heldout_ll": e["uniform_heldout_ll"],
        "arms": {"fixed_bin_em": e["arms"]["unrestricted"], "fixed_bin_adam": e["arms"]["width8"]},
        "adam_minus_em_heldout_ll": e["width8_minus_unrestricted_heldout_ll"],
        "actual_updates": e["actual_msteps"],
        "actual_em_msteps": sum(r["phases"][k]["updates"] for k in ("warm", "fixed_bin_em")),
        "actual_adam_updates": r["phases"]["fixed_bin_adam"]["updates"],
        "labels_used_in_learning_or_selection": False,
        "post_fit_training_episode_exposure": {
            "completed_per_state": completed.tolist(),
            "terminal_censored_per_state": censored.tolist(),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--seed", required=True, type=int, choices=(1922, 1923, 1924))
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
        parser.error("preserve evidence: use resume")
    if r["evaluation"] is not None:
        print("completed experiment; no new fitting or evaluation", flush=True)
        return
    torch.set_num_threads(1)
    d = r["contract"]["data"]
    x, _ = sequences(
        args.seed + d["training_seed_offset"],
        count=d["training_sequences"],
        steps=d["steps"],
        sigma=d["sigma"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save(args.output, r)
    if train(r, x, args.output, args.max_phases):
        r["evaluation"] = evaluate(r, x)
        save(args.output, r)
        print(json.dumps(r["evaluation"]), flush=True)


if __name__ == "__main__":
    main()
