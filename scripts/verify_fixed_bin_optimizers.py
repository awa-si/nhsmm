"""Verify frozen source/data/checkpoints and independently replay final evidence."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from scripts.adaptive_width_cap import validate_state
from scripts.block_mass_smoothing import geometry, marginal_objective
from scripts.compare_fixed_bin_optimizers import (
    ORDER,
    check_adam,
    check_history,
    digest,
    evaluate,
    expected_payload,
    fit_adam,
    fit_fixed,
)
from scripts.verify_isolated_trading_hsmm import sequences


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(actual, expected):
    if isinstance(actual, dict):
        require(actual.keys() == expected.keys(), "replay keys differ")
        for key in actual:
            equal(actual[key], expected[key])
    elif isinstance(actual, list):
        require(len(actual) == len(expected), "replay list length")
        for a, b in zip(actual, expected, strict=True):
            equal(a, b)
    elif isinstance(actual, (float, int)) and not isinstance(actual, bool):
        require(
            np.isfinite(actual) and np.isfinite(expected) and abs(actual - expected) <= 1e-7,
            "replay numerical mismatch",
        )
    else:
        require(actual == expected, "replay value differs")


def inspect(raw, source, *, partial=False):
    r = json.loads(raw)
    expected = expected_payload(r["seed"], source)
    require(
        all(r[k] == expected[k] for k in ("schema", "seed", "execution", "contract")),
        "frozen header identity",
    )
    require(list(r["phases"]) == list(ORDER[: len(r["phases"])]), "completed prefix")
    d, m = r["contract"]["data"], r["contract"]["method"]
    x, _ = sequences(
        r["seed"] + d["training_seed_offset"],
        count=d["training_sequences"],
        steps=d["steps"],
        sigma=d["sigma"],
    )
    require(digest(x) == r["training_sha256"], "training hash")
    records = dict(r["phases"])
    if r["active"] is not None:
        require(
            partial and len(records) < 3 and r["active"]["key"] == ORDER[len(records)],
            "active phase",
        )
        records[r["active"]["key"]] = r["active"]
    blocks = [[list(b) for b in geometry(260, 8)[0]] for _ in range(4)]
    for key, record in records.items():
        updates = m["shared_warm_em_updates"] if key == "warm" else m["updates_per_arm"]
        require(
            record["updates"] == updates
            and record["strength"] == 0.0
            and record["training_sha256"] == digest(x),
            "phase identity",
        )
        require(record["blocks"] == blocks, "fixed grid")
        validate_state(record["pmf"], blocks, 8)
        if key == "fixed_bin_adam":
            check_adam(record, updates, key in r["phases"])
        else:
            check_history(record["history"], updates, key in r["phases"])
        if key != "warm":
            warm = r["phases"]["warm"]
            ll = warm["history"][-1]["ll"]
            require(abs(record["history"][0]["ll"] - ll) <= 1e-7, "shared initial likelihood")
            objective = marginal_objective(ll, np.asarray(warm["pmf"]), 0.0)
            require(
                abs(record["history"][0]["penalized_objective"] - objective) <= 1e-7,
                "shared initial objective",
            )
    if partial:
        require(r["evaluation"] is None, "partial report already evaluated")
    else:
        require(
            r["active"] is None
            and list(r["phases"]) == list(ORDER)
            and r["evaluation"] is not None,
            "incomplete final report",
        )
    return r, x


def verify(path, source):
    torch.set_num_threads(1)
    raw = path.read_bytes()
    r, x = inspect(raw, source)
    for key in ("warm", "fixed_bin_em"):
        z = r["phases"][key]
        fit_fixed(
            x,
            0.0,
            iterations=z["updates"],
            solver_controls={
                "width": 8,
                "pseudocount": 1e-6,
                "max_iterations": 1000,
                "max_evaluations": 2000,
                "gradient_tolerance": 1e-8,
            },
            initial_pmf=z["pmf"],
            resume_history=z["history"],
            tag="independent " + key,
        )
    z = r["phases"]["fixed_bin_adam"]
    replay = fit_adam(x, z["pmf"], z["updates"], r["contract"]["method"], record=z)
    require(replay["adam_state"] == z["adam_state"], "terminal replay changed moments")
    require(
        abs(replay["history"][-1]["gradient_max"] - z["history"][-1]["gradient_max"]) <= 1e-7,
        "terminal Adam gradient evidence",
    )
    e = evaluate(r, x)
    equal(e, r["evaluation"])
    require(
        e["actual_updates"] == 57
        and e["actual_em_msteps"] == 30
        and e["actual_adam_updates"] == 27,
        "actual budget",
    )
    for arm in e["arms"].values():
        require(
            max(arm["gradient_check"].values()) <= 1e-7
            and arm["public_recursion"]["ll_error"] <= 1e-7
            and arm["public_recursion"]["maximum_state_probability_error"] <= 1e-8,
            "public numerical gates",
        )
        require(
            arm["full_gate"]
            == (max(arm["metrics"]["pmf_tv"]) <= 0.12 and arm["heldout_gain_nats_per_row"] > 0),
            "statistical gate arithmetic",
        )
    noop = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.compare_fixed_bin_optimizers",
            "--source-commit",
            source,
            "--seed",
            str(r["seed"]),
            "--output",
            str(path),
            "--resume",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    require(
        noop.returncode == 0
        and "completed experiment; no new fitting or evaluation" in noop.stdout
        and path.read_bytes() == raw,
        "completed no-op",
    )
    return {
        "schema": "nhsmm-fixed-bin-optimizer-verification-v1",
        "verified": True,
        "seed": r["seed"],
        "source_commit": source,
        "input_report_sha256": hashlib.sha256(raw).hexdigest(),
        "source_files": len(r["execution"]["source_sha256"]),
        "terminal_replay_optimizer_updates": 0,
        "source_software_contract_data_phase_history_grid_moments_pass": True,
        "numerical_replay_pass": True,
        "completed_resume_noop": True,
        "evaluation": e,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--snapshot", action="store_true")
    args = parser.parse_args()
    if args.snapshot:
        raw = args.input.read_bytes()
        r, _ = inspect(raw, args.source_commit, partial=True)
        args.output.write_bytes(raw)
        active = r["active"]
        print(
            json.dumps(
                {
                    "snapshot_verified": True,
                    "seed": r["seed"],
                    "phase": active["key"] if active else list(r["phases"])[-1],
                    "updates": (
                        active["history"][-1]["completed_updates"]
                        if active
                        else r["phases"][list(r["phases"])[-1]]["updates"]
                    ),
                }
            )
        )
    else:
        result = verify(args.input, args.source_commit)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    "verified": True,
                    "seed": result["seed"],
                    "arms": {
                        k: {
                            "tv": v["metrics"]["pmf_tv"],
                            "gain": v["heldout_gain_nats_per_row"],
                            "gate": v["full_gate"],
                        }
                        for k, v in result["evaluation"]["arms"].items()
                    },
                    "paired_ll": result["evaluation"]["adam_minus_em_heldout_ll"],
                }
            )
        )


if __name__ == "__main__":
    main()
