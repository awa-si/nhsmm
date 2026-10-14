"""Join final SHA-bound static-chain recovery proofs; no fitting or selection."""

import argparse
import hashlib
import json
from pathlib import Path


def summarize(directory):
    order = ("known_chain_reference", "estimated_chain_frozen", "estimated_chain_joint")
    inputs, rows, reports = {}, [], []
    for seed in (1934, 1935, 1936):
        prefix = directory / f"joint-static-chain-seed{seed}"
        raw = Path(str(prefix) + "-native-20261014.json").read_bytes()
        proof = Path(str(prefix) + "-verification-native-20261014.json").read_bytes()
        r, v = json.loads(raw), json.loads(proof)
        e = r["evaluation"]
        if (
            r["schema"] != "nhsmm-joint-static-chain-v1"
            or v["schema"] != "nhsmm-joint-static-chain-verification-v1"
            or r["seed"] != seed
            or v["seed"] != seed
            or not v["verified"]
            or not v["source_software_contract_data_initialization_history_grid_pass"]
            or not v["independent_terminal_outer_public_replay_pass"]
            or not v["completed_resume_noop"]
            or v["verification_em_updates"] != 0
            or v["source_files"] != 60
            or v["source_commit"] != r["execution"]["source_commit"]
            or v["input_report_sha256"] != hashlib.sha256(raw).hexdigest()
            or v["evaluation"] != e
            or e["actual_em_updates"] != 90
            or e["oracle"] is not False
            or e["labels_used_in_learning_or_selection"] is not False
            or list(e["arms"]) != list(order)
        ):
            raise ValueError("incomplete or inconsistent static-chain proof")
        for a in e["arms"].values():
            if (
                not a["numerical_gate"]
                or a["statistical_gate"] != all(a["component_gates"].values())
                or a["full_gate"] != (a["numerical_gate"] and a["statistical_gate"])
            ):
                raise ValueError("component gate arithmetic")
        rows.append(
            {
                "seed": seed,
                "joint_minus_frozen_heldout_ll": e["joint_minus_frozen_heldout_ll"],
                "post_fit_episode_exposure": e["post_fit_training_episode_exposure"],
                "arms": {
                    k: {
                        "tv": a["metrics"]["pmf_tv"],
                        "mean_error": a["mean_absolute_error"],
                        "variance_relative_error": a["variance_relative_error"],
                        "transition_row_tv": a["transition_row_tv"],
                        "initial_tv": a["initial_tv"],
                        "gain": a["heldout_gain_nats_per_row"],
                        "component_gates": a["component_gates"],
                        "full_gate": a["full_gate"],
                    }
                    for k, a in e["arms"].items()
                },
            }
        )
        inputs[str(seed)] = {
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "proof_sha256": hashlib.sha256(proof).hexdigest(),
        }
        reports.append(r)
    if (
        any(
            r["execution"] != reports[0]["execution"] or r["contract"] != reports[0]["contract"]
            for r in reports[1:]
        )
        or len({r["training_sha256"] for r in reports}) != 3
        or len({r["evaluation"]["heldout_sha256"] for r in reports}) != 3
    ):
        raise ValueError("static-chain source/protocol drift or reused data")
    return {
        "schema": "nhsmm-joint-static-chain-final-batch-v1",
        "verified": True,
        "source_commit": reports[0]["execution"]["source_commit"],
        "source_files": 60,
        "aggregation_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seeds": [1934, 1935, 1936],
        "actual_em_updates": 270,
        "verification_em_updates": 0,
        "inputs": inputs,
        "rows": rows,
        "full_gate_pass_counts": {
            k: sum(row["arms"][k]["full_gate"] for row in rows) for k in order
        },
        "component_gate_pass_counts": {
            k: {
                g: sum(row["arms"][k]["component_gates"][g] for row in rows)
                for g in rows[0]["arms"][k]["component_gates"]
            }
            for k in order
        },
        "joint_paired_outer_ll_win_count": sum(
            row["joint_minus_frozen_heldout_ll"] > 0 for row in rows
        ),
        "all_numerical_replays_and_exact_resume_pass": True,
        "oracle": False,
        "limits": "scalar Gaussian, fixed-bin duration, static transition/initial; known-chain control is information reference; bounded EM, not convergence or neural/Trading/legacy NPZ acceptance; no post-outer tuning",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path("docs/validation/results"))
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("preserve evidence: output already exists")
    result = summarize(a.results)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
