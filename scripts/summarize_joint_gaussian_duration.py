"""Join three immutable, SHA-bound joint recovery proofs without selection."""

import argparse
import hashlib
import json
from pathlib import Path


def summarize(directory):
    rows, inputs, reports = [], {}, []
    order = ("known_emission_control", "estimated_emission_frozen", "estimated_emission_joint")
    for seed in (1930, 1931, 1932):
        prefix = directory / f"joint-gaussian-duration-seed{seed}"
        path = Path(str(prefix) + "-native-20261014.json")
        proof_path = Path(str(prefix) + "-verification-native-20261014.json")
        raw, proof = path.read_bytes(), proof_path.read_bytes()
        r, v = json.loads(raw), json.loads(proof)
        e = r["evaluation"]
        if (
            r["schema"] != "nhsmm-joint-gaussian-duration-v1"
            or v["schema"] != "nhsmm-joint-gaussian-duration-verification-v1"
            or r["seed"] != seed
            or v["seed"] != seed
            or not v["verified"]
            or not v["source_software_contract_data_initialization_history_grid_pass"]
            or not v["independent_terminal_outer_public_replay_pass"]
            or not v["completed_resume_noop"]
            or v["verification_em_updates"] != 0
            or v["source_files"] != 52
            or v["source_commit"] != r["execution"]["source_commit"]
            or v["input_report_sha256"] != hashlib.sha256(raw).hexdigest()
            or v["evaluation"] != e
            or e["actual_em_updates"] != 90
            or e["oracle"] is not False
            or e["labels_used_in_learning_or_selection"] is not False
            or list(e["arms"]) != list(order)
        ):
            raise ValueError("incomplete or inconsistent joint proof")
        for key in order:
            a = e["arms"][key]
            if not a["numerical_gate"] or a["full_gate"] != (
                a["numerical_gate"] and a["statistical_gate"]
            ):
                raise ValueError("joint gate arithmetic")
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
                        "gain": a["heldout_gain_nats_per_row"],
                        "full_gate": a["full_gate"],
                        "statistical_gate": a["statistical_gate"],
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
        raise ValueError("joint source/protocol drift or reused observations")
    return {
        "schema": "nhsmm-joint-gaussian-duration-final-batch-v1",
        "verified": True,
        "source_commit": reports[0]["execution"]["source_commit"],
        "source_files": 52,
        "aggregation_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seeds": [1930, 1931, 1932],
        "actual_em_updates": 270,
        "verification_em_updates": 0,
        "inputs": inputs,
        "rows": rows,
        "full_gate_pass_counts": {
            k: sum(row["arms"][k]["full_gate"] for row in rows) for k in order
        },
        "joint_paired_outer_ll_win_count": sum(
            row["joint_minus_frozen_heldout_ll"] > 0 for row in rows
        ),
        "all_numerical_replays_and_exact_resume_pass": True,
        "oracle": False,
        "limits": "scalar Gaussian/fixed-bin duration EM, known static transition/initial; bounded updates, no global convergence, neural/Trading/legacy NPZ acceptance; known control is separate information reference; no post-outer tuning",
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
