"""Join completed SHA-bound recovery proofs; no fitting or model selection."""

import argparse
import hashlib
import json
from pathlib import Path


def summarize(directory):
    reports, rows, inputs = [], [], {}
    for seed in (1922, 1923, 1924):
        prefix = directory / f"fixed-bin-optimizer-seed{seed}"
        raw = Path(str(prefix) + "-native-20261014.json").read_bytes()
        r = json.loads(raw)
        v = json.loads(Path(str(prefix) + "-verification-native-20261014.json").read_text())
        if (
            r["seed"] != seed
            or v["seed"] != seed
            or v["source_commit"] != r["execution"]["source_commit"]
            or v["schema"] != "nhsmm-fixed-bin-optimizer-verification-v1"
            or r["schema"] != "nhsmm-fixed-bin-optimizer-recovery-v1"
            or r["evaluation"]["labels_used_in_learning_or_selection"] is not False
            or not v["verified"]
            or not v["numerical_replay_pass"]
            or not v["completed_resume_noop"]
            or not v["source_software_contract_data_phase_history_grid_moments_pass"]
            or v["terminal_replay_optimizer_updates"] != 0
            or v["source_files"] != 23
            or v["input_report_sha256"] != hashlib.sha256(raw).hexdigest()
            or v["evaluation"] != r["evaluation"]
            or r["evaluation"]["actual_updates"] != 57
            or r["evaluation"]["actual_em_msteps"] != 30
            or r["evaluation"]["actual_adam_updates"] != 27
        ):
            raise ValueError("incomplete or inconsistent seed proof")
        e = r["evaluation"]
        rows.append(
            {
                "seed": seed,
                "adam_minus_em_outer_ll": e["adam_minus_em_heldout_ll"],
                "adam_minus_em_final_training_objective": r["phases"]["fixed_bin_adam"]["history"][
                    -1
                ]["penalized_objective"]
                - r["phases"]["fixed_bin_em"]["history"][-1]["penalized_objective"],
                "post_fit_episode_exposure": e["post_fit_training_episode_exposure"],
                "arms": {
                    k: {
                        "tv": a["metrics"]["pmf_tv"],
                        "gain": a["heldout_gain_nats_per_row"],
                        "full_gate": a["full_gate"],
                        "final_training_objective": r["phases"][k]["history"][-1][
                            "penalized_objective"
                        ],
                    }
                    for k, a in e["arms"].items()
                },
            }
        )
        inputs[str(seed)] = {
            "raw_sha256": v["input_report_sha256"],
            "proof_sha256": hashlib.sha256(
                Path(str(prefix) + "-verification-native-20261014.json").read_bytes()
            ).hexdigest(),
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
        raise ValueError("source/protocol mismatch or reused data")
    return {
        "schema": "nhsmm-fixed-bin-optimizer-final-batch-v1",
        "verified": True,
        "source_commit": reports[0]["execution"]["source_commit"],
        "source_files": 23,
        "seeds": [1922, 1923, 1924],
        "actual_updates": 171,
        "actual_em_msteps": 90,
        "actual_adam_updates": 81,
        "verification_optimizer_updates": 0,
        "inputs": inputs,
        "rows": rows,
        "full_gate_pass_counts": {
            k: sum(row["arms"][k]["full_gate"] for row in rows)
            for k in ("fixed_bin_em", "fixed_bin_adam")
        },
        "adam_paired_outer_ll_win_count": sum(row["adam_minus_em_outer_ll"] > 0 for row in rows),
        "all_numerical_replays_and_exact_resume_pass": True,
        "oracle": False,
        "limits": "matched fixed duration families and data, bounded27-update optimizers after shared warm3; not equal compute or convergence guarantees; known nuisance only, no joint-neural/Trading or legacy dataset acceptance; no post-outer tuning",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("docs/validation/results"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.results)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
