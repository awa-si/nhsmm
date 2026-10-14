"""Post-fit static-chain gates, independent LL and public all-five checks."""

import numpy as np

from scripts.compare_unlabelled_duration_em import expectation, metrics
from scripts.generate_trading_regime_dataset import BASE_TRANSITION
from scripts.joint_gaussian_duration import scores
from scripts.joint_static_chain import objective
from scripts.static_chain_check import numerical_check
from scripts.verify_isolated_trading_hsmm import MEANS, sequences, true_law


def independent_ll(x, p, means, variances, transition, initial):
    total = 0.0
    for start in range(0, len(x), 16):
        ll, _, _, _ = expectation(
            scores(x[start : start + 16], means, variances), p, transition, initial
        )
        total += float(ll.sum())
    return total


def evaluate(r, x):
    from scripts.compare_joint_static_chain import ORDER, check_record, digest

    if r["active"] is not None or list(r["phases"]) != list(ORDER):
        raise ValueError("outer evaluation requires all three complete fits")
    d, gates = r["contract"]["data"], r["contract"]["gates"]
    for record in r["phases"].values():
        check_record(record, r["contract"]["fit"]["updates_per_arm"], complete=True)
    y, _ = sequences(
        r["seed"] + d["outer_seed_offset"],
        count=d["outer_sequences"],
        steps=d["steps"],
        sigma=d["sigma"],
    )
    truth, arms = true_law(), {}
    keys = ("pmf", "means", "variances", "transition", "initial_probabilities")
    for key in ORDER:
        record = r["phases"][key]
        p, means, variances, transition, initial = (np.asarray(record[k]) for k in keys)
        ll = independent_ll(x, p, means, variances, transition, initial)
        if (
            abs(ll - record["history"][-1]["ll"]) > 1e-7
            or abs(objective(ll, p, transition, initial) - record["history"][-1]["objective"])
            > 1e-7
        ):
            raise ValueError("independent terminal likelihood/objective")
        outer = independent_ll(y, p, means, variances, transition, initial)
        baseline = independent_ll(y, *(np.asarray(record["initial"][k]) for k in keys))
        order = np.argsort(means)
        m = metrics(p[order], truth)
        mean_error = np.abs(means[order] - MEANS)
        variance_error = np.abs(variances[order] - d["sigma"] ** 2) / d["sigma"] ** 2
        transition_tv = np.abs(transition[np.ix_(order, order)] - BASE_TRANSITION).sum(1) / 2
        initial_tv = float(np.abs(initial[order] - 0.25).sum() / 2)
        gain = (outer - baseline) / (y.shape[0] * y.shape[1])
        checks = numerical_check(y, p, means, variances, transition, initial)
        numerical = all(
            np.isfinite(v)
            and v
            <= (
                gates["filter_error_at_most"]
                if "filter" in k
                else gates["ll_and_all_parameter_gradients_error_at_most"]
            )
            for k, v in checks.items()
        )
        component_gates = {
            "duration": bool(max(m["pmf_tv"]) <= gates["each_state_duration_tv_at_most"]),
            "mean": bool(max(mean_error) <= gates["each_state_mean_absolute_error_at_most"]),
            "variance": bool(
                max(variance_error) <= gates["each_state_variance_relative_error_at_most"]
            ),
            "transition": bool(max(transition_tv) <= gates["each_transition_row_tv_at_most"]),
            "initial": bool(initial_tv <= gates["initial_tv_at_most"]),
            "outer_gain": bool(gain > gates["heldout_gain_nats_per_row_strictly_greater_than"]),
        }
        statistical = all(component_gates.values())
        arms[key] = {
            "metrics": m,
            "post_fit_mean_sort": order.tolist(),
            "mean_absolute_error": mean_error.tolist(),
            "variance_relative_error": variance_error.tolist(),
            "transition_row_tv": transition_tv.tolist(),
            "initial_tv": initial_tv,
            "heldout_ll": outer,
            "initial_heldout_ll": baseline,
            "heldout_gain_nats_per_row": gain,
            "terminal_train_ll_error": abs(ll - record["history"][-1]["ll"]),
            "numerical_check": checks,
            "numerical_gate": bool(numerical),
            "component_gates": component_gates,
            "statistical_gate": bool(statistical),
            "full_gate": bool(numerical and statistical),
        }
    replay, labels = sequences(
        r["seed"] + d["training_seed_offset"],
        count=d["training_sequences"],
        steps=d["steps"],
        sigma=d["sigma"],
    )
    if digest(replay) != r["training_sha256"]:
        raise ValueError("post-fit exposure identity")
    completed, censored = np.zeros(4, dtype=int), np.zeros(4, dtype=int)
    for row in labels:
        for end in np.flatnonzero(row[1:] != row[:-1]) + 1:
            completed[row[end - 1]] += 1
        censored[row[-1]] += 1
    return {
        "heldout_sha256": digest(y),
        "arms": arms,
        "joint_minus_frozen_heldout_ll": arms[ORDER[2]]["heldout_ll"]
        - arms[ORDER[1]]["heldout_ll"],
        "actual_em_updates": sum(len(v["history"]) - 1 for v in r["phases"].values()),
        "labels_used_in_learning_or_selection": False,
        "oracle": False,
        "post_fit_training_episode_exposure": {
            "completed_per_state": completed.tolist(),
            "terminal_censored_per_state": censored.tolist(),
        },
    }
