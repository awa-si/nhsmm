"""Post-fit joint-study recovery metrics and independent public numerical checks."""

import numpy as np
import torch

from nhsmm.filtering import filter_model_sequence
from scripts.block_mass_smoothing import marginal_objective
from scripts.compare_unlabelled_duration_em import expectation, metrics
from scripts.fixed_bin_posterior_score import density, score_gradient
from scripts.generate_trading_regime_dataset import BASE_TRANSITION
from scripts.joint_gaussian_duration import chunk_statistics, emission_gradients, scores, statistics
from scripts.verify_isolated_trading_hsmm import (
    MEANS,
    make_model,
    segment_reference,
    sequences,
    true_law,
)


def independent_ll(x, p, means, variances):
    total = 0.0
    for start in range(0, len(x), 16):
        ll, _, _, _ = expectation(
            scores(x[start : start + 16], means, variances), p, BASE_TRANSITION, np.full(4, 0.25)
        )
        total += float(ll.sum())
    return total


def numerical_check(x, p, means, variances):
    """Actual learned parameters; public autograd, segment LL/filter reference."""
    torch.set_num_threads(1)
    x = x[:2]
    raw = variances + np.log(-np.expm1(-variances))
    model = make_model(p, sigma=2.0)
    with torch.no_grad():
        model.dist.emission.mu.copy_(torch.tensor(means[:, None]))
        model.dist.emission.log_var.copy_(torch.tensor(raw[:, None]))
    model.dist.emission.mu.requires_grad_(True)
    model.dist.emission.log_var.requires_grad_(True)
    z = torch.nn.Parameter(torch.tensor(np.log(p[:, ::8][:, :-1] / p[:, -1, None])))
    pmf = density(z)
    np.testing.assert_allclose(pmf.detach().numpy(), p, atol=1e-14, rtol=1e-12)
    del model.dist.duration.logits
    model.dist.duration.logits = pmf.log()
    ll = model.log_likelihood(torch.tensor(x), reduce=True)
    (ll + 1e-6 * pmf.log().sum()).backward()
    s = statistics(x, p, means, variances)
    gm, gv = emission_gradients(s, means, variances)
    emission = scores(x, means, variances)
    references = [segment_reference(row, p, BASE_TRANSITION, np.full(4, 0.25)) for row in emission]
    reference_ll = sum(row[0] for row in references)
    reference_filter = np.stack([np.exp(row[1]) for row in references])
    with torch.no_grad():
        trace = filter_model_sequence(model, torch.tensor(x))
    forward = chunk_statistics(x, p, means, variances, return_filter=True)
    return {
        "public_ll_error": abs(float(ll.detach()) - s["ll"]),
        "segment_ll_error": abs(reference_ll - s["ll"]),
        "duration_gradient_error": float(
            torch.max(torch.abs(z.grad - score_gradient(z, s["duration_counts"])))
        ),
        "mean_gradient_error": float(
            np.max(np.abs(model.dist.emission.mu.grad.numpy()[:, 0] - gm))
        ),
        "raw_variance_gradient_error": float(
            np.max(np.abs(model.dist.emission.log_var.grad.numpy()[:, 0] - gv / (1 + np.exp(-raw))))
        ),
        "public_filter_error": float(
            np.max(np.abs(trace.state_posterior.numpy() - forward["filtered_state_probabilities"]))
        ),
        "segment_filter_error": float(
            np.max(np.abs(trace.state_posterior.numpy() - reference_filter))
        ),
    }


def evaluate(r, x):
    from scripts.compare_joint_gaussian_duration import ORDER, check_record, digest

    if r["active"] is not None or list(r["phases"]) != list(ORDER):
        raise ValueError("outer evaluation requires all three complete fits")
    d, gates = r["contract"]["data"], r["contract"]["gates"]
    for record in r["phases"].values():
        check_record(record, r["contract"]["fit"]["em_updates_per_arm"], complete=True)
    y, _ = sequences(
        r["seed"] + d["outer_seed_offset"],
        count=d["outer_sequences"],
        steps=d["steps"],
        sigma=d["sigma"],
    )
    truth = true_law()
    arms = {}
    for key in ORDER:
        record = r["phases"][key]
        p, means, variances = (np.asarray(record[k]) for k in ("pmf", "means", "variances"))
        ll = independent_ll(x, p, means, variances)
        if (
            abs(ll - record["history"][-1]["ll"]) > 1e-7
            or abs(marginal_objective(ll, p, 0) - record["history"][-1]["objective"]) > 1e-7
        ):
            raise ValueError("independent terminal likelihood")
        outer = independent_ll(y, p, means, variances)
        initial = record["initial"]
        baseline = independent_ll(
            y,
            np.asarray(initial["pmf"]),
            np.asarray(initial["means"]),
            np.asarray(initial["variances"]),
        )
        order = np.argsort(means)
        m = metrics(p[order], truth)
        mean_error = np.abs(means[order] - MEANS)
        variance_error = np.abs(variances[order] - d["sigma"] ** 2) / d["sigma"] ** 2
        gain = (outer - baseline) / y.shape[0] / y.shape[1]
        checks = numerical_check(y, p, means, variances)
        numerical = all(v <= (1e-8 if "filter" in k else 1e-7) for k, v in checks.items())
        statistical = bool(
            max(m["pmf_tv"]) <= gates["each_state_duration_tv_at_most"]
            and max(mean_error) <= gates["each_state_mean_absolute_error_at_most"]
            and max(variance_error) <= gates["each_state_variance_relative_error_at_most"]
            and gain > 0
        )
        arms[key] = {
            "metrics": m,
            "post_fit_mean_sort": order.tolist(),
            "mean_absolute_error": mean_error.tolist(),
            "variance_relative_error": variance_error.tolist(),
            "heldout_ll": outer,
            "initial_heldout_ll": baseline,
            "heldout_gain_nats_per_row": gain,
            "terminal_train_ll_error": abs(ll - record["history"][-1]["ll"]),
            "numerical_check": checks,
            "numerical_gate": numerical,
            "statistical_gate": statistical,
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
        ends = np.flatnonzero(row[1:] != row[:-1]) + 1
        for end in ends:
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
