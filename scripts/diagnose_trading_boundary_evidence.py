"""Separate hard marginal switches from causal one-edge transition evidence."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from nhsmm.filtering import HSMMFilterState, filter_model_sequence
from nhsmm.transitions import one_step_transition_forecast
from scripts.diagnose_trading_decoder_duration import score
from scripts.validate_trading_regime_usefulness import (
    RunBudget,
    _fit,
    _ll,
    _standardize,
    _standardize_context,
    _windows,
)

BUDGETS = (4, 30)
SEEDS = (901, 902, 903)


def update_edge(forecast, emission):
    """Bayes-update boundary and continuation mass with only the new observation."""
    emission = emission.to(forecast.next_state_prior)
    log_z = torch.logsumexp(forecast.next_state_prior.log() + emission, dim=-1)
    if not torch.isfinite(log_z).all():
        raise ValueError("observation has no finite predictive support")
    joint = (
        forecast.boundary_transition_joint.log() + emission[:, None, :] - log_z[:, None, None]
    ).exp()
    continuation = (forecast.next_state_prior - forecast.next_episode_state_joint).clamp_min(0)
    continuation = (continuation.log() + emission - log_z[:, None]).exp()
    return {
        "joint": joint,
        "current": continuation + joint.sum(1),
        "previous": continuation + joint.sum(2),
        "end": joint.sum((1, 2)),
        "change": joint.sum((1, 2)) - joint.diagonal(dim1=-2, dim2=-1).sum(-1),
    }


def match_switches(truth, predicted, tolerance=5):
    """Maximum-cardinality one-to-one matching; only for post-fit evaluation."""
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    matched = np.zeros_like(predicted, dtype=bool)
    for b, (true, pred) in enumerate(zip(truth, predicted)):
        ts = np.flatnonzero(true[1:] != true[:-1]) + 1
        ps = np.flatnonzero(pred[1:] != pred[:-1]) + 1
        i = j = 0
        while i < len(ts) and j < len(ps):
            if ps[j] < ts[i] - tolerance:
                j += 1
            elif ts[i] < ps[j] - tolerance:
                i += 1
            else:
                matched[b, ps[j]] = True
                i += 1
                j += 1
    return matched


def describe(values):
    if not len(values):
        return {"n": 0, "median": None, "q10": None, "q90": None, "below_half": None}
    v = np.asarray(values, dtype=float)
    return {
        "n": len(v),
        "median": float(np.median(v)),
        "q10": float(np.quantile(v, 0.1)),
        "q90": float(np.quantile(v, 0.9)),
        "below_half": float(np.mean(v < 0.5)),
    }


def inspect_edges(model, x, context, truth, true_age):
    with torch.inference_mode():
        tx, tc = torch.tensor(x), torch.tensor(context)
        trace = filter_model_sequence(model, tx, context=tc)
        state_prob = trace.state_posterior.numpy()
        age_prob = trace.log_posterior.exp().sum(2).numpy()
        predicted = state_prob.argmax(-1)
        sequence = model._build_sequence_set(tx, context=tc)
        duration = model.dist.duration.log_matrix(
            context=sequence.contexts, T=x.shape[1], soft_dmax=model.duration_logits_bias
        )
        transition = model.dist.transition.log_matrix(
            context=sequence.contexts, T=x.shape[1], soft_dmax=model.duration_logits_bias
        )
        tail = model.dist.duration.tail_probability().double()
        b_count, time, k = state_prob.shape
        prior_end = np.zeros((b_count, time))
        prior_change = np.zeros_like(prior_end)
        after_end = np.zeros_like(prior_end)
        after_change = np.zeros_like(prior_end)
        revised = np.zeros((b_count, time, k))
        edge_joint = np.zeros((b_count, time, k, k))
        max_state_error = max_age1_error = 0.0
        for t in range(1, time):
            previous = trace.log_posterior[:, t - 1].double()
            previous -= torch.logsumexp(previous.flatten(1), dim=1)[:, None, None]
            forecast = one_step_transition_forecast(
                HSMMFilterState(previous),
                duration[:, t - 1].double(),
                transition[:, t - 1].double(),
                tail,
            )
            updated = update_edge(forecast, sequence.log_probs[:, t].double())
            actual = torch.tensor(state_prob[:, t]).double()
            max_state_error = max(max_state_error, float((updated["current"] - actual).abs().max()))
            reset = torch.tensor(age_prob[:, t, 0]).double()
            max_age1_error = max(max_age1_error, float((updated["end"] - reset).abs().max()))
            prior_end[:, t] = forecast.episode_end_probability.numpy()
            prior_change[:, t] = forecast.state_change_probability.numpy()
            after_end[:, t] = updated["end"].numpy()
            after_change[:, t] = updated["change"].numpy()
            revised[:, t] = updated["previous"].numpy()
            edge_joint[:, t] = updated["joint"].numpy()
    # Float64 edge decomposition consumes Float32 model/trace scores.
    if max(max_state_error, max_age1_error) > 5e-5:
        raise AssertionError("edge decomposition disagrees with production causal filtering")
    switched = np.zeros_like(predicted, dtype=bool)
    switched[:, 1:] = predicted[:, 1:] != predicted[:, :-1]
    matched = match_switches(truth, predicted)
    eligible = np.ones_like(switched)
    eligible[:, 0] = False
    true_boundary = np.zeros_like(switched)
    true_boundary[:, 1:] = truth[:, 1:] != truth[:, :-1]
    categories = {
        "matched_argmax_switch": switched & matched,
        "unmatched_argmax_switch": switched & ~matched,
        "no_argmax_switch": eligible & ~switched,
        "true_state_boundary": true_boundary,
        "true_episode_boundary": eligible & (true_age == 1),
    }
    measurements = {
        "prior_episode_end": prior_end,
        "prior_state_change": prior_change,
        "updated_episode_end": after_end,
        "updated_state_change": after_change,
        "age_one_mass_after": age_prob[:, :, 0],
        "state_confidence_after": state_prob.max(-1),
        "state_margin_after": np.sort(state_prob, axis=-1)[:, :, -1]
        - np.sort(state_prob, axis=-1)[:, :, -2],
    }
    groups = {
        name: {m: describe(v[mask]) for m, v in measurements.items()}
        for name, mask in categories.items()
    }
    events = []
    ages = np.arange(1, age_prob.shape[-1] + 1)
    for b, t in zip(*np.nonzero(switched)):
        old, new = int(predicted[b, t - 1]), int(predicted[b, t])
        events.append(
            {
                "window": int(b),
                "row": int(t),
                "matched_tol5": bool(matched[b, t]),
                "old_state": old,
                "new_state": new,
                **{name: float(value[b, t]) for name, value in measurements.items()},
                "state_confidence_before": float(state_prob[b, t - 1].max()),
                "previous_state_revised_to_current": bool(revised[b, t].argmax() == new),
                "previous_was_old_updated": float(revised[b, t, old]),
                "old_to_new_updated": float(edge_joint[b, t, old, new]),
                "changed_to_new_updated": float(
                    edge_joint[b, t, :, new].sum() - edge_joint[b, t, new, new]
                ),
                "capped_age_mean_before": float(age_prob[b, t - 1] @ ages),
                "capped_age_mean_after": float(age_prob[b, t] @ ages),
                "age_mode_before": int(age_prob[b, t - 1].argmax() + 1),
                "age_mode_after": int(age_prob[b, t].argmax() + 1),
                "tail_age_mass_before": float(age_prob[b, t - 1, -1]),
                "tail_age_mass_after": float(age_prob[b, t, -1]),
            }
        )
    # Threshold .5 is a declared descriptive split, not a decoder policy.
    false_events = [e for e in events if not e["matched_tol5"]]
    return {
        "filter": score(truth, predicted),
        "groups": groups,
        "events": events,
        "true_state_boundaries": int(true_boundary.sum()),
        "true_episode_boundaries": int(categories["true_episode_boundary"].sum()),
        "unmatched_switches": len(false_events),
        "unmatched_updated_change_below_half": sum(
            e["updated_state_change"] < 0.5 for e in false_events
        ),
        "unmatched_previous_revised_to_current": sum(
            e["previous_state_revised_to_current"] for e in false_events
        ),
        "sum_prior_state_change": float(prior_change[:, 1:].sum()),
        "sum_updated_state_change": float(after_change[:, 1:].sum()),
        "max_state_posterior_error": max_state_error,
        "max_reset_posterior_error": max_age1_error,
    }


def run(job):
    seed, path, budgets = job
    torch.set_num_threads(1)
    with np.load(path) as z:
        data = {name: z[name] for name in z.files}
    n = len(data["close"])
    train = _windows(data, 0, int(0.6 * n), 8, window=240)
    test = _windows(data, int(0.8 * n), n, 4, window=240)
    train_x, test_x = _standardize(train[0], test[0])
    train_c, test_c = _standardize_context(train[1], test[1])
    rows = []
    for count in budgets:
        budget = RunBudget(240, 8, 4, count, 2)
        model = _fit(
            seed,
            train_x,
            train_c,
            context=True,
            hmm=False,
            emission_init_mode="kmeans_cluster_variance",
            budget=budget,
        )
        row = inspect_edges(model, test_x, test_c, test[2], test[3])
        row.update(max_iter=count, test_ll_per_row=_ll(model, test_x, test_c))
        rows.append(row)
    return {"seed": seed, "arms": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-iters", default="4,30")
    parser.add_argument("--seeds", default="901,902,903")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    try:
        budgets = tuple(int(x) for x in args.max_iters.split(","))
        seeds = tuple(int(x) for x in args.seeds.split(","))
        if not budgets or min(budgets) < 1 or len(set(budgets)) != len(budgets):
            raise ValueError("distinct positive iteration counts required")
        if not seeds or len(set(seeds)) != len(seeds) or args.workers < 1:
            raise ValueError("distinct seeds and positive workers required")
    except ValueError as exc:
        parser.error(str(exc))
    manifest = json.loads(args.dataset.with_name("manifest.json").read_text())
    if manifest.get("transition_context", {}).get("conditioning_time") != "episode_end":
        parser.error("corrected episode_end data required")
    jobs = [(seed, args.dataset, budgets) for seed in seeds]
    if args.workers == 1:
        rows = [run(job) for job in jobs]
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(args.workers, len(seeds))
        ) as pool:
            rows = list(pool.map(run, jobs))
    summary = {}
    for count in budgets:
        group = [a for r in rows for a in r["arms"] if a["max_iter"] == count]
        summary[str(count)] = {
            name: float(np.median([a[name] for a in group]))
            for name in (
                "unmatched_switches",
                "unmatched_updated_change_below_half",
                "unmatched_previous_revised_to_current",
                "sum_prior_state_change",
                "sum_updated_state_change",
                "max_state_posterior_error",
                "max_reset_posterior_error",
            )
        }
        summary[str(count)]["filter"] = {
            m: float(np.median([a["filter"][m] for a in group]))
            for m in ("accuracy", "ari", "boundary_f1_tol5", "switches", "run_duration_rel_error")
        }
        summary[str(count)]["unmatched_switch_groups"] = {
            m: float(
                np.median([a["groups"]["unmatched_argmax_switch"][m]["median"] for a in group])
            )
            for m in (
                "prior_episode_end",
                "prior_state_change",
                "updated_episode_end",
                "updated_state_change",
                "state_confidence_after",
                "state_margin_after",
            )
        }
    payload = {
        "schema": "nhsmm-trading-boundary-evidence-pilot-v1",
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "seeds": list(seeds),
        "joint_iterations": list(budgets),
        "window": 240,
        "train_windows": 8,
        "eval_windows": 4,
        "transition_refine_steps": 2,
        "full_protocol": False,
        "runs": rows,
        "summary": summary,
        "contract": {
            "prior": "boundary t-1 -> t uses filtered state/context at t-1, before observing x_t",
            "updated": "same edge Bayes-updated with emission at x_t; no future observations",
            "matching": "post-fit maximum-cardinality one-to-one state-boundary matching within 5 rows",
            "labels": "scoring/grouping only; no training or decoder/budget selection",
            "age": "age-one is episode-reset mass since D=64; D+ bucket caps reported age mean",
            "threshold": ".5 is descriptive only, not an operational policy",
            "sums": "sums of online one-edge probabilities with different information sets; not a common full-window posterior expected path count",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
