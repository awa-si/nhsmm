"""Compare fixed causal decision rules on matched fits; labels enter scoring only."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from nhsmm.filtering import (
    HSMMFilterState,
    duration_log_hazard,
    filter_model_sequence,
    filter_step,
    initialize_filter,
)
from scripts.diagnose_trading_decoder_duration import score
from scripts.validate_trading_regime_usefulness import (
    RunBudget,
    _fit,
    _ll,
    _standardize,
    _standardize_context,
    _windows,
)

RULES = ("marginal_argmax", "prefix_map_endpoint", "committed_state", "retrospective_viterbi")


def causal_decisions(initial, emissions, durations, transitions, tail):
    """Fixed, zero-lag rules for a rectangular batch of canonical log scores.

    Prefix MAP maximizes joint state/age/episode paths through the current row.
    Its historical best path may change; only its current endpoint is emitted.
    Committed-state sums over ages conditional on all earlier emitted states,
    chooses the current state marginal mode, and conditions on that choice.
    Ties use the first flattened index. Neither rule guarantees recovery.
    """
    b, time, k = emissions.shape
    d = durations.shape[-1]
    if time < 1:
        raise ValueError("at least one observation required")
    expected = ((b, k), (b, time, k, d), (b, time, k, d, k))
    if (initial.shape, durations.shape, transitions.shape) != expected:
        raise ValueError("canonical score shapes required")
    if not torch.isfinite(torch.logsumexp(initial + emissions[:, 0], -1)).all():
        raise ValueError("initial observation has no support")
    frontier = emissions.new_full((b, k, d), -torch.inf)
    frontier[..., 0] = initial + emissions[:, 0]
    conditional = initialize_filter(initial, emissions[:, 0], d)
    prefix = torch.empty((b, time), dtype=torch.long, device=emissions.device)
    committed = torch.empty_like(prefix)
    selected_age = torch.empty_like(prefix)
    for t in range(time):
        if t:
            end, continuation = duration_log_hazard(durations[:, t - 1], tail)
            predicted = torch.full_like(frontier, -torch.inf)
            if d > 1:
                predicted[..., 1:] = frontier[..., :-1] + continuation[..., :-1]
            boundary = frontier + end
            predicted[..., 0] = (boundary[..., None] + transitions[:, t - 1]).flatten(1, 2).amax(1)
            if tail is not None:
                tail_continue = frontier[..., -1] + continuation[..., -1]
                predicted[..., -1] = torch.maximum(predicted[..., -1], tail_continue)
            frontier = predicted + emissions[:, t, :, None]
            conditional = filter_step(
                conditional, emissions[:, t], durations[:, t - 1], transitions[:, t - 1], tail
            )
        best = frontier.flatten(1).amax(1)
        if not torch.isfinite(best).all():
            raise ValueError("prefix has no supported path")
        frontier = frontier - best[:, None, None]
        endpoint = frontier.flatten(1).argmax(1)
        prefix[:, t] = endpoint // d
        selected_age[:, t] = endpoint % d + 1
        chosen = conditional.state_log_posterior.argmax(1)
        committed[:, t] = chosen
        mask = torch.arange(k, device=emissions.device)[None, :, None] == chosen[:, None, None]
        restricted = conditional.log_posterior.masked_fill(~mask, -torch.inf)
        normalizer = torch.logsumexp(restricted.flatten(1), 1)
        conditional = HSMMFilterState(restricted - normalizer[:, None, None])
    return {"prefix_map_endpoint": prefix, "committed_state": committed}, selected_age


def evaluate_model(model, x, context, truth):
    """Extract the same canonical score tensors used by the production filter."""
    with torch.inference_mode():
        tx, tc = torch.tensor(x), torch.tensor(context)
        sequence = model._build_sequence_set(tx, context=tc)
        time = tx.shape[1]
        initial = model.dist.initial.log_matrix(context=sequence.canonical, T=time)[:, 0]
        durations = model.dist.duration.log_matrix(
            context=sequence.contexts, T=time, soft_dmax=model.duration_logits_bias
        )
        transitions = model.dist.transition.log_matrix(
            context=sequence.contexts, T=time, soft_dmax=model.duration_logits_bias
        )
        decisions, age = causal_decisions(
            initial.double(),
            sequence.log_probs.double(),
            durations.double(),
            transitions.double(),
            model.dist.duration.tail_probability().double(),
        )
        decisions["marginal_argmax"] = filter_model_sequence(
            model, tx, context=tc
        ).state_posterior.argmax(-1)
        decisions["retrospective_viterbi"] = torch.stack(
            model.predict(tx, context=tc, verbose=False)
        )
    paths = {name: value.numpy() for name, value in decisions.items()}
    return {
        "scores": {name: score(truth, path) for name, path in paths.items()},
        "paths": {name: path.tolist() for name, path in paths.items()},
        "prefix_endpoint_age": age.tolist(),
        "true_paths": truth.tolist(),
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
    arms = []
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
        row = evaluate_model(model, test_x, test_c, test[2])
        row.update(max_iter=count, test_ll_per_row=_ll(model, test_x, test_c))
        arms.append(row)
        print(f"seed={seed} iterations={count} complete", flush=True)
    return {"seed": seed, "arms": arms}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", default="901,902,903")
    parser.add_argument("--max-iters", default="4,30")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    try:
        seeds = tuple(int(v) for v in args.seeds.split(","))
        budgets = tuple(int(v) for v in args.max_iters.split(","))
        if not seeds or len(set(seeds)) != len(seeds) or args.workers < 1:
            raise ValueError("distinct seeds and positive worker count required")
        if not budgets or min(budgets) < 1 or len(set(budgets)) != len(budgets):
            raise ValueError("distinct positive iteration counts required")
    except ValueError as exc:
        parser.error(str(exc))
    manifest = json.loads(args.dataset.with_name("manifest.json").read_text())
    if manifest.get("transition_context", {}).get("conditioning_time") != "episode_end":
        parser.error("corrected episode_end dataset required")
    jobs = [(seed, args.dataset, budgets) for seed in seeds]
    if args.workers == 1:
        rows = [run(job) for job in jobs]
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(args.workers, len(seeds))
        ) as pool:
            rows = list(pool.map(run, jobs))
    metrics = ("accuracy", "ari", "boundary_f1_tol5", "run_duration_rel_error", "switches")
    summary = {
        str(count): {
            rule: {
                metric: float(
                    np.median(
                        [
                            arm["scores"][rule][metric]
                            for row in rows
                            for arm in row["arms"]
                            if arm["max_iter"] == count
                        ]
                    )
                )
                for metric in metrics
            }
            for rule in RULES
        }
        for count in budgets
    }
    payload = {
        "schema": "nhsmm-trading-causal-decision-pilot-v1",
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
            "labels": "scoring only; rules fixed before this experiment",
            "prefix_map_endpoint": "zero-lag endpoint of prefix joint MAP; emitted history need not form a joint MAP path",
            "committed_state": "zero-lag greedy conditional state marginal mode, summing over ages; irrevocable past output constraints may amplify errors",
            "retrospective_viterbi": "full-window MAP control, unavailable at decision time",
            "likelihood": "one unchanged fitted model per seed/budget; decision rules do not change its likelihood",
            "selection": "no rule promoted or package default changed based on this OOS pilot",
            "window_start": "age one, independently initialized windows as in earlier pilots",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
