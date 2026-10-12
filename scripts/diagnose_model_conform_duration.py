"""Direct duration-law diagnostic on exactly model-conform stationary HSMM data."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import hashlib
import math
import platform
import random
from pathlib import Path

import numpy as np
import torch

from nhsmm import ModelConfig, NHSMM
from nhsmm.filtering import filter_model_sequence

PMF = np.array(
    [
        [0.01, 0.02, 0.04, 0.08, 0.15, 0.20, 0.25, 0.25],
        [0.15, 0.25, 0.25, 0.15, 0.08, 0.05, 0.04, 0.03],
    ],
    dtype=float,
)
TAIL = np.array([0.12, 0.35])
TRANSITION = np.array([[0.02, 0.98], [0.98, 0.02]])
MEANS = np.array([[-3.0, 0.0], [3.0, 0.0]])
SIGMA = 0.35
D = 8


def survival(pmf, hazard, age):
    """P(L >= age), including the D+ geometric bucket."""
    if age < 1:
        raise ValueError("age must be positive")
    if age < len(pmf):
        return float(np.sum(pmf[age - 1 :]))
    return float(pmf[-1] * (1 - hazard) ** (age - len(pmf)))


def duration_mean(pmf, hazard):
    d = len(pmf)
    return float(pmf[:-1] @ np.arange(1, d) + pmf[-1] * (d - 1 + 1 / hazard))


def generate(seed, *, tail, count=6, steps=128, transition=None):
    """Independent episode-start sequences; final episode always right-censored."""
    rng = np.random.default_rng(seed)
    transition = TRANSITION if transition is None else np.asarray(transition, dtype=float)
    if (
        transition.shape != (2, 2)
        or not np.isfinite(transition).all()
        or (transition < 0).any()
        or not np.allclose(transition.sum(axis=1), 1.0)
    ):
        raise ValueError("transition must be a finite normalized nonnegative 2x2 matrix")
    x, states, ledger = [], [], []
    for _ in range(count):
        rows = np.empty((steps, 2), dtype=np.float32)
        labels = np.empty(steps, dtype=int)
        state = int(rng.integers(2))
        t = 0
        episodes = []
        while t < steps:
            bucket = int(rng.choice(D, p=PMF[state])) + 1
            length = bucket
            if tail and bucket == D:
                length = D - 1 + int(rng.geometric(TAIL[state]))
            observed = min(length, steps - t)
            rows[t : t + observed] = MEANS[state] + rng.normal(0, SIGMA, (observed, 2))
            labels[t : t + observed] = state
            episodes.append(
                {"state": state, "observed": observed, "censored": t + observed == steps}
            )
            t += observed
            state = int(rng.choice(2, p=transition[state]))
        x.append(rows)
        states.append(labels)
        ledger.append(episodes)
    return np.stack(x), np.stack(states), ledger


def complete_duration_nll(pmf, hazards, ledger):
    """Oracle episode-label score with survival for each terminal censored episode."""
    total = 0.0
    for episodes in ledger:
        for e in episodes:
            s, length = e["state"], e["observed"]
            if e["censored"]:
                mass = survival(pmf[s], hazards[s], length)
            elif length < D:
                mass = pmf[s, length - 1]
            else:
                mass = pmf[s, -1] * hazards[s] * (1 - hazards[s]) ** (length - D)
            total -= math.log(max(float(mass), 1e-300))
    return total


class FixedTemperatureNHSMM(NHSMM):
    """Script-only likelihood control; production optimizer and restarts unchanged."""

    def _compute_loss(self, *args, **kwargs):
        kwargs.update(t_min=1.0, t_max=1.0)
        return super()._compute_loss(*args, **kwargs)


def make_model(seed, tail, iterations, *, fixed=False):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    cfg = ModelConfig(
        n_states=2,
        n_features=2,
        max_duration=D,
        causal=True,
        duration_tail=tail,
        duration_tail_init_probability=0.5,
        use_context_encoder=False,
        dropout=0.0,
        emission_init_mode="kmeans_cluster_variance",
        duration_init_mode="uniform",
        transition_init_mode="uniform",
        initial_init_mode="uniform",
        n_init=1,
        max_iter=iterations,
        lr=0.008,
        use_scheduler=False,
        convergence_stop=False,
        seed=seed,
        verbose=False,
    )
    model = (FixedTemperatureNHSMM if fixed else NHSMM)(cfg, device="cpu")
    model.initialize_distributions()
    return model


def law(model, temperature=1.0):
    with torch.no_grad():
        p = (
            model.dist.duration.log_matrix(
                T=1, temperature=temperature, soft_dmax=model.duration_logits_bias
            )
            .exp()
            .reshape(2, D)
            .numpy()
        )
        h = model.dist.duration.tail_probability()
    return p, np.ones(2) if h is None else h.detach().numpy()


def evaluate(model, x, truth, ledger, tail, *, align=True):
    model.eval()
    tx = torch.tensor(x)
    with torch.inference_mode():
        pred = filter_model_sequence(model, tx).state_posterior.argmax(-1).numpy()
        identity = np.mean(pred == truth)
        swapped = np.mean((1 - pred) == truth)
        perm = [0, 1] if not align or identity >= swapped else [1, 0]
        p, h = law(model)
        p, h = p[perm], h[perm]
        ll = float(model.log_likelihood(tx, reduce=True)) / (x.shape[0] * x.shape[1])
    with torch.no_grad():
        transition = model.dist.transition.log_matrix(T=1).exp().reshape(2, D, 2).numpy()
        transition = transition[perm][:, :, perm]
        self_transition = np.stack([transition[k, :, k] for k in range(2)])
    true_h = TAIL if tail else np.ones(2)

    # Include finite masses and explicit tail offsets, plus one residual >1024 bucket.
    def extended(prob, hazard):
        finite = prob[:, :-1]
        offsets = np.arange(1025 - D)
        tailp = prob[:, -1, None] * hazard[:, None] * (1 - hazard[:, None]) ** offsets
        residual = prob[:, -1, None] * (1 - hazard[:, None]) ** (1025 - D)
        return np.concatenate([finite, tailp, residual], axis=1)

    tv = np.abs(extended(p, h) - extended(PMF, true_h)).sum(1) * 0.5
    endtemp = 0.3 + 0.7 / (
        1
        + math.exp(
            (10 / model.config.max_iter) * (model.config.max_iter - 1 - model.config.max_iter / 2)
        )
    )
    terminal_p, _ = law(model, endtemp)
    result = {
        "accuracy": float(max(identity, swapped) if align else identity),
        "self_transition_probability_by_age": self_transition.tolist(),
        "self_transition_mean": float(self_transition.mean()),
        "duration_tv_mean": float(tv.mean()),
        "duration_tv_states": tv.tolist(),
        "pmf": p.tolist(),
        "tail_hazard": h.tolist(),
        "learned_mean": [duration_mean(a, b) for a, b in zip(p, h)],
        "true_mean": [duration_mean(a, b) for a, b in zip(PMF, true_h)],
        "complete_episode_nll": complete_duration_nll(p, h, ledger),
        "true_complete_episode_nll": complete_duration_nll(PMF, true_h, ledger),
        "test_ll_per_row": ll,
        "switches": int(np.sum(pred[:, 1:] != pred[:, :-1])),
        "true_switches": int(np.sum(truth[:, 1:] != truth[:, :-1])),
        "permutation": perm,
        "scheduled_terminal_temperature": endtemp,
        "pmf_at_scheduled_terminal_temperature": terminal_p[perm].tolist(),
    }
    if not np.isfinite([ll, *tv, *h]).all():
        raise AssertionError("nonfinite diagnostic")
    return result


def oracle_duration_fit(seed, tail, x, iterations):
    """Known nuisance parameters; no episode labels enter marginal likelihood training."""
    model = make_model(seed, tail, iterations, fixed=True)
    with torch.no_grad():
        model.dist.initial.logits.fill_(-math.log(2))
        model.dist.transition.logits.copy_(
            torch.tensor(np.log(TRANSITION), dtype=torch.float32)[:, None, :].expand(2, D, 2)
        )
        model.dist.emission.mu.copy_(torch.tensor(MEANS, dtype=torch.float32))
        model.dist.emission.log_var.fill_(math.log(math.expm1(SIGMA**2)))
        model.duration_logits_bias.fill_(1.0)
        model.dist.duration.initialize(mode="uniform", jitter=0.0)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    params = [model.dist.duration.logits]
    if tail:
        params.append(model.dist.duration.tail_end_probability)
    for p in params:
        p.requires_grad_(True)
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.Adam(params, lr=0.008)
    tx = torch.tensor(x)
    completed_iterations = 0
    for _ in range(iterations):
        optimizer.zero_grad()
        _, loss = model._compute_loss(tx, loss_bias=0.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        optimizer.step()
        model.dist.duration._project_tail_parameter()
        completed_iterations += 1
    for name, p in model.named_parameters():
        if name in frozen and not torch.equal(p.detach(), frozen[name]):
            raise AssertionError("oracle nuisance parameters changed")
    model.eval()
    model._diagnostic_completed_iterations = completed_iterations
    return model


def optimization_evidence(model, requested):
    """Count recorded optimizer iterations, excluding final restart scoring."""
    counts = torch.isfinite(model._convergence.scores).sum(dim=0).tolist()
    if model.config.convergence_stop or counts != [requested] * model.config.n_init:
        raise AssertionError(f"fixed iteration budget not completed: {counts} != {requested}")
    return {
        "requested_iterations_per_restart": requested,
        "completed_iterations_per_restart": counts,
        "convergence_stop": model.config.convergence_stop,
        "count_source": "finite entries in production Convergence.scores; restart scoring excluded",
    }


def ledger_coverage(ledger):
    """Observed complete-tail counts distinguish absent information from failed learning."""
    return [
        {
            "state": state,
            "complete_episodes": sum(
                not e["censored"] and e["state"] == state for row in ledger for e in row
            ),
            "complete_D_plus_episodes": sum(
                not e["censored"] and e["state"] == state and e["observed"] >= D
                for row in ledger
                for e in row
            ),
            "censored_episodes": sum(
                e["censored"] and e["state"] == state for row in ledger for e in row
            ),
        }
        for state in range(2)
    ]


def run(job):
    seed, scenarios, joint_iters, oracle_iters = job
    torch.set_num_threads(1)
    rows = []
    for scenario in scenarios:
        tail = scenario == "tail"
        train, _, train_ledger = generate(seed + 20000, tail=tail)
        test, truth, ledger = generate(seed + 30000, tail=tail, count=4)
        arms = {}
        for name, it, fixed in [
            ("scheduled_short", 4, False),
            ("scheduled_joint", joint_iters, False),
            ("fixed_temperature_joint", joint_iters, True),
        ]:
            model = make_model(seed, tail, it, fixed=fixed)
            model.optimize(torch.tensor(train))
            arms[name] = evaluate(model, test, truth, ledger, tail)
            arms[name]["optimization"] = optimization_evidence(model, it)
            print(f"seed={seed} scenario={scenario} arm={name} complete", flush=True)
        model = oracle_duration_fit(seed, tail, train, oracle_iters)
        arms["oracle_duration_only"] = evaluate(model, test, truth, ledger, tail, align=False)
        arms["oracle_duration_only"]["optimization"] = {
            "requested_iterations": oracle_iters,
            "completed_iterations": model._diagnostic_completed_iterations,
            "count_source": "completed optimizer steps in script-local duration-only loop",
        }
        print(f"seed={seed} scenario={scenario} arm=oracle_duration_only complete", flush=True)
        rows.append(
            {
                "scenario": scenario,
                "arms": arms,
                "train_duration_coverage": ledger_coverage(train_ledger),
                "test_duration_coverage": ledger_coverage(ledger),
            }
        )
    return {"seed": seed, "scenarios": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", default="901,902,903")
    parser.add_argument("--scenarios", default="finite,tail")
    parser.add_argument("--joint-iters", type=int, default=40)
    parser.add_argument("--oracle-iters", type=int, default=80)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--base-commit", default=None)
    parser.add_argument("--runtime-label", default="unspecified")
    args = parser.parse_args()
    seeds = tuple(int(v) for v in args.seeds.split(","))
    scenarios = tuple(args.scenarios.split(","))
    if not seeds or len(set(seeds)) != len(seeds) or args.workers < 1:
        parser.error("distinct seeds and positive workers required")
    if (
        not scenarios
        or len(set(scenarios)) != len(scenarios)
        or not set(scenarios) <= {"finite", "tail"}
    ):
        parser.error("distinct finite/tail scenarios required")
    if min(args.joint_iters, args.oracle_iters) < 1:
        parser.error("iterations must be positive")
    jobs = [(s, scenarios, args.joint_iters, args.oracle_iters) for s in seeds]
    if args.workers == 1:
        rows = [run(job) for job in jobs]
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(args.workers, len(seeds))
        ) as pool:
            rows = list(pool.map(run, jobs))
    names = (
        "scheduled_short",
        "scheduled_joint",
        "fixed_temperature_joint",
        "oracle_duration_only",
    )
    metrics = (
        "accuracy",
        "duration_tv_mean",
        "complete_episode_nll",
        "test_ll_per_row",
        "switches",
    )
    summary = {
        scenario: {
            name: {
                m: float(
                    np.median(
                        [
                            row[m]
                            for r in rows
                            for s in r["scenarios"]
                            if s["scenario"] == scenario
                            for n, row in s["arms"].items()
                            if n == name
                        ]
                    )
                )
                for m in metrics
            }
            for name in names
        }
        for scenario in scenarios
    }
    payload = {
        "schema": "nhsmm-model-conform-duration-diagnostic-v2",
        "execution": {
            "runtime": args.runtime_label,
            "base_commit": args.base_commit,
            "diagnostic_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "workers": args.workers,
        },
        "seeds": list(seeds),
        "joint_iters": args.joint_iters,
        "oracle_iters": args.oracle_iters,
        "runs": rows,
        "summary": summary,
        "generative_law": {
            "pmf": PMF.tolist(),
            "tail_hazard": TAIL.tolist(),
            "transition": TRANSITION.tolist(),
            "means": MEANS.tolist(),
            "sigma": SIGMA,
            "initial": [0.5, 0.5],
            "train_sequences": 6,
            "test_sequences": 4,
            "steps": 128,
        },
        "contract": {
            "labels": "evaluation only; known nuisance parameters in oracle arm",
            "censoring": "final episode always scored by survival, not completed mass",
            "oracle": "only duration logits/tail learn at temperature 1; nuisance parameters and gate fixed",
            "joint": "production optimize; fixed-temperature subclass changes only training temperature",
            "pmf": "evaluated at inference temperature 1 with production soft duration gate",
            "tv": "full finite/D+ law with residual after 1024 included",
            "comparison": "scheduled_joint versus fixed_temperature_joint matched seeds/data/iterations/lr",
            "oracle_budget": f"{args.oracle_iters} versus {args.joint_iters} iterations; not a matched-budget joint/oracle comparison",
            "status": "bounded exploratory mechanism diagnostic, no new acceptance thresholds or defaults",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
