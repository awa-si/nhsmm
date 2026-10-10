"""Compare native age-tied and age-specific transition logits under known boundaries.

Both arms use the same NHSMM distribution, optimizer and train/test episodes.
Age tying is implemented by a gradient-sharing hook, without changes to NHSMM.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from nhsmm import ModelConfig, NHSMM

K = 4
D = 64
REGIMES = ("bull", "bear", "range", "chaotic")


def run(dataset: Path, *, steps: int = 220) -> dict:
    with np.load(dataset) as z:
        data = {key: z[key] for key in z.files}

    y = data["gt_regime_id"].astype(np.int64)
    episodes = data["gt_episode_id"]
    starts = np.r_[0, np.flatnonzero(episodes[1:] != episodes[:-1]) + 1]
    boundary = starts[1:] - 1
    cut = int(0.6 * len(y))
    train = starts[1:] < cut
    test = starts[1:] >= cut
    ages = np.minimum(np.maximum(data["gt_episode_age"][boundary].astype(np.int64) - 1, 0), D - 1)
    source = y[boundary]
    target = y[boundary + 1]
    gt = np.column_stack([data["gt_p_next_" + name] for name in REGIMES])

    torch.set_num_threads(1)
    tr_src = torch.tensor(source[train])
    tr_age = torch.tensor(ages[train])
    tr_target = torch.tensor(target[train])
    te_src = torch.tensor(source[test])
    te_age = torch.tensor(ages[test])
    te_target = torch.tensor(target[test])
    te_gt = gt[boundary[test]]

    def experiment(seed: int, tied: bool) -> dict:
        torch.manual_seed(seed)
        config = ModelConfig(
            n_states=K,
            n_features=8,
            max_duration=D,
            causal=True,
            duration_tail=True,
            dropout=0.0,
            seed=seed,
            transition_init_mode="uniform",
            use_context_encoder=False,
            verbose=False,
        )
        model = NHSMM(config, device="cpu")
        model.initialize_distributions(jitter=0.0)
        dist = model.dist.transition
        if tied:
            # Identical gradients and initial values keep all age-indexed rows
            # equal during Adam updates; the forward call stays fully native.
            handle = dist.logits.register_hook(
                lambda grad: grad.sum(dim=1, keepdim=True).expand_as(grad)
            )
        else:
            handle = None
        optimizer = torch.optim.Adam([dist.logits], lr=0.012)

        def logprob():
            matrix = dist.log_matrix()[0, 0]
            return matrix[tr_src, tr_age, tr_target]

        initial_nll = float((-logprob().mean()).detach())
        for _ in range(steps):
            optimizer.zero_grad()
            loss = -logprob().mean()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            probabilities = dist.log_matrix()[0, 0].exp()
            predicted = probabilities[te_src, te_age]
            train_nll = float((-logprob().mean()).detach())
            test_nll = float(-predicted[torch.arange(len(te_target)), te_target].log().mean())
            mae = float(np.abs(predicted.numpy() - te_gt).mean())
            max_age_difference = float(
                (probabilities - probabilities[:, :1, :]).abs().max()
            )
        if handle is not None:
            handle.remove()
        return {
            "seed": seed,
            "age_tied": tied,
            "initial_train_nll": initial_nll,
            "final_train_nll": train_nll,
            "test_nll": test_nll,
            "test_transition_mae": mae,
            "max_age_probability_difference": max_age_difference,
        }

    outcomes = [
        experiment(seed, tied)
        for tied in (True, False)
        for seed in (2311, 2312)
    ]
    counts = np.zeros((K, K), dtype=float)
    np.add.at(counts, (source[train], target[train]), 1)
    empirical = counts / counts.sum(axis=1, keepdims=True)
    empirical_pred = empirical[source[test]]
    empirical_nll = float(
        -np.log(np.clip(empirical_pred[np.arange(int(test.sum())), target[test]], 1e-12, 1)).mean()
    )
    return {
        "dataset": str(dataset),
        "protocol": "true episode boundaries, chronological 60/40 split, native NHSMM transition, 220 Adam steps, no context",
        "train_boundaries": int(train.sum()),
        "test_boundaries": int(test.sum()),
        "runs": outcomes,
        "empirical_pooled_baseline": {
            "test_nll": empirical_nll,
            "test_transition_mae": float(np.abs(empirical_pred - te_gt).mean()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/reference/trading-regimes-v1/trading_regimes.npz"))
    parser.add_argument("--output", type=Path, default=Path("/data/workspace/stage2/native-age-tied-transition.json"))
    args = parser.parse_args()
    result = run(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
