"""Bounded structural diagnostics; no production model or default changes."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from nhsmm import ModelConfig, NHSMM
from nhsmm.filtering import filter_model_sequence
from scripts.diagnose_model_conform_duration import D, TAIL, generate


def nonidentifiability():
    models = []
    for hazard, reset in [(0.2, 0.5), (0.125, 0.2)]:
        m = NHSMM(
            ModelConfig(
                n_states=2,
                n_features=1,
                max_duration=1,
                causal=True,
                duration_tail=True,
                use_context_encoder=False,
                seed=7,
            ),
            device="cpu",
        )
        m.initialize_distributions()
        m.double()
        m.eval()
        with torch.no_grad():
            m.dist.initial.logits.fill_(0)
            m.dist.emission.mu.copy_(torch.tensor([[-1.0], [1.0]], dtype=torch.float64))
            m.dist.emission.log_var.fill_(0)
            m.dist.duration.tail_end_probability.fill_(hazard)
            matrix = torch.tensor([[reset, 1 - reset], [1 - reset, reset]], dtype=torch.float64)
            m.dist.transition.logits.copy_(matrix.log()[:, None, :])
        models.append(m)
    torch.manual_seed(8)
    x = torch.randn(3, 20, 1, dtype=torch.float64)
    with torch.inference_mode():
        ll = [float(m.log_likelihood(x, reduce=True)) for m in models]
        posterior = [filter_model_sequence(m, x).state_posterior for m in models]
    difference = float((posterior[0] - posterior[1]).abs().max())
    assert abs(ll[0] - ll[1]) < 1e-12 and difference < 1e-12
    return {
        "hazards": [0.2, 0.125],
        "self_reset_probabilities": [0.5, 0.2],
        "episode_means": [5.0, 8.0],
        "observable_switch_hazards": [0.1, 0.1],
        "observable_run_means": [10.0, 10.0],
        "log_likelihoods": ll,
        "likelihood_absolute_difference": abs(ll[0] - ll[1]),
        "posterior_max_absolute_difference": difference,
        "scope": "exact equivalence in allowed D=1 geometric-tail subfamily; not proof for every finite law",
    }


class IdentityEncoder(nn.Module):
    causal = True

    def forward(self, x, mask=None):
        return x


def neural_normalization():
    m = NHSMM(
        ModelConfig(
            n_states=2,
            n_features=1,
            max_duration=2,
            causal=True,
            context_dim=1,
            dropout=0,
            seed=4,
        ),
        encoder=IdentityEncoder(),
        device="cpu",
    )
    m.initialize_distributions()
    m.eval()
    measurements = []
    with torch.no_grad():
        for p in m.dist.parameters():
            p.zero_()
        em = m.dist.emission
        em.mu[1, 0] = 4.0
        em.log_var.fill_(math.log(math.expm1(1.0)))
        em.delta_scale.fill_(0.5)
        em._proj.weight.fill_(1.0)
        em.context_net[0].weight[0, 0] = 0.001
        em.context_net[0].weight[1, 0] = -0.001
        em.context_net[1].weight.fill_(1.0)
        em.context_net[3].weight[0, 0] = 1.0
        em.context_net[3].weight[1, 0] = -1.0
        for count in [28001, 56001]:
            grid = torch.linspace(-12, 16, count)
            x = grid[:, None, None]
            density = m.log_likelihood(x).reshape(-1).exp()
            fixed = em.log_prob(x, context=torch.zeros_like(x)).exp().mean(-1).reshape(-1)
            area = float(torch.trapezoid(density, grid))
            control = float(torch.trapezoid(fixed, grid))
            assert area > 1.05 and abs(control - 1) < 2e-6
            measurements.append(
                {"points": count, "score_integral": area, "fixed_context_integral": control}
            )
    assert abs(measurements[0]["score_integral"] - measurements[1]["score_integral"]) < 2e-6
    return {
        "interval": [-12, 16],
        "measurements": measurements,
        "scope": "supported injected causal identity encoder, two-state one-row model, Float32",
        "interpretation": "current-observation-conditioned emission scores need not form a normalized density over observations",
        "limitations": "counterexample to general generative-likelihood claim; not a measured defect in encoderless pilot fits",
    }


def observable_run_law(pmf, hazard, reset, horizon=4096):
    """Independent probability-space absorbing chain over episode age."""
    width = len(pmf)
    end = pmf / np.cumsum(pmf[::-1])[::-1]
    end[-1] = hazard
    matrix = np.zeros((width, width))
    for age in range(width):
        matrix[age, min(age + 1, width - 1)] += 1 - end[age]
        matrix[age, 0] += end[age] * reset[age]
    exit_probability = 1 - matrix.sum(1)
    frontier = np.eye(width)[0]
    masses = []
    for _ in range(horizon):
        masses.append(float(frontier @ exit_probability))
        frontier = frontier @ matrix
    masses.append(float(frontier.sum()))
    mean = float(np.linalg.solve(np.eye(width) - matrix, np.ones(width))[0])
    return np.asarray(masses), mean


def fitted_run_laws(pilot):
    truth = np.asarray(pilot["generative_law"]["pmf"])
    output = {}
    for scenario in pilot["runs"][0]["scenarios"]:
        tail = scenario["scenario"] == "tail"
        hazards = np.asarray(pilot["generative_law"]["tail_hazard"]) if tail else np.ones(2)
        actual = [observable_run_law(truth[k], hazards[k], np.repeat(0.02, D)) for k in range(2)]
        arms = {}
        for name, row in scenario["arms"].items():
            fitted = [
                observable_run_law(
                    np.asarray(row["pmf"])[k],
                    row["tail_hazard"][k],
                    np.asarray(row["self_transition_probability_by_age"])[k],
                )
                for k in range(2)
            ]
            arms[name] = {
                "episode_duration_tv_mean": row["duration_tv_mean"],
                "observable_run_tv_by_state": [
                    float(np.abs(fitted[k][0] - actual[k][0]).sum() / 2) for k in range(2)
                ],
                "observable_run_mean_by_state": [v[1] for v in fitted],
                "true_observable_run_mean_by_state": [v[1] for v in actual],
            }
        output[scenario["scenario"]] = arms
    # Independent analytic geometric control.
    mass, mean = observable_run_law(np.ones(1), 0.2, np.array([0.5]), horizon=100)
    np.testing.assert_allclose(mass[:-1], 0.1 * 0.9 ** np.arange(100), atol=1e-14)
    assert abs(mean - 10) < 1e-12
    return output


def tail_information():
    _, _, ledger = generate(20901, tail=True)
    rows = []
    for state in range(2):
        lengths = [
            e["observed"]
            for row in ledger
            for e in row
            if e["state"] == state and not e["censored"] and e["observed"] >= D
        ]
        exposure = sum(length - D + 1 for length in lengths)
        rows.append(
            {
                "state": state,
                "completed_tail_lengths": lengths,
                "tail_end_count": len(lengths),
                "tail_exposure": exposure,
                "completed_episode_hazard_mle": len(lengths) / exposure if exposure else None,
                "generator_hazard": float(TAIL[state]),
            }
        )
    return {
        "rows": rows,
        "scope": "known labeled training episodes; completed-tail MLE excludes censored contributions",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-commit", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    pilot = json.loads(args.pilot.read_text())
    result = {
        "schema": "nhsmm-structural-diagnostics-v1",
        "source_commit": args.base_commit,
        "pilot_sha256": hashlib.sha256(args.pilot.read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "nonidentifiability": nonidentifiability(),
        "neural_normalization": neural_normalization(),
        "observable_run_laws": fitted_run_laws(pilot),
        "tail_information": tail_information(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "observable_run_laws"}))


if __name__ == "__main__":
    main()
