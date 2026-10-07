"""Cross-check NHSMM explicit-duration inference against an external HSMM.

This harness is intentionally outside the canonical pytest suite. It compares
fixed-parameter, non-causal, stationary explicit-duration inference against
``gaussian-hsmm==0.1.0``. Training/optimizer behavior is excluded.

Reference environment requirements:

    pip install gaussian-hsmm==0.1.0

The package currently imports ``hmmlearn`` for ``fit()`` initialization, but
this harness uses only its explicit-duration ``score()`` and ``decode()`` paths.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Any
from unittest.mock import patch

import numpy as np
import torch

from nhsmm import ModelConfig, NHSMM
from nhsmm.context import SequenceSet


@dataclass(frozen=True)
class ExternalReferenceResult:
    cases: int
    max_abs_log_likelihood_diff: float
    median_abs_log_likelihood_diff: float
    viterbi_mismatches: int
    passed: bool


def _load_reference() -> Any:
    try:
        from gaussian_hsmm import GaussianHSMM
    except ImportError as exc:
        raise RuntimeError(
            "external reference unavailable; install gaussian-hsmm==0.1.0 "
            "in a dedicated reference environment"
        ) from exc
    return GaussianHSMM


def _gaussian_log_prob(
    observations: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    delta = observations[:, None, :] - means[None, :, :]
    return (
        -0.5
        * (np.log(2.0 * np.pi * variances)[None, :, :] + np.square(delta) / variances[None, :, :])
    ).sum(axis=-1)


def _sequence_set(observations: np.ndarray, log_probs: np.ndarray) -> SequenceSet:
    timesteps = observations.shape[0]
    sequence = torch.tensor(observations, dtype=torch.float64).unsqueeze(0)
    return SequenceSet(
        sequences=sequence,
        lengths=torch.tensor([timesteps]),
        masks=torch.ones(1, timesteps, 1, dtype=torch.bool),
        contexts=torch.empty(1, timesteps, 0, dtype=torch.float64),
        canonical=torch.empty(1, 1, 0, dtype=torch.float64),
        log_probs=torch.tensor(log_probs, dtype=torch.float64).unsqueeze(0),
    )


def _external_model(
    reference_cls: Any,
    *,
    n_states: int,
    max_duration: int,
    n_features: int,
    initial: np.ndarray,
    transition: np.ndarray,
    duration: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> Any:
    model = reference_cls(
        n_components=n_states,
        covariance_type="diag",
        max_duration=max_duration,
        n_iter=1,
    )
    model.n_features_ = n_features
    model.startprob_ = initial.copy()
    model.transmat_ = transition.copy()
    model.means_ = means.copy()
    model.covars_ = variances.copy()

    # gaussian-hsmm stores duration 0 as impossible and 1..D as supported.
    duration_probs = np.zeros((n_states, max_duration + 1), dtype=float)
    duration_probs[:, 1:] = duration
    model.duration_probs_ = duration_probs
    model._log_duration_probs_ = np.full_like(duration_probs, -np.inf)
    model._log_duration_probs_[:, 1:] = np.log(duration)
    model.duration_means_ = (duration * np.arange(1, max_duration + 1, dtype=float)[None, :]).sum(
        axis=1
    )
    model.n_parameters_ = 1
    return model


def run_crosscheck(*, cases: int = 20, seed_base: int = 1000) -> ExternalReferenceResult:
    reference_cls = _load_reference()
    n_states, max_duration, n_features, timesteps = 3, 5, 2, 11
    likelihood_diffs: list[float] = []
    viterbi_mismatches = 0

    for offset in range(cases):
        rng = np.random.default_rng(seed_base + offset)
        initial = rng.dirichlet(np.ones(n_states) * 1.7)
        transition = np.zeros((n_states, n_states), dtype=float)
        for state in range(n_states):
            destinations = [candidate for candidate in range(n_states) if candidate != state]
            transition[state, destinations] = rng.dirichlet(np.ones(n_states - 1) * 1.3)
        duration = np.vstack(
            [rng.dirichlet(np.linspace(1.1, 2.0, max_duration)) for _ in range(n_states)]
        )
        means = rng.normal(0.0, 2.0, (n_states, n_features))
        variances = np.exp(rng.normal(-0.2, 0.35, (n_states, n_features)))
        observations = rng.normal(0.0, 2.0, (timesteps, n_features))
        emission_log_probs = _gaussian_log_prob(observations, means, variances)

        external = _external_model(
            reference_cls,
            n_states=n_states,
            max_duration=max_duration,
            n_features=n_features,
            initial=initial,
            transition=transition,
            duration=duration,
            means=means,
            variances=variances,
        )
        external_ll = float(external.score(observations))
        _, external_path = external.decode(observations)

        ours = NHSMM(
            ModelConfig(
                n_states=n_states,
                n_features=n_features,
                max_duration=max_duration,
                causal=False,
                dropout=0.0,
                verbose=False,
            ),
            device="cpu",
        )
        ours.initialize_distributions(jitter=0.0)
        ours.eval()
        sequence = _sequence_set(observations, emission_log_probs)

        initial_log = (
            torch.tensor(np.log(initial), dtype=torch.float64)
            .view(1, 1, n_states)
            .expand(1, timesteps, n_states)
        )
        duration_log = (
            torch.tensor(np.log(duration), dtype=torch.float64)
            .view(1, 1, n_states, max_duration)
            .expand(1, timesteps, n_states, max_duration)
        )
        transition_log_np = np.full((n_states, n_states), -np.inf, dtype=float)
        supported = transition > 0.0
        transition_log_np[supported] = np.log(transition[supported])
        transition_log = (
            torch.tensor(transition_log_np, dtype=torch.float64)
            .view(1, 1, n_states, 1, n_states)
            .expand(1, timesteps, n_states, max_duration, n_states)
        )

        with (
            patch.object(ours.dist.initial, "log_matrix", return_value=initial_log),
            patch.object(ours.dist.duration, "log_matrix", return_value=duration_log),
            patch.object(ours.dist.transition, "log_matrix", return_value=transition_log),
        ):
            alpha = ours.forward(sequence)
            ours_ll = float(torch.logsumexp(alpha[0, timesteps - 1].reshape(-1), dim=0))
            ours_path = np.asarray(ours._viterbi(sequence)[0].tolist(), dtype=int)

        likelihood_diffs.append(abs(ours_ll - external_ll))
        if not np.array_equal(ours_path, external_path):
            viterbi_mismatches += 1

    max_diff = max(likelihood_diffs)
    median_diff = float(np.median(likelihood_diffs))
    return ExternalReferenceResult(
        cases=cases,
        max_abs_log_likelihood_diff=max_diff,
        median_abs_log_likelihood_diff=median_diff,
        viterbi_mismatches=viterbi_mismatches,
        passed=viterbi_mismatches == 0 and max_diff <= 1e-12,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    if args.cases < 1:
        parser.error("--cases must be >= 1")

    result = run_crosscheck(cases=args.cases, seed_base=args.seed_base)
    payload = asdict(result)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
