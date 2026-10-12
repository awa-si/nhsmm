"""Audit Trading ground truth, censoring, and exact duration-law compatibility."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
import torch

from nhsmm.filtering import duration_log_hazard
from scripts.generate_trading_regime_dataset import BASE_TRANSITION, REGIMES, _duration, generate
from scripts.validate_trading_regime_usefulness import FEATURES, _run_medians, _windows


def duration_cdf(name, x):
    x = np.asarray(x, dtype=np.float64)
    if name in ("bull", "chaotic"):
        shift, shape, scale = (36, 4.8, 18) if name == "bull" else (8, 2, 13)
        values = torch.tensor(np.maximum((x - shift) / scale, 0), dtype=torch.float64)
        return torch.special.gammainc(torch.full_like(values, shape), values).numpy()
    if name == "bear":
        return -np.expm1(-np.maximum((x - 28) / 78, 0) ** 1.55)
    if name == "range":

        def normal(mean, std):
            return np.array([0.5 * (1 + math.erf((v - mean) / std / math.sqrt(2))) for v in x])

        return 0.44 * normal(38, 11) + 0.56 * normal(150, 26)
    raise ValueError(name)


def duration_pmf(name):
    lower, upper = {"bull": (42, 260), "bear": (30, 240), "range": (18, 250), "chaotic": (8, 105)}[
        name
    ]
    ages = np.arange(1, upper + 1)
    cdf = duration_cdf(name, ages + 0.5)
    cdf[: lower - 1] = 0
    cdf[-1] = 1
    pmf = np.maximum(np.diff(np.r_[0, cdf]), 0)
    np.testing.assert_allclose(pmf.sum(), 1, rtol=0, atol=1e-12)
    return pmf


def geometric_projection(pmf, width=64):
    """KL-optimal D+ geometric projection, not a minimum-TV assertion."""
    ages = np.arange(1, len(pmf) + 1)
    tail_mass = float(pmf[width - 1 :].sum())
    hazard = tail_mass / float(np.sum(pmf[width - 1 :] * (ages[width - 1 :] - width + 1)))
    projected = pmf.copy()
    projected[width - 1 :] = tail_mass * hazard * (1 - hazard) ** (ages[width - 1 :] - width)
    overflow = tail_mass * (1 - hazard) ** (len(pmf) - width + 1)
    positive = pmf > 0
    kl = float(np.sum(pmf[positive] * np.log(pmf[positive] / projected[positive])))
    tv = float((np.abs(pmf - projected).sum() + overflow) / 2)
    return {
        "tail_mass": tail_mass,
        "tail_hazard": hazard,
        "kl_nats_per_episode": kl,
        "projection_tv": tv,
        "probability_beyond_true_max": overflow,
    }


def audit_laws():
    laws = {}
    for index, name in enumerate(REGIMES):
        pmf = duration_pmf(name)
        survival = np.cumsum(pmf[::-1])[::-1]
        expected = pmf / survival
        tensor = torch.tensor(pmf, dtype=torch.float64).log()[None, None]
        end, _ = duration_log_hazard(tensor)
        np.testing.assert_allclose(end.exp().numpy()[0, 0], expected, rtol=0, atol=1e-12)
        rng = np.random.default_rng(11400 + index)
        samples = np.array([_duration(rng, name) for _ in range(20000)])
        empirical = np.bincount(samples, minlength=len(pmf) + 1)[1 : len(pmf) + 1] / len(samples)
        cdf_error = float(np.max(np.abs(np.cumsum(empirical) - np.cumsum(pmf))))
        if cdf_error > 0.02:
            raise AssertionError(f"duration sampler/CDF mismatch {name}: {cdf_error}")
        laws[name] = {
            "pmf": pmf.tolist(),
            "max_duration": len(pmf),
            "mean_duration": float(pmf @ np.arange(1, len(pmf) + 1)),
            "hazards": {
                str(age): float(expected[age - 1]) if age <= len(pmf) else None
                for age in (64, 96, 128, 160, 224, 260)
            },
            "d64_projection": geometric_projection(pmf),
            "sampler_cdf_error_20000": cdf_error,
        }
    return laws


def boundary_probabilities(state, macro):
    stress = float(np.clip(macro, -2.5, 2.5))
    tilt = np.array([-0.38 * stress, 0.34 * stress, -0.12 * abs(stress), 0.42 * abs(stress)])
    weights = BASE_TRANSITION[state] * np.exp(tilt)
    return weights / weights.sum()


def audit_labels(data, metadata):
    rows = len(data["close"])
    np.testing.assert_array_equal(
        data["gt_latent_remaining_duration"], data["gt_planned_duration"] - data["gt_episode_age"]
    )
    np.testing.assert_allclose(data["open"][1:], data["close"][:-1], rtol=0, atol=0)
    np.testing.assert_allclose(
        np.log(data["close"] / data["open"]), data["log_return_1"], rtol=0, atol=1e-12
    )
    assert np.all(data["high"] >= np.maximum(data["open"], data["close"]))
    assert np.all(data["low"] <= np.minimum(data["open"], data["close"]))
    assert np.all(data["volume"] > 0)
    probabilities = np.column_stack([data[f"gt_p_next_{name}"] for name in REGIMES])
    for count in (3, 12):
        cumulative = np.r_[0, np.cumsum(data["log_return_1"])]
        expected = cumulative[count + 1 :] - cumulative[1:-count]
        np.testing.assert_allclose(
            data[f"log_return_{count}"][count:], expected, rtol=0, atol=1e-10
        )
    stop = 0
    completed = {name: [] for name in REGIMES}
    censored = []
    for i, episode in enumerate(metadata["episodes"]):
        start, end = episode["start_row"], episode["end_row_exclusive"]
        planned, realized = episode["planned_duration"], episode["realized_duration"]
        state = episode["regime_id"]
        assert start == stop and end - start == realized and realized <= planned
        assert episode["right_censored"] == (realized < planned)
        np.testing.assert_array_equal(
            data["gt_episode_right_censored"][start:end], int(realized < planned)
        )
        np.testing.assert_array_equal(data["gt_episode_id"][start:end], i)
        np.testing.assert_array_equal(data["gt_regime_id"][start:end], state)
        np.testing.assert_array_equal(data["gt_episode_age"][start:end], np.arange(1, realized + 1))
        np.testing.assert_array_equal(data["gt_planned_duration"][start:end], planned)
        np.testing.assert_array_equal(
            data["gt_remaining_duration"][start:end], np.arange(realized - 1, -1, -1)
        )
        expected = boundary_probabilities(state, data["macro_stress"][end - 1])
        np.testing.assert_allclose(
            probabilities[start:end], np.broadcast_to(expected, (realized, 4)), rtol=0, atol=1e-12
        )
        assert probabilities[start, state] == 0
        assert episode["macro_stress_at_boundary"] == data["macro_stress"][end - 1]
        if realized < planned:
            assert end == rows
            censored.append(
                {
                    "episode_id": i,
                    "realized_duration": realized,
                    "planned_duration": planned,
                    "stored_remaining_at_last_row": 0,
                    "latent_remaining_after_last_row": planned - realized,
                }
            )
        else:
            completed[REGIMES[state]].append(realized)
        if end < rows:
            assert data["gt_regime_id"][end] != state and data["gt_episode_age"][end] == 1
        stop = end
    assert stop == rows
    return {
        "episodes": len(metadata["episodes"]),
        "terminal_right_censoring": censored,
        "completed_episode_medians": {
            k: float(np.median(v)) if v else None for k, v in completed.items()
        },
        "completed_episode_counts": {k: len(v) for k, v in completed.items()},
        "clipped_return_rows": int(np.sum(np.abs(data["log_return_1"]) == 0.075)),
    }


def window_audit(data, metadata, start, end, count, width):
    anchors = np.linspace(start, end - width, count, dtype=int)
    windows = _windows(data, start, end, count, window=width)
    ledger = metadata["episodes"]
    complete = {k: [] for k in REGIMES}
    left = right = warm_continuation = 0
    for a in anchors:
        left += int(data["gt_episode_age"][a] > 1)
        right += int(
            data["gt_episode_age"][a + width - 1] < data["gt_planned_duration"][a + width - 1]
        )
        warm_continuation += int(
            data["gt_episode_id"][a + 64] == data["gt_episode_id"][a]
            and data["gt_episode_age"][a] > 1
        )
        for episode in ledger:
            if (
                episode["start_row"] >= a
                and episode["end_row_exclusive"] < a + width
                and episode["realized_duration"] == episode["planned_duration"]
            ):
                complete[episode["regime"]].append(episode["planned_duration"])
    naive = _run_medians(windows[2])
    return {
        "anchors": anchors.tolist(),
        "windows": count,
        "left_censored_windows": left,
        "right_censored_windows": right,
        "same_left_censored_episode_after_warmup64": warm_continuation,
        "complete_episode_counts": {k: len(v) for k, v in complete.items()},
        "complete_episode_medians": {
            k: float(np.median(v)) if v else None for k, v in complete.items()
        },
        "naive_argmax_run_medians": {
            k: float(naive[i]) if np.isfinite(naive[i]) else None for i, k in enumerate(REGIMES)
        },
    }


def run(seed):
    data, metadata = generate(
        rows=60000, seed=seed, start_price=50000, transition_context_mode="episode_end"
    )
    prior_path = Path(f"docs/validation/results/hmm-control-step10-data{seed}-native-20261012.json")
    prior_bytes = prior_path.read_bytes()
    prior = json.loads(prior_bytes)["run"]
    feature_sha = hashlib.sha256(
        np.column_stack([data[f] for f in FEATURES]).astype(float).tobytes()
    ).hexdigest()
    assert feature_sha == prior["dataset_feature_sha256"]
    short, _ = generate(
        rows=1100, seed=seed, start_price=50000, transition_context_mode="episode_end"
    )
    for name in metadata["feature_columns"]:
        np.testing.assert_array_equal(data[name][:1100], short[name])
    return {
        "generator_seed": seed,
        "feature_sha256": feature_sha,
        "prior_report_sha256": hashlib.sha256(prior_bytes).hexdigest(),
        "ground_truth": audit_labels(data, metadata),
        "prefix_non_gt_columns_verified": len(metadata["feature_columns"]),
        "windows": {
            f"{label}_{split}": window_audit(data, metadata, start, end, count, width)
            for label, width, train_count, test_count in (
                ("pilot240", 240, 8, 4),
                ("maintained480", 480, 24, 8),
            )
            for split, start, end, count in (
                ("train", 0, 36000, train_count),
                ("test", 48000, 60000, test_count),
            )
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    paths = [
        "scripts/diagnose_trading_ground_truth.py",
        "scripts/generate_trading_regime_dataset.py",
        "scripts/validate_trading_regime_usefulness.py",
        "nhsmm/filtering.py",
        "nhsmm/distributions/default.py",
    ]
    payload = {
        "schema": "nhsmm-trading-ground-truth-audit-v1",
        "execution": {
            "runtime": "openai-native-workspace",
            "base_commit": args.base_commit,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "workers": 1,
            "source_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
        },
        "contract": {
            "scope": "generator v2, three frozen seeds; exact clipped/rounded duration PMFs; ledger/row labels and causal prefixes; benchmark censoring; no parameter training",
            "tail_projection": "static no-self-reset D64+geometric tail KL projection; not a lower bound on every context-modulated/state-split observable run model",
            "duration_sampler_control": "20000 fresh independent duration draws per state; fixed CDF error bound0.02",
            "remaining": "stored gt_remaining_duration counts observed file rows, not necessarily latent remaining life for terminal censored episode",
        },
        "duration_laws": audit_laws(),
        "runs": [run(seed) for seed in (424242, 424243, 424244)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "laws": {
                    k: {"mean": v["mean_duration"], "projection": v["d64_projection"]}
                    for k, v in payload["duration_laws"].items()
                },
                "datasets": [
                    {
                        "seed": r["generator_seed"],
                        "ground_truth": r["ground_truth"],
                        "windows": r["windows"]["pilot240_test"],
                    }
                    for r in payload["runs"]
                ],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
