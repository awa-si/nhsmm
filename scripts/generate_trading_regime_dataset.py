from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

REGIMES = ("bull", "bear", "range", "chaotic")
REGIME_ID = {name: i for i, name in enumerate(REGIMES)}

BASE_TRANSITION = np.asarray(
    [
        [0.00, 0.12, 0.60, 0.28],
        [0.15, 0.00, 0.52, 0.33],
        [0.43, 0.30, 0.00, 0.27],
        [0.31, 0.39, 0.30, 0.00],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class RegimeSpec:
    drift_bp: float
    sigma_bp: float
    ar1: float
    mean_reversion: float
    jump_probability: float
    jump_sigma_bp: float
    volume_multiplier: float


SPECS = {
    "bull": RegimeSpec(2.6, 18.0, 0.18, 0.00, 0.008, 42.0, 1.15),
    "bear": RegimeSpec(-3.2, 24.0, 0.12, 0.00, 0.018, 68.0, 1.35),
    "range": RegimeSpec(0.0, 11.0, -0.28, 0.08, 0.004, 25.0, 0.82),
    "chaotic": RegimeSpec(0.0, 42.0, -0.05, 0.00, 0.065, 115.0, 1.90),
}


def _duration(rng: np.RGEnerator, regime: str) -> int:
    if regime == "bull":
        value = 36 + rng.gamma(shape=4.8, scale=18.0)
        return int(np.clip(round(value), 42, 260))
    if regime == "bear":
        value = 28 + rng.weibull(1.55) * 78.0
        return int(np.clip(round(value), 30, 240))
    if regime == "range":
        center = 38.0 if rng.random() < 0.44 else 150.0
        value = rng.normal(center, 11.0 if center < 100 else 26.0)
        return int(np.clip(round(value), 18, 250))
    if regime == "chaotic":
        value = 8 + rng.gamma(shape=2.0, scale=13.0)
        return int(np.clip(round(value), 8, 105))
    raise ValueError(regime)


def _next_state(
    rng: np.RGenerator,
    current: int,
    macro_stress: float,
) -> tuple[int, np.array]:
    p = BASE_TRANSITION[current].copy()
    stress = float(np.clip(macro_stress, -2.5, 2.5))
    tilt = np.asarray(
        [-0.38 * stress, 0.34 * stress, -0.12 * abs(stress), 0.42 * abs(stress)],
        dtype=np.float64,
    )
    weights = p * np.exp(tilt)
    weights[current] = 0.0
    weights /= weights.sum()
    nxt = int(rng.choice(len(REGIMES), p=weights))
    return nxt, weights


def _rolling_features(
    close: np.ndarray, volume: np.ndarray, ret: np.ndarray
) -> dict[str, np.ndarray]:
    n = len(close)
    out = {
        "log_return_1": ret.copy(),
        "log_return_3": np.zeros(n),
        "log_return_12": np.zeros(n),
        "trend_12": np.zeros(n),
        "trend_48": np.zeros(n),
        "ewma_vol_12": np.zeros(n),
        "ewma_vol_48": np.zeros(n),
        "volume_z_48": np.zeros(n),
    }
    for t in range(n):
        if t >= 3:
            out["log_return_3"][t] = math.log(close[t] / close[t - 3])
        if t >= 12:
            out["log_return_12"][t] = math.log(close[t] / close[t - 12])
            out["trend_12"][t] = close[t] / np.mean(close[t - 11 : t + 1]) - 1.0
        if t >= 48:
            out["trend_48"][t] = close[t] / np.mean(close[t - 47 : t + 1]) - 1.0
            hist = volume[t - 47 : t + 1]
            sd = float(np.std(hist))
            out["volume_z_48"][t] = 0.0 if sd < 1e-12 else (volume[t] - float(np.mean(hist))) / sd

    for span, name in ((12, "ewma_vol_12"), (48, "ewma_vol_48")):
        alpha = 2.0 / (span + 1.0)
        variance = 0.0
        for t in range(n):
            variance = (1.0 - alpha) * variance + alpha * float(ret[t] * ret[t])
            out[name][t] = math.sqrt(max(variance, 0.0))
    return out


def generate(*, rows: int, seed: int, start_price: float) -> tuple[dict[str, np.ndarray], dict]:
    if rows < 1000:
        raise ValueError("rows must be >= 1000")
    rng = np.random.default_rng(seed)

    open_ = np.empty(rows)
    high = np.empty(rows)
    low = np.empty(rows)
    close = np.empty(rows)
    volume = np.empty(rows)
    ret = np.empty(rows)
    macro = np.empty(rows)
    state = np.empty(rows, dtype=np.int64)
    episode = np.empty(rows, dtype=np.int64)
    age = np.empty(rows, dtype=np.int64)
    planned_duration = np.empty(rows, dtype=np.int64)
    remaining = np.empty(rows, dtype=np.int64)
    transition_prob = np.empty((rows, len(REGIMES)))

    price = start_price
    previous_return = 0.0
    macro_value = -0.20
    state_id = REGIME_ID["range"]
    episode_id = 0
    t = 0
    episodes: list[dict] = []

    while t < rows:
        regime = REGIMES[state_id]
        duration = _duration(rng, regime)
        end = min(rows, t + duration)
        realized_duration = end - t
        anchor = price

        next_state, probs = _next_state(rng, state_id, macro_value)
        episodes.append(
            {
                "episode_id": episode_id,
                "start_row": t,
                "end_row_exclusive": end,
                "regime_id": state_id,
                "regime": regime,
                "planned_duration": duration,
                "realized_duration": realized_duration,
                "macro_stress_at_start": macro_value,
                "next_state_probabilities": {
                    REGIMES[i]: float(probs[i]) for i in range(len(REGIMES))
                },
            }
        )

        spec = SPECS[regime]
        for local, row in enumerate(range(t, end)):
            macro_value = 0.992 * macro_value + rng.normal(0.0, 0.055)
            macro_value = float(np.clip(macro_value, -3.0, 3.0))

            open_[row] = price
            mean_revert = 0.0
            if regime == "range":
                mean_revert = spec.mean_reversion * math.log(anchor / price)

            innovation = rng.standard_t(df=5) * (spec.sigma_bp * 1e-4)
            jump = 0.0
            if rng.random() < spec.jump_probability:
                jump = rng.normal(0.0, spec.jump_sigma_bp * 1e-4)
                if regime == "bear":
                    jump -= abs(rng.normal(0.0, 22.0e-4))

            drift = spec.drift_bp * 1e-4
            stress_drift = -0.22e-4 * macro_value
            r = drift + stress_drift + spec.ar1 * previous_return + mean_revert + innovation + jump
            r = float(np.clip(r, -0.075, 0.075))
            next_price = price * math.exp(r)

            intrabar = abs(rng.standard_t(df=6)) * spec.sigma_bp * 0.75e-4
            hi_base = max(price, next_price)
            lo_base = min(price, next_price)
            high[row] = hi_base * math.exp(intrabar)
            low[row] = lo_base * math.exp(-intrabar)
            close[row] = next_price
            ret[row] = r

            shock = min(abs(r) / max(spec.sigma_bp * 1e-4, 1e-9), 8.0)
            base_volume = 145.0 * spec.volume_multiplier
            volume[row] = base_volume * math.exp(rng.normal(0.0, 0.32)) * (1.0 + 0.18 * shock)

            macro[row] = macro_value
            state[row] = state_id
            episode[row] = episode_id
            age[row] = local + 1
            planned_duration[row] = duration
            remaining[row] = realized_duration - local - 1
            transition_prob[row] = probs

            previous_return = r
            price = next_price

        t = end
        state_id = next_state
        episode_id += 1

    features = _rolling_features(close, volume, ret)
    data = {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "macro_stress": macro,
        "range_frac": (high - low) / open_,
        "body_frac": (close - open_) / open_,
        **features,
        "gt_regime_id": state,
        "gt_episode_id": episode,
        "gt_episode_age": age,
        "gt_planned_duration": planned_duration,
        "gt_remaining_duration": remaining,
    }
    for i, name in enumerate(REGIMES):
        data[f"gt_p_next_{name}"] = transition_prob[:, i]

    metadata = {
        "schema": "nhsmm-synthetic-trading-regimes-v1",
        "seed": seed,
        "rows": rows,
        "bar_interval_minutes": 5,
        "start_price": start_price,
        "regimes": REGIMES,
        "regime_specs": {name: asdict(spec) for name, spec in SPECS.items()},
        "base_transition_matrix": BASE_TRANSITION.tolist(),
        "transition_context": {
            "feature": "macro_stress",
            "description": (
                "high stress tilts next-episode probability toward bear/chaotic; "
                "low stress toward bull/range"
            ),
        },
        "duration_families": {
            "bull": "shifted gamma, clipped [42,260]",
            "bear": "shifted Weibull, clipped [30,240]",
            "range": "bimodal Gaussian mixture, clipped [18,250]",
            "chaotic": "shifted gamma, clipped [8,105]",
        },
        "regime_row_counts": {name: int(np.sum(state == i)) for i, name in enumerate(REGIMES)},
        "regime_episode_counts": {
            name: int(sum(item["regime"] == name for item in episodes)) for name in REGIMES
        },
        "episodes": episodes,
        "feature_columns": [key for key in data if not key.startswith("gt_")],
        "ground_truth_columns": [key for key in data if key.startswith("gt_")],
        "causality": (
            "all non-gt feature columns are contemporaneous or backward-looking; "
            "gt_* columns are evaluation labels and must never be model inputs"
        ),
    }
    return data, metadata


def _split_name(index: int, rows: int) -> str:
    x = index / rows
    if x < 0.60:
        return "train"
    if x < 0.80:
        return "validation"
    return "test"


def _write_csv(path: Path, data: dict[str, np.ndarray], rows: int) -> None:
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    columns = ["timestamp", "split", *data.keys()]
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for i in range(rows):
            row = [
                (start + timedelta(minutes=5 * i)).isoformat().replace("+00:00", "Z"),
                _split_name(i, rows),
            ]
            for key, values in data.items():
                value = values[i]
                if np.issubdtype(values.dtype, np.integer):
                    row.append(int(value))
                else:
                    row.append(f"{float(value):.10g}")
            writer.writerow(row)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a ground-truth synthetic trading regime dataset."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--start-price", type=float, default=50_000.0)
    args = parser.parse_args()

    data, metadata = generate(rows=args.rows, seed=args.seed, start_price=args.start_price)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = args.output_dir / "trading_regimes.npz"
    csv_path = args.output_dir / "trading_regimes.csv.gz"
    manifest_path = args.output_dir / "manifest.json"

    np.savez_compressed(npz_path, **data)
    _write_csv(csv_path, data, args.rows)

    metadata["files"] = {
        "trading_regimes.npz": {
            "sha256": _sha256(npz_path),
            "format": "NumPy NPZ",
        },
        "trading_regimes.csv.gz": {
            "sha256": _sha256(csv_path),
            "format": "gzip CSV",
        },
    }
    metadata["chronological_split"] = {
        "train": [0, int(args.rows * 0.60)],
        "validation": [int(args.rows * 0.60), int(args.rows * 0.80)],
        "test": [int(args.rows * 0.80), args.rows],
    }
    manifest_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "rows": args.rows,
                "episodes": len(metadata["episodes"]),
                "regime_row_counts": metadata["regime_row_counts"],
                "regime_episode_counts": metadata["regime_episode_counts"],
                "npz_sha256": metadata["files"]["trading_regimes.npz"]["sha256"],
                "csv_sha256": metadata["files"]["trading_regimes.csv.gz"]["sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
