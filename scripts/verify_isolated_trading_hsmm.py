"""Finite-D260 oracle duration learning and independent censored HSMM recursion."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import types
from pathlib import Path

import numpy as np
import torch

from nhsmm import ModelConfig, NHSMM
from nhsmm.distributions.default import Duration
from nhsmm.filtering import filter_model_sequence
from scripts.diagnose_trading_ground_truth import duration_pmf
from scripts.generate_trading_regime_dataset import BASE_TRANSITION, REGIMES, _duration

D = 260
MEANS = np.array([-6.0, -2.0, 2.0, 6.0])
SIGMA = 0.35


def true_law():
    return np.stack(
        [np.pad(duration_pmf(name), (0, D - len(duration_pmf(name)))) for name in REGIMES]
    )


def masked_duration(self, logits, mask=None):
    return Duration._apply_constraints(self, logits, mask=self.oracle_support)


def make_model(pmf, *, uniform=False, sigma=SIGMA):
    model = NHSMM(
        ModelConfig(
            n_states=4,
            n_features=1,
            max_duration=D,
            duration_tail=False,
            causal=True,
            use_context_encoder=False,
            dropout=0,
            seed=901,
        ),
        device="cpu",
    )
    model.initialize_distributions()
    model.double()
    support = pmf > 0
    model.dist.duration.register_buffer("oracle_support", torch.tensor(support))
    model.dist.duration._apply_constraints = types.MethodType(masked_duration, model.dist.duration)
    with torch.no_grad():
        model.dist.initial.logits.fill_(-math.log(4))
        model.dist.emission.mu.copy_(torch.tensor(MEANS[:, None]))
        model.dist.emission.log_var.fill_(math.log(math.expm1(sigma**2)))
        model.duration_logits_bias.fill_(1)
        model.dist.duration.logits.copy_(torch.tensor(np.log(np.maximum(pmf, 1e-300))))
        if uniform:
            model.dist.duration.logits.zero_()
        model.dist.transition.logits.copy_(
            torch.tensor(np.log(np.maximum(BASE_TRANSITION, 1e-300)))[:, None, :].expand(4, D, 4)
        )
        mask = ~torch.eye(4, dtype=torch.bool)[:, None, :].expand(4, D, 4)
        model.dist.transition._structural_constraint.copy_(mask)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model


def model_law(model):
    return model.dist.duration.log_matrix(T=1, temperature=1, soft_dmax=model.duration_logits_bias)[
        0, 0
    ].exp()


def censored_counts(seed, count=8192):
    rng = np.random.default_rng(seed)
    complete = np.zeros((4, D))
    censored = np.zeros_like(complete)
    for k, name in enumerate(REGIMES):
        for _ in range(count):
            length = _duration(rng, name)
            cutoff = int(rng.integers(80, 301))
            if length < cutoff:
                complete[k, length - 1] += 1
            else:
                censored[k, cutoff - 1] += 1
    return complete, censored


def count_ll(pmf, complete, censored):
    survival = np.cumsum(pmf[:, ::-1], axis=1)[:, ::-1]
    a = complete > 0
    b = censored > 0
    return float(np.sum(complete[a] * np.log(pmf[a])) + np.sum(censored[b] * np.log(survival[b])))


def labelled_em(complete, censored, support, iterations=200):
    p = support / support.sum(1, keepdims=True)
    history = [count_ll(p, complete, censored)]
    for _ in range(iterations):
        survival = np.cumsum(p[:, ::-1], axis=1)[:, ::-1]
        ratios = np.divide(censored, survival, out=np.zeros_like(censored), where=survival > 0)
        expected = complete + p * np.cumsum(ratios, axis=1)
        p = expected / expected.sum(1, keepdims=True)
        history.append(count_ll(p, complete, censored))
    if min(np.diff(history)) < -1e-7:
        raise AssertionError("labelled EM objective decreased")
    return p, history


def fit_duration(pmf, complete, censored, iterations=1200):
    model = make_model(pmf, uniform=True)
    model.dist.duration.logits.requires_grad_(True)
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    c, r = torch.tensor(complete), torch.tensor(censored)
    optimizer = torch.optim.Adam([model.dist.duration.logits], lr=0.05)
    history = []
    for step in range(iterations):
        optimizer.zero_grad()
        logp = model.dist.duration.log_matrix(
            T=1, temperature=1, soft_dmax=model.duration_logits_bias
        )[0, 0]
        logs = torch.logcumsumexp(logp.flip(-1), dim=-1).flip(-1)
        loss = -((c * logp).sum() + (r * logs).sum()) / (c.sum() + r.sum())
        loss.backward()
        optimizer.step()
        if step in (0, 399, iterations - 1):
            history.append(
                {"completed_updates": step + 1, "pre_update_nll_per_record": float(loss.detach())}
            )
    for n, p in model.named_parameters():
        if n in frozen and not torch.equal(p.detach(), frozen[n]):
            raise AssertionError("frozen nuisance parameter changed")
    return model, history


def lse(x, axis):
    maximum = np.max(x, axis=axis, keepdims=True)
    safe = np.where(np.isfinite(maximum), maximum, 0)
    with np.errstate(divide="ignore"):
        result = safe + np.log(np.exp(x - safe).sum(axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis)


def segment_reference(emissions, pmf, transition, initial):
    """Independent completed-segment forward recursion; terminal uses survival."""
    steps, states = emissions.shape
    width = pmf.shape[1]
    with np.errstate(divide="ignore"):
        logp = np.log(pmf)
        logs = np.log(np.cumsum(pmf[:, ::-1], axis=1)[:, ::-1])
        logt = np.log(transition)
        logi = np.log(initial)
    prefix = np.vstack([np.zeros(states), np.cumsum(emissions, axis=0)])
    incoming = np.full((steps + 1, states), -np.inf)
    incoming[0] = logi
    posterior = np.empty((steps, states))
    final_ll = None
    for t in range(steps):
        ages = np.arange(1, min(width, t + 1) + 1)
        starts = t - ages + 1
        segments = incoming[starts] + prefix[t + 1] - prefix[starts]
        ended = lse(segments + logp[:, ages - 1].T, 0)
        incoming[t + 1] = lse(ended[:, None] + logt, 0)
        alive = lse(segments + logs[:, ages - 1].T, 0)
        final_ll = float(lse(alive, 0))
        posterior[t] = alive - final_ll
    return final_ll, posterior


def sequences(seed, count=8, steps=600, sigma=SIGMA):
    rng = np.random.default_rng(seed)
    x = np.empty((count, steps, 1), dtype=np.float32)
    labels = np.empty((count, steps), dtype=int)
    for b in range(count):
        k = int(rng.integers(4))
        t = 0
        while t < steps:
            length = _duration(rng, REGIMES[k])
            end = min(steps, t + length)
            x[b, t:end, 0] = MEANS[k] + rng.normal(0, sigma, end - t)
            labels[b, t:end] = k
            t = end
            k = int(rng.choice(4, p=BASE_TRANSITION[k]))
    return x, labels


def check_recursion(model, x, labels, sigma=SIGMA):
    pmf = model_law(model).detach().numpy()
    with torch.inference_mode():
        actual = float(model.log_likelihood(torch.tensor(x), reduce=True))
        trace = filter_model_sequence(model, torch.tensor(x))
        probability = trace.state_posterior.numpy()
        trans = model.dist.transition.log_matrix(T=1).exp().numpy()[0, 0, :, 0, :]
    np.testing.assert_allclose(np.diag(trans), 0, rtol=0, atol=0)
    np.testing.assert_allclose(trans, BASE_TRANSITION, rtol=0, atol=1e-14)
    references = []
    max_error = 0.0
    for b in range(len(x)):
        emission = -0.5 * (
            math.log(2 * math.pi * sigma**2)
            + (x[b, :, 0, None].astype(float) - MEANS) ** 2 / sigma**2
        )
        ll, posterior = segment_reference(emission, pmf, trans, np.full(4, 0.25))
        references.append(ll)
        max_error = max(max_error, float(np.max(np.abs(np.exp(posterior) - probability[b]))))
    error = abs(actual - sum(references))
    if error > 1e-7 or max_error > 1e-8:
        raise AssertionError(f"HSMM recursion mismatch LL={error} probability={max_error}")
    return {
        "public_ll": actual,
        "independent_ll": sum(references),
        "ll_error": error,
        "maximum_state_probability_error": max_error,
        "accuracy": float(np.mean(probability.argmax(-1) == labels)),
        "uncertain_rows_max_probability_below_0_99": int(np.sum(probability.max(-1) < 0.99)),
        "rows": int(labels.size),
    }


def run(seed):
    pmf = true_law()
    complete, censored = censored_counts(seed + 100000)
    oracle, history = labelled_em(complete, censored, pmf > 0)
    model, optimization = fit_duration(pmf, complete, censored)
    fitted = model_law(model).detach().numpy()
    records = float(complete.sum() + censored.sum())
    gap = (count_ll(oracle, complete, censored) - count_ll(fitted, complete, censored)) / records
    tv = np.abs(fitted - pmf).sum(1) / 2
    if gap > 0.01 or max(tv) > 0.12:
        raise AssertionError(f"duration gate failed: objective gap={gap} TV={tv}")
    x, labels = sequences(seed + 200000)
    known = check_recursion(make_model(pmf), x, labels)
    learned = check_recursion(model, x, labels)
    ambiguous_x, ambiguous_labels = sequences(seed + 300000, count=2, steps=300, sigma=2.0)
    ambiguous = check_recursion(
        make_model(pmf, sigma=2.0), ambiguous_x, ambiguous_labels, sigma=2.0
    )
    if ambiguous["uncertain_rows_max_probability_below_0_99"] == 0:
        raise AssertionError("overlapping-emission control has no uncertain rows")
    if known["accuracy"] < 0.99 or learned["accuracy"] < 0.99:
        raise AssertionError("known-emission state recovery gate failed")
    print(f"seed={seed} complete", flush=True)
    return {
        "seed": seed,
        "data_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
        "complete_counts_by_state": complete.sum(1).tolist(),
        "censored_counts_by_state": censored.sum(1).tolist(),
        "train_count_sha256": hashlib.sha256(np.stack([complete, censored]).tobytes()).hexdigest(),
        "actual_duration_updates": 1200,
        "duration_tv_states": tv.tolist(),
        "objective_gap_to_em_nats_per_record": gap,
        "em_initial_ll": history[0],
        "em_final_ll": history[-1],
        "adam_final_ll": count_ll(fitted, complete, censored),
        "optimizer_checkpoints": optimization,
        "true_mean": (pmf @ np.arange(1, D + 1)).tolist(),
        "learned_mean": (fitted @ np.arange(1, D + 1)).tolist(),
        "fitted_pmf": fitted.tolist(),
        "known_law_recursion": known,
        "learned_law_recursion": learned,
        "overlapping_emission_recursion": ambiguous,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    paths = [
        "scripts/verify_isolated_trading_hsmm.py",
        "scripts/diagnose_trading_ground_truth.py",
        "scripts/generate_trading_regime_dataset.py",
        "nhsmm/models/base.py",
        "nhsmm/filtering.py",
        "nhsmm/distributions/default.py",
    ]
    payload = {
        "schema": "nhsmm-isolated-trading-hsmm-v1",
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
            "duration": "exact Trading clipped PMFs, D260 finite with known support mask; no tail",
            "transitions": "known stationary BASE_TRANSITION, exact zero diagonal, same at every age",
            "observations": "known IID Gaussian, one feature, means -6/-2/2/6, sigma0.35; fresh episode starts, uniform initial state",
            "duration_training": "8192 labelled administrative-censor records/state; cutoff uniform integer80..300; end if L<cutoff else survival P(L>=cutoff); no initial-age ambiguity",
            "optimization": "only production Duration logits, script Adam lr0.05/1200 steps; independent labelled EM200; all nuisance parameters frozen",
            "gates": "objective gap <=0.01 nats/record; each PMF TV<=0.12; state recovery>=0.99; independent segment LL error<=1e-7, state probability error<=1e-8",
            "limits": "oracle labelled duration-stage training and known support/nuisance parameters; tests production duration module and public marginal likelihood/filter but not end-to-end unlabeled optimize, macro transition learning, Trading features, runtime/artifact masks or live utility",
        },
        "runs": [run(seed) for seed in (901, 902, 903)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            [
                {
                    k: r[k]
                    for k in (
                        "seed",
                        "duration_tv_states",
                        "objective_gap_to_em_nats_per_record",
                        "known_law_recursion",
                        "learned_law_recursion",
                    )
                }
                for r in payload["runs"]
            ]
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
