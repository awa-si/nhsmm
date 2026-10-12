"""Coverage gates, hazard isolation and unbounded run support controls."""

import numpy as np
import pytest
import torch

from scripts.diagnose_duration_resets import fit
from scripts.diagnose_model_conform_duration import D, PMF, generate, law
from scripts.diagnose_tail_coverage import require_coverage, run_lengths, tail_exposure, TRANSITION


def test_tail_gate_counts_censored_exposure_without_end_and_fails_small_data():
    ledger = [
        [
            {"state": 0, "observed": D + 2, "censored": False},
            {"state": 0, "observed": D + 3, "censored": True},
        ]
    ]
    row = tail_exposure(ledger)[0]
    assert row["tail_trials"] == 6
    assert row["completed_tail_episodes"] == 1
    assert row["hazard_mle"] == pytest.approx(1 / 6)
    assert tail_exposure(ledger)[1]["hazard_mle"] is None
    with pytest.raises(ValueError, match="exposure gate"):
        require_coverage(ledger)


def test_hazard_only_fit_preserves_true_pmf_and_initialization():
    x, _, _ = generate(41, tail=True, count=2, steps=24)
    model = fit(42, True, x, TRANSITION, "known_transition", 2, known_duration_pmf=True)
    p, h = law(model)
    np.testing.assert_allclose(p, PMF, atol=1e-14)
    assert not model.dist.duration.logits.requires_grad
    assert model.dist.duration.tail_end_probability.requires_grad
    assert model._diagnostic_completed_iterations == 2
    assert np.max(np.abs(h - 0.5)) > 1e-6


def test_D_plus_supports_same_state_run_above_finite_D_without_reset():
    x, _, _ = generate(41, tail=False, count=1, steps=12)
    finite = fit(42, False, x, TRANSITION, "no_resets", 1)
    extended = fit(42, True, x, TRANSITION, "no_resets", 1)
    p, _ = law(finite)
    q, h = law(extended)
    assert p[0, -1] > 0
    assert q[0, -1] * (1 - h[0]) > 0  # P(L >= D+1), unlike finite support.
    with torch.no_grad():
        trans = extended.dist.transition.log_matrix(T=1).exp().reshape(2, D, 2)
        assert torch.equal(trans[0, :, 0], torch.zeros(D, dtype=trans.dtype))
    assert run_lengths(np.zeros((1, D + 1), dtype=int)) == {
        "max_observed_run": D + 1,
        "observed_runs_above_D": 1,
    }


@pytest.mark.parametrize("seed", [901, 902, 903])
def test_fixed_preregistered_tail_dataset_meets_exposure_gate(seed):
    _, _, ledger = generate(seed + 20000, tail=True, count=128, transition=TRANSITION)
    rows = require_coverage(ledger)
    assert min(r["completed_tail_episodes"] for r in rows) >= 30
    assert min(r["tail_trials"] for r in rows) >= 60
