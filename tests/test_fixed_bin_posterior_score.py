import numpy as np
import pytest
import torch

from scripts.block_mass_smoothing import count_objective, geometry
from scripts.compare_unlabelled_duration_em import aggregate, emission_scores
from scripts.fixed_bin_posterior_score import density, score_gradient
from scripts.verify_isolated_trading_hsmm import make_model, sequences


@pytest.mark.parametrize("strength", [0.0, 3.0])
def test_count_objective_analytic_gradient_with_short_last_bin(strength):
    torch.manual_seed(42)
    z = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    counts = np.random.default_rng(42).uniform(0, 20, (4, 14))
    blocks, lengths, r = geometry(14, 4)
    masses = np.add.reduceat(counts + 1e-6, [a for a, _ in blocks], axis=1)
    full = torch.cat((z, z.new_zeros((4, 1))), 1)
    obj = count_objective(
        full,
        torch.tensor(masses),
        torch.tensor(lengths, dtype=torch.float64),
        torch.tensor(r),
        strength,
    ).sum()
    reference = torch.autograd.grad(obj, z)[0]
    actual = score_gradient(z, counts, size=14, width=4, strength=strength)
    torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)
    assert density(z, 14, 4).shape == (4, 14)


def test_censored_public_gradient_and_adam_step_match_analytic_score():
    torch.set_num_threads(1)
    torch.manual_seed(1920)
    x, _ = sequences(401920, count=3, steps=60, sigma=2.0)
    z = torch.nn.Parameter(torch.randn(4, 32, dtype=torch.float64) * 0.1)
    analytic_z = torch.nn.Parameter(z.detach().clone())
    p = density(z)
    model = make_model(p.detach().numpy(), sigma=2.0)
    # Connect bin logits to the unchanged public likelihood graph, with all
    # nuisance parameters frozen and all configured ages permitted.
    del model.dist.duration.logits
    model.dist.duration.logits = p.log()
    public = model.log_likelihood(torch.tensor(x), reduce=True)
    objective = public + 1e-6 * p.log().sum()
    (-objective).backward()
    _, counts = aggregate(emission_scores(x), p.detach().numpy())
    analytic_z.grad = -score_gradient(analytic_z, counts)
    torch.testing.assert_close(z.grad, analytic_z.grad, atol=1e-7, rtol=0)
    torch.nn.utils.clip_grad_norm_([z], 5.0)
    torch.nn.utils.clip_grad_norm_([analytic_z], 5.0)
    torch.optim.Adam([z], lr=0.01).step()
    torch.optim.Adam([analytic_z], lr=0.01).step()
    torch.testing.assert_close(z, analytic_z, atol=1e-10, rtol=0)


@pytest.mark.parametrize("kind", ["negative", "nonfinite", "shape"])
def test_invalid_counts_rejected(kind):
    z = torch.zeros(4, 32, dtype=torch.float64)
    counts = np.ones((4, 260))
    if kind == "negative":
        counts[0, 0] = -1
    elif kind == "nonfinite":
        counts[0, 0] = np.nan
    else:
        counts = counts[:, :-1]
    with pytest.raises(ValueError):
        score_gradient(z, counts)
