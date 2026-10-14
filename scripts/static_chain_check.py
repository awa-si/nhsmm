"""Public all-static parameter score and independent segment/filter checks."""

import numpy as np
import torch

from nhsmm.filtering import filter_model_sequence
from scripts.fixed_bin_posterior_score import density, score_gradient
from scripts.joint_gaussian_duration import emission_gradients, scores
from scripts.joint_static_chain import MASK, chain_gradients, chunk_statistics, statistics
from scripts.verify_isolated_trading_hsmm import make_model, segment_reference


def numerical_check(x, p, means, variances, transition, initial):
    torch.set_num_threads(1)
    x = np.asarray(x[:2])
    raw = variances + np.log(-np.expm1(-variances))
    model = make_model(p, sigma=2.0)
    with torch.no_grad():
        model.dist.emission.mu.copy_(torch.tensor(means[:, None]))
        model.dist.emission.log_var.copy_(torch.tensor(raw[:, None]))
    model.dist.emission.mu.requires_grad_(True)
    model.dist.emission.log_var.requires_grad_(True)
    z = torch.nn.Parameter(torch.tensor(np.log(p[:, ::8][:, :-1] / p[:, -1, None])))
    tz = torch.nn.Parameter(torch.tensor(np.log(np.where(MASK, transition, 1))))
    iz = torch.nn.Parameter(torch.tensor(np.log(initial)))
    pmf = density(z)
    tm = tz.masked_fill(~torch.tensor(MASK), -torch.inf).softmax(1)
    ip = iz.softmax(0)
    del model.dist.duration.logits
    model.dist.duration.logits = pmf.log()
    del model.dist.transition.logits
    model.dist.transition.logits = tm.clamp_min(1e-300).log()[:, None, :].expand(4, 260, 4)
    del model.dist.initial.logits
    model.dist.initial.logits = ip.log()
    ll = model.log_likelihood(torch.tensor(x), reduce=True)
    (ll + 1e-6 * (pmf.log().sum() + tm[torch.tensor(MASK)].log().sum() + ip.log().sum())).backward()
    s = statistics(x, p, means, variances, transition, initial)
    gm, gv = emission_gradients(s, means, variances)
    gt, gi = chain_gradients(s, transition, initial)
    references = [
        segment_reference(row, p, transition, initial) for row in scores(x, means, variances)
    ]
    with torch.no_grad():
        trace = filter_model_sequence(model, torch.tensor(x))
    forward = chunk_statistics(x, p, means, variances, transition, initial, return_filter=True)
    return {
        "public_ll_error": abs(float(ll.detach()) - s["ll"]),
        "segment_ll_error": abs(sum(row[0] for row in references) - s["ll"]),
        "duration_gradient_error": float(
            torch.max(torch.abs(z.grad - score_gradient(z, s["duration_counts"])))
        ),
        "mean_gradient_error": float(
            np.max(np.abs(model.dist.emission.mu.grad.numpy()[:, 0] - gm))
        ),
        "raw_variance_gradient_error": float(
            np.max(np.abs(model.dist.emission.log_var.grad.numpy()[:, 0] - gv / (1 + np.exp(-raw))))
        ),
        "transition_gradient_error": float(np.max(np.abs(tz.grad.numpy() - gt))),
        "initial_gradient_error": float(np.max(np.abs(iz.grad.numpy() - gi))),
        "public_filter_error": float(
            np.max(np.abs(trace.state_posterior.numpy() - forward["filtered_state_probabilities"]))
        ),
        "segment_filter_error": float(
            np.max(
                np.abs(
                    trace.state_posterior.numpy() - np.stack([np.exp(row[1]) for row in references])
                )
            )
        ),
    }
