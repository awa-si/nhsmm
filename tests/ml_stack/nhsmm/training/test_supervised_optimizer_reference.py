"""Ragged supervised Adam updates against raw-weight probability products."""

import pytest
import torch
from nhsmm import CausalContext

from tests.ml_stack.nhsmm.reference.neural_parameter_reference import enumerate_weights, path_weight
from tests.ml_stack.nhsmm.reference.test_neural_parameter_math import fixture, parts


def batch(x, c):
    lengths = [4, 0, 1, 3]
    xs = [x[:n].detach().clone() for n in lengths]
    labels = [torch.tensor([0, 0, 1, 2][:n], dtype=torch.long) for n in lengths]
    contexts = [
        CausalContext(c.emission[:n].detach(), c.boundary[:n].detach(), c.available[:n])
        for n in lengths
    ]
    combined = CausalContext(
        [v.emission for v in contexts],
        [v.boundary for v in contexts],
        [v.available for v in contexts],
    )
    return xs, labels, contexts, combined


def reference_score(model, xs, labels, contexts, filtering):
    total = model.duration_logits_bias.sum() * 0
    hmm = model.config.max_duration == 1 and not model.config.duration_tail
    for x, z, c in zip(xs, labels, contexts, strict=True):
        if not len(z):
            continue
        raw = parts(model, x, c)
        if not filtering:
            total = total + path_weight(z.tolist(), *raw, hmm=hmm)[0].log()
            continue
        for n, label in enumerate(z.tolist(), 1):
            paths = enumerate_weights(raw, hmm=hmm, length=n)
            denominator = torch.stack([w for _, w, _ in paths]).sum()
            numerator = torch.stack([w for p, w, _ in paths if p[-1] == label]).sum()
            total = total + (numerator / denominator).log()
    return total


@pytest.mark.parametrize("duration,tail", [(1, False), (2, False), (1, True), (2, True)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("context_only", [False, True])
@pytest.mark.parametrize("filtering", [False, True])
def test_ragged_supervised_adam_matches_probability_oracle(
    duration, tail, dtype, context_only, filtering
):
    actual, x, c = fixture(duration, tail, first_available=False)
    expected, _, _ = fixture(duration, tail, first_available=False)
    actual.to(dtype=dtype)
    expected.to(dtype=dtype)
    x = x.to(dtype)
    c = CausalContext(c.emission.to(dtype), c.boundary.to(dtype), c.available)
    xs, labels, contexts, combined = batch(x, c)
    # A caller-frozen base weight must remain frozen in either fitting mode.
    for model in (actual, expected):
        model.dist.emission.mu.requires_grad_(False)
    flags = {n: p.requires_grad for n, p in actual.named_parameters()}
    original = {n: p.detach().clone() for n, p in actual.named_parameters()}
    for name, p in expected.named_parameters():
        if context_only:
            p.requires_grad_(
                flags[name]
                and (".context_net." in name or "._proj." in name or name.endswith(".delta_scale"))
            )
    parameters = [p for p in expected.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        parameters,
        lr=0.005,
        betas=(expected.config.adam_beta1, expected.config.adam_beta2),
        eps=expected.config.adam_eps,
    )
    rows = sum(map(len, labels))
    initial = float(reference_score(expected, xs, labels, contexts, filtering).detach())
    # Non-default clipping exercises normalization, clipping and optimizer wiring.
    actual.config.supervised_grad_clip_norm = 0.07
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = -reference_score(expected, xs, labels, contexts, filtering) / rows
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 0.07, error_if_nonfinite=True)
        optimizer.step()
        if tail and not context_only:
            h = expected.dist.duration.tail_end_probability.detach()
            # These updates stay interior, so projection must be an identity.
            assert bool(((h > 0.1) & (h < 0.9)).all())
    objective = "causal_filter_log_score" if filtering else "observed_joint_causal_prefix"
    summary = actual.fit_supervised(
        xs,
        labels,
        context=combined,
        steps=3,
        lr=0.005,
        context_only=context_only,
        objective=objective,
    )
    atol, rtol = (3e-6, 3e-5) if dtype == torch.float32 else (2e-11, 2e-10)
    for name, p in actual.named_parameters():
        torch.testing.assert_close(p, dict(expected.named_parameters())[name], atol=atol, rtol=rtol)
        assert p.requires_grad == flags[name]
        trainable = dict(expected.named_parameters())[name].requires_grad
        if not trainable:
            assert torch.equal(p, original[name])
    assert not actual.training
    assert summary["rows"] == rows == 8 and summary["updates"] == 3
    assert summary["initial_nll_per_row"] == pytest.approx(-initial / rows, abs=atol, rel=rtol)
    final = float(reference_score(expected, xs, labels, contexts, filtering).detach())
    assert summary["final_nll_per_row"] == pytest.approx(-final / rows, abs=atol, rel=rtol)
