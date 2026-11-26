"""Trained HSMM artifacts preserve filtering, forecasts and fresh-Adam continuation."""

from dataclasses import fields

import pytest
import torch
from nhsmm import NHSMM, CausalContext, HSMMFilterRuntime, ModelConfig
from nhsmm.artifact import build_artifact, load_artifact, save_artifact
from nhsmm.filtering import filter_model_sequence

from tests.ml_stack.nhsmm.reference.neural_parameter_reference import enumerate_weights
from tests.ml_stack.nhsmm.reference.test_neural_parameter_math import fixture, parts


def trained_model(duration, tail, dtype, internal, objective):
    source, x, context = fixture(duration, tail, first_available=False)
    if internal:
        config = source.config.to_dict()
        config.update(use_context_encoder=True, context_dim=4, hidden_dim=4, cnn_channels=3)
        source = NHSMM(ModelConfig.from_dict(config), device="cpu")
        source.initialize_distributions()
        source.eval()
        context = None
    source.to(dtype=dtype)
    x = x.to(dtype)
    if context is not None:
        context = CausalContext(
            context.emission.detach().to(dtype),
            context.boundary.detach().to(dtype),
            context.available,
        )
    labels = torch.tensor([0, 0, 1, 2])
    before = {name: value.clone() for name, value in source.state_dict().items()}
    train_context = (
        None
        if context is None
        else CausalContext(
            context.emission.repeat(2, 1),
            context.boundary.repeat(2, 1),
            context.available.repeat(2),
        )
    )
    # The first D+ episode passes the finite cutoff and ends in the tail.
    # Thus the artifact contains an actually learned tail hazard.
    train_labels = torch.tensor([0, 0, 0, 0, 0, 1, 2, 2] if tail else [0, 0, 1, 2, 2, 0, 1, 1])
    source.fit_supervised(
        x.repeat(2, 1),
        train_labels,
        context=train_context,
        steps=4,
        lr=0.003,
        objective=objective,
    )
    if tail:
        assert not torch.equal(
            source.dist.duration.tail_end_probability,
            before["dist.duration.tail_end_probability"],
        )
    assert any(not torch.equal(v, before[n]) for n, v in source.state_dict().items())
    return source, x, labels, context


def assert_same_state(left, right):
    assert left.config == right.config
    assert left.state_dict().keys() == right.state_dict().keys()
    for name, value in left.state_dict().items():
        other = right.state_dict()[name]
        assert value.dtype == other.dtype and value.device == other.device
        assert torch.equal(value, other), name


def reference_filter(model, x, context):
    """Enumerate all true-age paths from raw trained parameters."""
    raw = parts(model, x, context)
    hmm = model.config.max_duration == 1 and not model.config.duration_tail
    result = []
    for n in range(1, len(x) + 1):
        rows = enumerate_weights(raw, hmm=hmm, length=n)
        total = torch.stack([weight for _, weight, _ in rows]).sum()
        mass = x.new_zeros(model.config.n_states, model.config.max_duration)
        for path, weight, age in rows:
            if age <= model.config.max_duration or model.config.duration_tail:
                mass[path[-1], min(age, model.config.max_duration) - 1] += weight
        result.append(mass / total)
    return torch.stack(result)


@pytest.mark.parametrize("duration,tail", [(1, False), (3, False), (3, True)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("internal", [False, True])
@pytest.mark.parametrize("objective", ["observed_joint_causal_prefix", "causal_filter_log_score"])
def test_trained_artifact_roundtrip_and_supervised_continuation(
    duration, tail, dtype, internal, objective, tmp_path
):
    source, x, labels, context = trained_model(duration, tail, dtype, internal, objective)
    snapshot = build_artifact(source)
    snapshot_values = {n: v.clone() for n, v in snapshot["state_dict"].items()}
    path = save_artifact(source, tmp_path / "trained.pt")
    frozen = load_artifact(path, device="cpu", require_causal=True)
    trainable = load_artifact(path, device="cpu", require_causal=True, freeze=False)
    assert not frozen.training and not trainable.training
    assert all(not p.requires_grad for p in frozen.parameters())
    assert {n: p.requires_grad for n, p in trainable.named_parameters()} == {
        n: p.requires_grad for n, p in source.named_parameters()
    }
    assert_same_state(source, frozen)
    assert_same_state(source, trainable)
    tolerance = 3e-6 if dtype == torch.float32 else 3e-12
    with torch.no_grad():
        expected = filter_model_sequence(source, x, context=context).log_posterior
        actual = filter_model_sequence(frozen, x, context=context).log_posterior
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        torch.testing.assert_close(
            frozen.log_likelihood(x, context=context),
            source.log_likelihood(x, context=context),
            atol=tolerance,
            rtol=tolerance,
        )
        for score in ("supervised_log_likelihood", "supervised_filter_log_score"):
            torch.testing.assert_close(
                getattr(frozen, score)(x, labels, context=context),
                getattr(source, score)(x, labels, context=context),
                atol=tolerance,
                rtol=tolerance,
            )
        if context is not None:
            torch.testing.assert_close(
                actual[0].exp(),
                reference_filter(frozen, x, context),
                atol=tolerance,
                rtol=tolerance,
            )
    runtimes = [HSMMFilterRuntime(source), HSMMFilterRuntime(frozen)]
    for t, row in enumerate(x):
        one_context = (
            None
            if context is None
            else CausalContext(context.emission[t], context.boundary[t], context.available[t])
        )
        states = [runtime.step(row, context=one_context, timestamp=t) for runtime in runtimes]
        for state in states:
            torch.testing.assert_close(
                state.log_posterior, expected[:, t], atol=tolerance, rtol=tolerance
            )
        for method, args in (("forecast_survival", ([1, 3, 12],)), ("forecast_transition", ())):
            left, right = [getattr(runtime, method)(*args) for runtime in runtimes]
            for field in fields(left):
                torch.testing.assert_close(
                    getattr(left, field.name),
                    getattr(right, field.name),
                    atol=tolerance,
                    rtol=tolerance,
                )
    # Inference artifacts contain weights/config, not optimizer moments. Both
    # continuations intentionally start fresh Adam instances from equal weights.
    left_summary = source.fit_supervised(
        x, labels, context=context, steps=3, lr=0.002, objective=objective
    )
    right_summary = trainable.fit_supervised(
        x, labels, context=context, steps=3, lr=0.002, objective=objective
    )
    assert_same_state(source, trainable)
    assert left_summary == right_summary
    assert any(not torch.equal(v, snapshot_values[n]) for n, v in source.state_dict().items())
    for name, value in snapshot["state_dict"].items():
        assert torch.equal(value, snapshot_values[name])
        assert torch.equal(frozen.state_dict()[name], snapshot_values[name])
    # Streaming reset starts a fresh episode under the loaded artifact.
    runtimes[1].reset()
    first_context = (
        None
        if context is None
        else CausalContext(context.emission[0], context.boundary[0], context.available[0])
    )
    reset_state = runtimes[1].step(x[0], context=first_context, timestamp=0)
    torch.testing.assert_close(
        reset_state.log_posterior, actual[:, 0], atol=tolerance, rtol=tolerance
    )
