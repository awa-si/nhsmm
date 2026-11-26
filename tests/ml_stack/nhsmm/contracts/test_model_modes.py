"""Explicit static/neural mode selection, independent scores and usable fitting."""

from copy import deepcopy

import pytest
import torch
from nhsmm import (
    BASE_CONTEXT,
    NHSMM,
    CausalContext,
    HSMMFilterRuntime,
    ModelConfig,
    build_artifact,
    load_artifact,
    save_artifact,
)
from nhsmm.filtering import filter_model_sequence

from tests.ml_stack.nhsmm.reference.neural_parameter_reference import enumerate_weights, path_weight
from tests.ml_stack.nhsmm.reference.test_neural_parameter_math import parts


def make(mode, *, tail=False, dtype=torch.float64, **options):
    settings = dict(n_states=2, n_features=1, dropout=0.0, seed=7)
    if mode != "hmm":
        settings.update(max_duration=4, duration_tail=tail)
    settings.update(options)
    config = ModelConfig.for_mode(mode, **settings)
    model = NHSMM(config, device="cpu")
    model.initialize_distributions()
    return model.to(dtype=dtype).eval()


@pytest.mark.parametrize("mode", ["hmm", "hsmm"])
def test_static_modes_do_not_construct_or_train_neural_networks(monkeypatch, mode):
    import nhsmm.models.base as base

    def forbidden(*args, **kwargs):
        raise AssertionError("static model constructed an observation encoder")

    monkeypatch.setattr(base, "DefaultEncoder", forbidden)
    model = make(mode)
    assert model.config.model_mode == mode
    assert model.encoder is None and model.context_dim is None
    assert not any(
        isinstance(m, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.LSTM)) for m in model.modules()
    )
    for name in ("initial", "duration", "transition", "emission"):
        component = getattr(model.dist, name)
        assert component.context_net is None and component._proj is None
        assert not component.delta_scale.requires_grad
    x = torch.zeros(4, 1, dtype=torch.float64)
    z = torch.tensor([0, 0, 1, 1])
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    with pytest.raises(ValueError, match="active external context"):
        model.fit_supervised(x, z, context_only=True, steps=1)
    for name, p in model.named_parameters():
        assert torch.equal(p, before[name])


@pytest.mark.parametrize("mode,tail", [("hmm", False), ("hsmm", False), ("hsmm", True)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_static_mode_scores_and_posteriors_match_raw_path_oracle(mode, tail, dtype):
    model = make(mode, tail=tail, dtype=dtype)
    x = torch.tensor([[-0.4], [0.6], [-0.8], [0.2], [0.9]], dtype=dtype)
    context = CausalContext(
        torch.zeros(5, 1, dtype=dtype),
        torch.zeros(5, 1, dtype=dtype),
        torch.zeros(5, dtype=torch.bool),
    )
    raw = parts(model, x, context)
    runtime = HSMMFilterRuntime(model)
    atol = 2e-5 if dtype == torch.float32 else 2e-12
    for n, row in enumerate(x, 1):
        paths = enumerate_weights(raw, hmm=mode == "hmm", length=n)
        denominator = torch.stack([w for _, w, _ in paths]).sum()
        expected = torch.stack(
            [
                torch.stack([w for p, w, _ in paths if p[-1] == state]).sum() / denominator
                for state in range(2)
            ]
        )
        actual = runtime.step(row)
        torch.testing.assert_close(actual.state_posterior[0], expected, rtol=0, atol=atol)
    torch.testing.assert_close(model.log_likelihood(x)[0], denominator.log(), rtol=0, atol=atol)
    labels = torch.tensor([0, 0, 1, 1, 0])
    joint = path_weight(labels.tolist(), *raw, hmm=mode == "hmm")[0].log()
    torch.testing.assert_close(
        model.supervised_log_likelihood(x, labels)[0], joint, rtol=0, atol=atol
    )


@pytest.mark.parametrize("mode", ["hmm", "hsmm"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_supervised_static_modes_train_save_load_and_stream(tmp_path, mode, dtype):
    model = make(mode, dtype=dtype)
    labels = torch.tensor([0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0])
    x = (labels.to(dtype) * 4 - 2).unsqueeze(-1)
    x += torch.linspace(-0.05, 0.05, len(x), dtype=dtype).unsqueeze(-1)
    summary = model.fit_supervised(x, labels, steps=25, lr=0.03)
    assert summary["updates"] == 25 and summary["rows"] == len(x)
    assert summary["final_nll_per_row"] < summary["initial_nll_per_row"] - 0.01
    trace = filter_model_sequence(model, x)
    predicted = trace.state_posterior[0].argmax(-1)
    assert float((predicted == labels).to(torch.float64).mean()) >= 0.9
    assert set(predicted.tolist()) == {0, 1}
    path = tmp_path / "static.pt"
    save_artifact(model, path)
    assert build_artifact(model)["artifact_version"] == 3
    assert "model_mode" not in model.config.to_dict()
    loaded = load_artifact(path, require_causal=True)
    assert loaded.encoder is None and loaded.config.model_mode == mode
    assert loaded.config.to_dict() == model.config.to_dict()
    runtime = HSMMFilterRuntime(loaded)
    atol = 2e-5 if dtype == torch.float32 else 2e-12
    for n, row in enumerate(x):
        state = runtime.step(row, timestamp=n + 1)
        torch.testing.assert_close(
            state.state_posterior[0], trace.state_posterior[0, n], rtol=0, atol=atol
        )
    continued = load_artifact(path, require_causal=True, freeze=False)
    resumed = continued.fit_supervised(x, labels, steps=2, lr=0.005)
    assert resumed["final_nll_per_row"] < resumed["initial_nll_per_row"]
    assert continued.config.model_mode == mode


@pytest.mark.parametrize("external", [False, True])
def test_neural_mode_constructs_only_the_selected_context_path(tmp_path, external):
    model = make("nhsmm", **({"context_dim": 2} if external else {}))
    assert model.config.model_mode == "nhsmm"
    assert (model.encoder is None) is external
    assert all(
        getattr(model.dist, name).context_net is not None
        for name in ("initial", "duration", "transition", "emission")
    )
    path = tmp_path / "neural.pt"
    save_artifact(model, path)
    loaded = load_artifact(path, require_causal=True)
    assert loaded.config.model_mode == "nhsmm"
    assert (loaded.encoder is None) is external


def test_external_neural_mode_falls_back_exactly_when_context_unavailable():
    model = make("nhsmm", context_dim=2)
    x = torch.tensor([[-0.4], [0.6], [-0.8], [0.2]], dtype=torch.float64)
    e = torch.tensor([[0.2, -0.3], [-0.4, 0.7], [0.5, 0.1], [0.8, -0.2]], dtype=x.dtype)
    absent = CausalContext(e, -e, torch.zeros(4, dtype=torch.bool))
    torch.testing.assert_close(
        model.log_likelihood(x, context=absent),
        model.log_likelihood(x, context=BASE_CONTEXT),
        rtol=0,
        atol=0,
    )
    present = CausalContext(e, -e, torch.tensor([False, True, True, True]))
    assert not torch.allclose(
        model.log_likelihood(x, context=present),
        model.log_likelihood(x, context=BASE_CONTEXT),
        rtol=0,
        atol=1e-8,
    )


@pytest.mark.parametrize("mode", ["HMM", "", None, True, 12])
def test_unknown_modes_rejected(mode):
    with pytest.raises(ValueError, match="mode"):
        ModelConfig.for_mode(mode, n_states=2, n_features=1)


@pytest.mark.parametrize(
    "mode,settings",
    [
        ("hmm", {"max_duration": 3}),
        ("hmm", {"duration_tail": True}),
        ("hmm", {"allow_self_transitions": False}),
        ("hmm", {"context_dim": 2}),
        ("hmm", {"hidden_dim": 2}),
        ("hmm", {"use_context_encoder": True}),
        ("hmm", {"causal": False}),
        ("hsmm", {"max_duration": 1}),
        ("hsmm", {"allow_self_transitions": True}),
        ("hsmm", {"context_dim": 2}),
        ("hsmm", {"hidden_dim": 2}),
        ("hsmm", {"use_context_encoder": True}),
        ("hsmm", {"causal": False}),
        ("nhsmm", {"max_duration": 1}),
        ("nhsmm", {"allow_self_transitions": True}),
        ("nhsmm", {"use_context_encoder": False}),
        ("nhsmm", {"causal": False}),
    ],
)
def test_conflicting_mode_options_fail_without_silent_overrides(mode, settings):
    with pytest.raises(ValueError):
        ModelConfig.for_mode(mode, n_states=2, n_features=1, **settings)


def test_legacy_configuration_payload_is_unchanged_and_mode_is_resolved():
    payload = ModelConfig(n_states=2, n_features=1).to_dict()
    assert "model_mode" not in payload and "mode" not in payload
    cfg = ModelConfig.from_dict(payload)
    assert cfg.to_dict() == payload
    assert cfg.model_mode == "hsmm"
    assert cfg.with_overrides(max_duration=1).model_mode == "hmm"
    assert cfg.with_overrides(context_dim=2).model_mode == "nhsmm"


@pytest.mark.parametrize("mode,tail", [("hmm", False), ("hsmm", False), ("hsmm", True)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_static_gaussian_location_translation_preserves_causal_inference(mode, tail, dtype):
    """A common observation/emission translation preserves each likelihood factor."""
    model = make(mode, tail=tail, dtype=dtype)
    with torch.no_grad():
        model.dist.emission.mu.copy_(torch.tensor([[-1.0], [1.0]], dtype=dtype))
        model.dist.emission.log_var.fill_(0.5)
    translated = deepcopy(model)
    with torch.no_grad():
        translated.dist.emission.mu.add_(1.0)

    observations = torch.tensor([[-0.5], [0.25], [0.75], [-0.25], [0.5], [-0.75]], dtype=dtype)
    shifted = observations + 1.0
    labels = torch.tensor([0, 1, 1, 0, 1, 0])
    tolerance = 2e-6 if dtype == torch.float32 else 2e-12

    # Gaussian p(x | z; mu, sigma) = p(x+c | z; mu+c, sigma), independently
    # of the state/age law. Test both the normalized and absolute score paths.
    left = filter_model_sequence(model, observations)
    right = filter_model_sequence(translated, shifted)
    torch.testing.assert_close(left.log_posterior, right.log_posterior, atol=tolerance, rtol=0)
    torch.testing.assert_close(
        model.log_likelihood(observations),
        translated.log_likelihood(shifted),
        atol=tolerance,
        rtol=0,
    )
    torch.testing.assert_close(
        model.supervised_filter_log_score(observations, labels),
        translated.supervised_filter_log_score(shifted, labels),
        atol=tolerance,
        rtol=0,
    )

    original_stream = HSMMFilterRuntime(model)
    shifted_stream = HSMMFilterRuntime(translated)
    for t, (row, shifted_row) in enumerate(zip(observations, shifted, strict=True)):
        original = original_stream.step(row, timestamp=t)
        moved = shifted_stream.step(shifted_row, timestamp=t)
        torch.testing.assert_close(
            original.log_posterior, moved.log_posterior, atol=tolerance, rtol=0
        )
        a = original_stream.forecast_transition()
        b = shifted_stream.forecast_transition()
        torch.testing.assert_close(a.next_state_prior, b.next_state_prior, atol=tolerance, rtol=0)
        torch.testing.assert_close(
            a.episode_end_probability,
            b.episode_end_probability,
            atol=tolerance,
            rtol=0,
        )

    # An unadjusted shifted observation is not the same model experiment.
    unadjusted = filter_model_sequence(model, shifted)
    assert torch.max(torch.abs(left.state_posterior - unadjusted.state_posterior)) > 0.01
