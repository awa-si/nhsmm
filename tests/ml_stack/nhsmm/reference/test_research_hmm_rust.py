"""Strict native HMM first-order parity; requires an explicit local Rust library."""

import copy
import os
from pathlib import Path

import pytest
import torch
from nhsmm import NHSMM, ModelConfig


@pytest.fixture
def rust_library():
    value = os.environ.get("NHSMM_RESEARCH_HMM_RUST_LIB")
    if not value or not Path(value).is_file():
        pytest.skip("set NHSMM_RESEARCH_HMM_RUST_LIB to the compiled research cdylib")
    return value


@pytest.mark.parametrize("length", [1, 19])
def test_rust_optin_matches_torch_likelihood_gradients_and_adam(rust_library, length):
    torch.set_num_threads(1)
    config = ModelConfig.for_mode(
        "hmm",
        n_states=3,
        n_features=1,
        seed=7,
        max_iter=4,
        n_init=1,
        verbose=False,
        convergence_stop=False,
        use_scheduler=False,
        optimizer_temperature_mode="constant",
        optimizer_initialization="current",
        loss_bias=0.0,
    )
    x = torch.linspace(-1, 1, length, dtype=torch.float64).unsqueeze(-1)
    original = NHSMM(config, device="cpu").to(dtype=torch.float64)
    original.initialize_distributions(jitter=0)
    torch_model, rust_model = copy.deepcopy(original), copy.deepcopy(original)
    rust_model.set_backend("rust", library_path=rust_library)
    assert rust_model.config.to_dict() == torch_model.config.to_dict()
    assert rust_model.state_dict().keys() == torch_model.state_dict().keys()

    outputs = []
    for model in (torch_model, rust_model):
        model.zero_grad(set_to_none=True)
        ll, loss = model._compute_loss(x, loss_bias=0, it=0, max_iter=4, t_min=1, t_max=1)
        loss.backward()
        outputs.append(
            (
                ll.item(),
                {
                    k: p.grad.detach().clone()
                    for k, p in model.named_parameters()
                    if p.grad is not None
                },
            )
        )
    (ll0, grad0), (ll1, grad1) = outputs
    assert abs(ll0 - ll1) < 1e-9
    assert grad0.keys() == grad1.keys()
    assert all(torch.allclose(grad0[k], grad1[k], atol=1e-8, rtol=1e-8) for k in grad0)

    for model in (torch_model, rust_model):
        model.optimize(x)
    assert torch_model.optimization_summary()["main_updates"] == 4
    assert rust_model.optimization_summary()["main_updates"] == 4
    assert all(
        torch.allclose(a, b, atol=1e-8, rtol=1e-8)
        for a, b in zip(torch_model.parameters(), rust_model.parameters())
    )
    rust_model.set_backend("python")
    assert torch.allclose(torch_model.log_likelihood(x), rust_model.log_likelihood(x), atol=1e-8)


def test_rust_optin_rejects_out_of_scope_inputs(rust_library):
    model = NHSMM(ModelConfig.for_mode("hmm", n_states=3, n_features=1), device="cpu")
    model.initialize_distributions(jitter=0)
    model.set_backend("rust", library_path=rust_library)
    x = torch.zeros(5, 1, dtype=torch.float64)
    with pytest.raises(ValueError, match="exactly one sequence"):
        model.log_likelihood([x, x])
    with pytest.raises(ValueError, match="CPU Float64"):
        model.log_likelihood(x.float())
    model.to(dtype=torch.float64)
    with pytest.raises(RuntimeError, match="first-order"):
        torch.autograd.grad(
            model.log_likelihood(x.double(), reduce=True), model.dist.emission.mu, create_graph=True
        )[0].sum().backward()
    model.set_backend("python")
    assert torch.isfinite(model.log_likelihood(x)).all()
