"""Decision rules checked against exhaustive probability-space path enumeration."""

import numpy as np
import pytest
import torch

from scripts.diagnose_trading_causal_decisions import causal_decisions


def enumerate_prefixes(initial, emission, duration, transition, tail):
    """Independent explicit state/age paths, without production inference helpers."""
    k, d = duration.shape[1:]
    paths = [((s,), (1,), initial[s] * emission[0, s]) for s in range(k)]
    prefixes = [paths]
    for t in range(1, len(emission)):
        extended = []
        for states, ages, weight in paths:
            state, age = states[-1], ages[-1]
            p = duration[t - 1, state]
            survival = p[age - 1 :].sum()
            end = p[age - 1] / survival if survival else 0.0
            if age == d and tail is not None:
                end = tail[state]
            if 1 - end > 0:
                extended.append(
                    (
                        states + (state,),
                        ages + (min(age + 1, d),),
                        weight * (1 - end) * emission[t, state],
                    )
                )
            for dst in range(k):
                mass = weight * end * transition[t - 1, state, age - 1, dst] * emission[t, dst]
                if mass > 0:
                    extended.append((states + (dst,), ages + (1,), mass))
        paths = extended
        prefixes.append(paths)
    return prefixes


@pytest.mark.parametrize(
    "k,d,tail_value",
    [(1, 1, 0.3), (2, 1, None), (2, 1, 0.4), (2, 2, None), (2, 2, 0.25), (2, 3, 1.0)],
)
def test_decisions_match_independent_exhaustive_paths(k, d, tail_value):
    rng = np.random.default_rng(812 + d)
    initial = rng.uniform(0.1, 1, k)
    initial /= initial.sum()
    emission = rng.uniform(0.1, 2, (5, k))
    duration = rng.uniform(0.1, 1, (5, k, d))
    duration /= duration.sum(-1, keepdims=True)
    transition = rng.uniform(0.1, 1, (5, k, d, k))
    transition /= transition.sum(-1, keepdims=True)
    tail = None if tail_value is None else np.full(k, tail_value)
    prefixes = enumerate_prefixes(initial, emission, duration, transition, tail)
    expected_map, expected_age, committed = [], [], []
    for paths in prefixes:
        best = max(paths, key=lambda path: path[2])
        expected_map.append(best[0][-1])
        expected_age.append(best[1][-1])
        state_mass = np.zeros(k)
        for states, _, weight in paths:
            if list(states[:-1]) == committed:
                state_mass[states[-1]] += weight
        assert state_mass.sum() > 0
        committed.append(int(state_mass.argmax()))
    tensors = [
        torch.tensor(a, dtype=torch.float64).log()[None]
        for a in (initial, emission, duration, transition)
    ]
    result, age = causal_decisions(*tensors, None if tail is None else torch.tensor(tail))
    assert result["prefix_map_endpoint"][0].tolist() == expected_map
    # In K=1,D=1 the reset/continuation paths share exactly the same endpoint age.
    assert age[0].tolist() == expected_age
    assert result["committed_state"][0].tolist() == committed


def test_decisions_are_prefix_invariant_and_batched():
    torch.manual_seed(6)
    b, time, k, d = 2, 6, 2, 3
    initial = torch.randn(b, k, dtype=torch.float64).log_softmax(-1)
    emission = torch.randn(b, time, k, dtype=torch.float64)
    duration = torch.randn(b, time, k, d, dtype=torch.float64).log_softmax(-1)
    transition = torch.randn(b, time, k, d, k, dtype=torch.float64).log_softmax(-1)
    tail = torch.tensor([0.2, 0.3], dtype=torch.float64)
    full, age = causal_decisions(initial, emission, duration, transition, tail)
    for length in range(1, time + 1):
        prefix, pa = causal_decisions(
            initial, emission[:, :length], duration[:, :length], transition[:, :length], tail
        )
        for rule in full:
            assert torch.equal(prefix[rule], full[rule][:, :length])
        assert torch.equal(pa, age[:, :length])
    for row in range(b):
        single, _ = causal_decisions(
            initial[row : row + 1],
            emission[row : row + 1],
            duration[row : row + 1],
            transition[row : row + 1],
            tail,
        )
        for rule in full:
            assert torch.equal(single[rule][0], full[rule][row])


def test_committed_rule_preserves_support_when_marginal_mode_revises():
    initial = torch.tensor([[0.51, 0.49]], dtype=torch.float64).log()
    emission = torch.tensor([[[1.0, 1.0], [1.0, 2.0]]], dtype=torch.float64).log()
    duration = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]] * 2], dtype=torch.float64).log()
    transition = torch.full((1, 2, 2, 2, 2), 0.5, dtype=torch.float64).log()
    result, _ = causal_decisions(initial, emission, duration, transition, None)
    assert result["prefix_map_endpoint"].tolist() == [[0, 1]]
    assert result["committed_state"].tolist() == [[0, 0]]


def test_impossible_committed_observation_fails_explicitly():
    initial = torch.tensor([[1.0, 0.0]], dtype=torch.float64).log()
    emission = torch.tensor([[[1.0, 1.0], [0.0, 1.0]]], dtype=torch.float64).log()
    duration = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]] * 2], dtype=torch.float64).log()
    transition = torch.full((1, 2, 2, 2, 2), 0.5, dtype=torch.float64).log()
    with pytest.raises(ValueError):
        causal_decisions(initial, emission, duration, transition, None)
