"""Synthetic sequence data helpers used by tests and examples."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from nhsmm.config import DTYPE


def _validate_generation_args(
    *,
    n_states: int,
    n_features: int,
    seg_len_range: tuple[int, int],
    n_segments_per_state: int,
    noise_scale: float,
    context_dim: int | None,
    context_noise_scale: float,
) -> None:
    for name, value in (
        ("n_states", n_states),
        ("n_features", n_features),
        ("n_segments_per_state", n_segments_per_state),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be an integer >= 1")

    if (
        not isinstance(seg_len_range, tuple)
        or len(seg_len_range) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in seg_len_range)
    ):
        raise ValueError("seg_len_range must be a tuple of two integers")

    low, high = seg_len_range
    if low < 1 or high <= low:
        raise ValueError("seg_len_range must satisfy 1 <= low < high")

    for name, value in (
        ("noise_scale", noise_scale),
        ("context_noise_scale", context_noise_scale),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and >= 0")

    if context_dim is not None and (
        isinstance(context_dim, bool) or not isinstance(context_dim, int) or context_dim < 1
    ):
        raise ValueError("context_dim must be None or an integer >= 1")


def generate_gaussian_sequence(
    n_states: int,
    n_features: int,
    seed: int,
    seg_len_range: tuple[int, int] = (5, 20),
    n_segments_per_state: int = 3,
    noise_scale: float = 0.05,
    context_dim: Optional[int] = None,
    context_noise_scale: float = 0.05,
    normalize: bool = False,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor | None]:
    """Generate a segmented Gaussian latent-state sequence.

    The helper owns only a local NumPy RNG. It does not mutate PyTorch's global
    random state.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not isinstance(normalize, bool):
        raise ValueError("normalize must be a bool")

    _validate_generation_args(
        n_states=n_states,
        n_features=n_features,
        seg_len_range=seg_len_range,
        n_segments_per_state=n_segments_per_state,
        noise_scale=noise_scale,
        context_dim=context_dim,
        context_noise_scale=context_noise_scale,
    )

    rng = np.random.default_rng(seed)
    observations: list[np.ndarray] = []
    states: list[int] = []
    contexts: list[np.ndarray] = []

    base_means = rng.uniform(-1.0, 1.0, size=(n_states, n_features))
    for state in range(n_states):
        for _ in range(n_segments_per_state):
            length = int(rng.integers(*seg_len_range))
            observations.append(
                base_means[state] + rng.normal(scale=noise_scale, size=(length, n_features))
            )
            states.extend([state] * length)
            if context_dim is not None:
                contexts.append(
                    rng.normal(
                        scale=context_noise_scale,
                        size=(length, context_dim),
                    )
                )

    observation_array = np.vstack(observations)
    state_array = np.asarray(states, dtype=np.int64)
    context_array = np.vstack(contexts) if contexts else None

    if normalize:
        observation_array = (observation_array - observation_array.mean(axis=0)) / (
            observation_array.std(axis=0) + 1e-8
        )

    observation_tensor = torch.as_tensor(observation_array, dtype=DTYPE)
    context_tensor = (
        torch.as_tensor(context_array, dtype=DTYPE) if context_array is not None else None
    )
    return state_array, observation_tensor, context_tensor


class SequenceDataset(Dataset):
    """Dataset wrapper around one generated latent-state sequence."""

    def __init__(
        self,
        n_states: int,
        n_features: int,
        seed: int = 42,
        seg_len_range: tuple[int, int] = (5, 20),
        n_segments_per_state: int = 3,
        noise_scale: float = 0.05,
        context_dim: Optional[int] = None,
        context_noise_scale: float = 0.05,
        normalize: bool = False,
        variable_length: bool = False,
    ) -> None:
        self.variable_length = variable_length
        states, observations, context = generate_gaussian_sequence(
            n_states=n_states,
            n_features=n_features,
            seed=seed,
            seg_len_range=seg_len_range,
            n_segments_per_state=n_segments_per_state,
            noise_scale=noise_scale,
            context_dim=context_dim,
            context_noise_scale=context_noise_scale,
            normalize=normalize,
        )

        self.X: torch.Tensor | list[torch.Tensor] = observations
        self.C: torch.Tensor | list[torch.Tensor] | None = context
        self.states: torch.Tensor | list[torch.Tensor] = torch.as_tensor(
            states,
            dtype=torch.long,
        )

        if self.variable_length:
            self.X = [observations]
            self.states = [torch.as_tensor(states, dtype=torch.long)]
            if context is not None:
                self.C = [context]

    def __len__(self) -> int:
        return len(self.X) if self.variable_length else self.X.shape[0]

    def __getitem__(self, index: int):
        if self.variable_length:
            if self.C is not None:
                return self.X[index], self.C[index], self.states[index]
            return self.X[index], self.states[index]

        if self.C is not None:
            return self.X[index], self.C[index], self.states[index]
        return self.X[index], self.states[index]

    def collate_fn(self, batch):
        observations = [item[0] for item in batch]
        lengths = torch.tensor([len(item) for item in observations], dtype=torch.long)
        padded_observations = torch.nn.utils.rnn.pad_sequence(
            observations,
            batch_first=True,
        )

        if self.C is not None:
            contexts = [item[1] for item in batch]
            states = [item[2] for item in batch]
            padded_contexts = torch.nn.utils.rnn.pad_sequence(
                contexts,
                batch_first=True,
            )
            padded_states = torch.nn.utils.rnn.pad_sequence(
                states,
                batch_first=True,
            )
            return padded_observations, padded_contexts, padded_states, lengths

        states = [item[1] for item in batch]
        padded_states = torch.nn.utils.rnn.pad_sequence(
            states,
            batch_first=True,
        )
        return padded_observations, padded_states, lengths

    def loader(
        self,
        batch_size: int = 64,
        shuffle: bool = True,
    ) -> DataLoader:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be an integer >= 1")
        if not isinstance(shuffle, bool):
            raise ValueError("shuffle must be a bool")

        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn if self.variable_length else None,
        )
