"""Helpers for splitting SPE11B pressure-transition datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class IndexSubset:
    """A lightweight view over selected items from a dataset."""

    dataset: object
    indices: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]


def _split_counts(n_items: int, fractions: Sequence[float]) -> tuple[int, int, int]:
    if len(fractions) != 3:
        raise ValueError("Expected three fractions: train, val, test.")
    if not abs(sum(fractions) - 1.0) < 1e-8:
        raise ValueError("Split fractions must sum to 1.0.")

    train = int(round(n_items * fractions[0]))
    val = int(round(n_items * fractions[1]))
    test = n_items - train - val
    if test < 0:
        raise ValueError("Split fractions produced a negative test set size.")
    return train, val, test


def split_indices(
    n_items: int,
    fractions: Sequence[float] = (0.7, 0.15, 0.15),
    seed: int = 0,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Split indices into train/val/test partitions."""

    import random

    indices = list(range(n_items))
    random.Random(seed).shuffle(indices)
    train_n, val_n, test_n = _split_counts(n_items, fractions)
    train = tuple(indices[:train_n])
    val = tuple(indices[train_n : train_n + val_n])
    test = tuple(indices[train_n + val_n : train_n + val_n + test_n])
    return train, val, test


def make_subsets(dataset, fractions: Sequence[float] = (0.7, 0.15, 0.15), seed: int = 0):
    """Split a dataset into train/val/test subsets."""

    train_idx, val_idx, test_idx = split_indices(len(dataset), fractions=fractions, seed=seed)
    return (
        IndexSubset(dataset=dataset, indices=train_idx),
        IndexSubset(dataset=dataset, indices=val_idx),
        IndexSubset(dataset=dataset, indices=test_idx),
    )

