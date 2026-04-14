"""Normalization helpers for SPE11B benchmark features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class FeatureNormalizer:
    """Standardize feature matrices with training-set statistics."""

    mean: np.ndarray
    std: np.ndarray

    def normalize(self, array: np.ndarray) -> np.ndarray:
        array = np.asarray(array, dtype=np.float64)
        array = np.where(np.isnan(array), self.mean, array)
        return (array - self.mean) / self.std

    def denormalize(self, array: np.ndarray) -> np.ndarray:
        array = np.asarray(array, dtype=np.float64)
        array = np.where(np.isnan(array), 0.0, array)
        return array * self.std + self.mean


def fit_feature_normalizer(array: np.ndarray) -> FeatureNormalizer:
    """Fit a mean/std normalizer from a feature matrix."""

    array = np.asarray(array, dtype=np.float64)
    mean = np.nanmean(array, axis=0)
    std = np.nanstd(array, axis=0)
    std = np.where(std == 0.0, 1.0, std)
    return FeatureNormalizer(mean=mean, std=std)


# Backwards-compatible aliases for the earlier pressure-only prototype.
PressureNormalizer = FeatureNormalizer


def fit_pressure_normalizer(dataset) -> PressureNormalizer:
    """Fit a mean/std normalizer from a pressure-transition dataset."""

    values = []
    for i in range(len(dataset)):
        sample = dataset[i]
        values.append(sample.input_pressure.reshape(-1))
        values.append(sample.target_pressure.reshape(-1))

    concatenated = np.concatenate(values, axis=0)
    mean = float(np.mean(concatenated))
    std = float(np.std(concatenated))
    if std == 0.0:
        std = 1.0
    return PressureNormalizer(mean=mean, std=std)
