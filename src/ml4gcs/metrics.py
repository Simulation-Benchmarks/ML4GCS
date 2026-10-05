"""Benchmark metrics for SPE11B surrogate models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import time
from typing import Any

import numpy as np


def pressure_l2_distance(prediction: np.ndarray, target: np.ndarray) -> float:
    """Return the L2 distance between two pressure fields."""

    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch: {prediction.shape} vs {target.shape}")
    return float(np.linalg.norm(prediction - target))


def pressure_relative_l2_distance(prediction: np.ndarray, target: np.ndarray) -> float:
    """Return the relative L2 distance between two pressure fields."""

    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    denom = float(np.linalg.norm(target))
    if denom == 0.0:
        return pressure_l2_distance(prediction, target)
    return pressure_l2_distance(prediction, target) / denom


def pressure_rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    """Return the root-mean-square error between two pressure fields."""

    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch: {prediction.shape} vs {target.shape}")
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def count_parameters(model: Any, trainable_only: bool = True) -> int:
    """Count parameters in a model-like object.

    Supports objects that expose ``parameters()`` like PyTorch modules or
    ``state_dict()``-style mappings of numpy arrays / tensors.
    """

    if hasattr(model, "parameters"):
        total = 0
        for param in model.parameters():
            if trainable_only and hasattr(param, "requires_grad") and not param.requires_grad:
                continue
            total += int(np.prod(tuple(param.shape)))
        return total

    if hasattr(model, "state_dict"):
        state_dict = model.state_dict()
        total = 0
        for value in state_dict.values():
            shape = getattr(value, "shape", None)
            if shape is None:
                continue
            total += int(np.prod(tuple(shape)))
        return total

    raise TypeError("Unsupported model type for parameter counting.")


@dataclass(frozen=True, slots=True)
class InferenceTiming:
    """Summary statistics for inference latency."""

    mean_seconds: float
    std_seconds: float
    min_seconds: float
    max_seconds: float
    repeats: int


def benchmark_inference(
    predict_fn: Callable[[Any], Any],
    example_input: Any,
    repeats: int = 20,
    warmup: int = 5,
) -> InferenceTiming:
    """Measure average inference time for a callable."""

    durations: list[float] = []

    for _ in range(warmup):
        predict_fn(example_input)

    for _ in range(repeats):
        start = time.perf_counter()
        predict_fn(example_input)
        durations.append(time.perf_counter() - start)

    mean_seconds = float(np.mean(durations))
    std_seconds = float(np.std(durations))
    return InferenceTiming(
        mean_seconds=mean_seconds,
        std_seconds=std_seconds,
        min_seconds=float(np.min(durations)),
        max_seconds=float(np.max(durations)),
        repeats=repeats,
    )
