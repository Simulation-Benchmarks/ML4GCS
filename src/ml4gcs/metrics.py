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


def r_squared(prediction: np.ndarray, target: np.ndarray) -> float:
    """Return the coefficient of determination (R^2) between two fields."""

    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch: {prediction.shape} vs {target.shape}")
    ss_res = float(np.sum((target - prediction) ** 2))
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def plot_r_squared(prediction: np.ndarray, target: np.ndarray, output_path: str) -> float:
    """Plot predicted vs. target values annotated with R^2 and save as a PDF.

    Returns the computed R^2 value.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"text.usetex": True, "font.family": "serif"})

    prediction = np.asarray(prediction, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    r2 = r_squared(prediction, target)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(target, prediction, s=10, alpha=0.5)
    lo = float(min(target.min(), prediction.min()))
    hi = float(max(target.max(), prediction.max()))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    ax.set_xlabel("Target")
    ax.set_ylabel("Prediction")
    ax.set_title(f"$R^2$ = {r2:.4f}")
    fig.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)

    return r2


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
