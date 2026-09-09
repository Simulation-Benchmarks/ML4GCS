"""Dataset utilities for Wasserstein-distance regression."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp


CASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CASE_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ml4gcs.data import load_wasserstein_arrays  # noqa: E402

TRAIN_SPLIT = 0.7
VALIDATION_SPLIT = 0.15
DATA_PATH = CASE_DIR / "wasserstein_pairs.npz"


def _linear_scale(data: np.ndarray, feature_range: tuple[float, float]) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32).copy()
    data_min = data.min()
    data_max = data.max()
    range_min, range_max = feature_range
    if data_max == data_min:
        data.fill(range_min)
        return data
    return (data - data_min) * ((range_max - range_min) / (data_max - data_min)) + range_min


def scale_data(
    x: np.ndarray,
    y: np.ndarray,
    feature_range: tuple[float, float] = (0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Scale inputs and labels independently to the requested range."""

    return _linear_scale(x, feature_range), _linear_scale(y, feature_range)


def split_data(
    x: np.ndarray,
    y: np.ndarray,
    train_split: float = TRAIN_SPLIT,
    validation_split: float = VALIDATION_SPLIT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays in their existing order into train/validation/test sets."""

    if x.shape[0] != y.shape[0]:
        raise ValueError(f"x and y have different lengths: {x.shape[0]} and {y.shape[0]}")
    if not 0 < train_split < 1 or not 0 < validation_split < 1:
        raise ValueError("split fractions must be between 0 and 1")
    if train_split + validation_split >= 1:
        raise ValueError("train_split + validation_split must be less than 1")
    train_end = int(train_split * x.shape[0])
    validation_end = train_end + int(validation_split * x.shape[0])
    return (
        x[:train_end], y[:train_end],
        x[train_end:validation_end], y[train_end:validation_end],
        x[validation_end:], y[validation_end:],
    )


def create_datasets(
    data_path: str | Path = DATA_PATH,
    train_split: float = TRAIN_SPLIT,
    validation_split: float = VALIDATION_SPLIT,
    scale_range: tuple[float, float] = (0.0, 1.0),
    **_legacy_options,
):
    """Load prepared Wasserstein pairs, scale them, and split them."""

    x, y = load_wasserstein_arrays(data_path)
    x, y = scale_data(x, y, scale_range)
    return tuple(jnp.array(part) for part in split_data(x, y, train_split, validation_split))
