"""Dataset creation utilities."""

from pdb import set_trace as st

import csv
import pickle
from pathlib import Path

import numpy as np
import jax.numpy as jnp

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "spe11b_tmco2_dt50y.npz"
METADATA_PATH = BASE_DIR / "metadata.pkl"
TRAIN_SPLIT = 0.7


def _load_array_from_npz(
    npz_path: str | Path, array_key: str = "global_array"
) -> np.ndarray:
    with np.load(npz_path, allow_pickle=True) as archive:
        return archive[array_key]


def _load_distance_table(year: int) -> dict[str, dict[str, float]]:
    path = BASE_DIR / f"dense/spe11b_co2mass_w1_diff_{year}y.csv"
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        names = header[1:]
        return {
            row[0]: {name: float(value) for name, value in zip(names, row[1:])}
            for row in reader
        }


def _distance_lookup(
    name1: str, name2: str, year: int, cache: dict[int, dict[str, dict[str, float]]]
) -> float:
    if year not in cache:
        cache[year] = _load_distance_table(year)

    distances = cache[year]
    row_name = name1[:-1] if name1.endswith("1") and name1 not in distances else name1
    row = distances[row_name]
    col_name = name2[:-1] if name2.endswith("1") and name2 not in row else name2
    return row[col_name]


def create_datasets(
    total_number_images: int = 45,
    step: int = 1,
    start: int = 35,
    data_path: str | Path = DATA_PATH,
    train_split: float = TRAIN_SPLIT,
):
    """
    Load image pairs and distances, split into train/validation sets.

    Args:
        total_number_images: Upper bound for image index range.
        step: Step size when iterating over image indices.
        start: Starting index for image range.
        data_path: Path to the .npz data file.
        train_split: Fraction of data to use for training.

    Returns:
        x_train, y_train, x_validation, y_validation as JAX arrays.
    """
    global_array = np.asarray(_load_array_from_npz(data_path), dtype=np.float32)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    x, y = [], []
    n_rows, n_cols = 120, 840
    expected_length = n_rows * n_cols
    distance_cache = {}

    indices = range(start, total_number_images, step)
    for i in indices:
        for j in indices:
            print(f"Loading pair ({i}, {j})")
            name1, year1 = metadata[i]
            name2, year2 = metadata[j]
            if year1 != year2:
                raise ValueError(
                    f"year1 = {year1} and year2 = {year2} have to coincide."
                )

            img1 = global_array[:expected_length, i].reshape((n_rows, n_cols))
            img2 = global_array[:expected_length, j].reshape((n_rows, n_cols))
            distance = _distance_lookup(name1, name2, year1, distance_cache)

            x.append(np.stack([img1, img2]))  # shape: (2, H, W)
            y.append(float(distance))

    x = jnp.array(np.array(x, dtype=np.float32))  # shape: (N, 2, H, W)
    y = jnp.array(np.array(y, dtype=np.float32))  # shape: (N,)

    split = int(train_split * x.shape[0])

    return x[:split], y[:split], x[split:], y[split:]
