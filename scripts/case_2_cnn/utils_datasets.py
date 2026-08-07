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
VALIDATION_SPLIT = 0.15


def _linear_scale(data: np.ndarray, feature_range: tuple[float, float]) -> np.ndarray:
    data_min = data.min()
    data_max = data.max()
    scale = data_max - data_min
    range_min, range_max = feature_range

    if scale == 0:
        data.fill(range_min)
        return data

    factor = np.float32((range_max - range_min) / scale)
    data -= data_min
    data *= factor
    data += range_min

    return data


def scale_data(
    x: np.ndarray,
    y: np.ndarray,
    input_scale_range: tuple[float, float] = (0.0, 1.0),
    output_scale_range: tuple[float, float] = (0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly scale the full x and y datasets, independently."""
    x = _linear_scale(x, input_scale_range)
    y = _linear_scale(y, output_scale_range)
    return x, y


def split_data(
    x: np.ndarray,
    y: np.ndarray,
    train_split: float = TRAIN_SPLIT,
    validation_split: float = VALIDATION_SPLIT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split x and y into train, validation, and test subsets."""
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"x and y have different lengths: {x.shape[0]} and {y.shape[0]}"
        )
    if not 0 < train_split < 1:
        raise ValueError("train_split must be between 0 and 1.")
    if not 0 < validation_split < 1:
        raise ValueError("validation_split must be between 0 and 1.")
    if train_split + validation_split >= 1:
        raise ValueError("train_split + validation_split must be less than 1.")

    train_end = int(train_split * x.shape[0])
    validation_end = train_end + int(validation_split * x.shape[0])

    return (
        x[:train_end],
        y[:train_end],
        x[train_end:validation_end],
        y[train_end:validation_end],
        x[validation_end:],
        y[validation_end:],
    )


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
    validation_split: float = VALIDATION_SPLIT,
    input_scale_range: tuple[float, float] = (0.0, 1.0),
    output_scale_range: tuple[float, float] = (0.0, 1.0),
):
    """
    Load image pairs and distances, split into train/validation/test sets.

    Args:
        total_number_images: Upper bound for image index range.
        step: Step size when iterating over image indices.
        start: Starting index for image range.
        data_path: Path to the .npz data file.
        train_split: Fraction of data to use for training.
        validation_split: Fraction of data to use for validation.
        input_scale_range: Target range for linear scaling of x (the image pairs).
        output_scale_range: Target range for linear scaling of y (the distances).

    Returns:
        x_train, y_train, x_validation, y_validation, x_test, y_test as JAX arrays.
    """
    global_array = np.asarray(_load_array_from_npz(data_path), dtype=np.float32)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    x, y = [], []
    n_rows, n_cols = 120, 840
    expected_length = n_rows * n_cols
    distance_cache = {}

    stop = min(total_number_images, len(metadata), global_array.shape[1])
    indices_by_year = {}
    for index in range(start, stop, step):
        indices_by_year.setdefault(metadata[index][1], []).append(index)

    for same_year_indices in indices_by_year.values():
        for i in same_year_indices:
            name1, year1 = metadata[i]
            img1 = global_array[:expected_length, i].reshape((n_rows, n_cols))
            for j in same_year_indices:
                name2, _ = metadata[j]
                img2 = global_array[:expected_length, j].reshape((n_rows, n_cols))
                distance = _distance_lookup(name1, name2, year1, distance_cache)

                print(f"Loading pair ({i}, {j})")
                x.append(np.stack([img1, img2]))
                y.append(float(distance))

    if not x:
        raise ValueError("No same-year image pairs found for the selected range.")

    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    x, y = scale_data(x, y, input_scale_range, output_scale_range)

    x_train, y_train, x_validation, y_validation, x_test, y_test = split_data(
        x, y, train_split, validation_split
    )

    return (
        jnp.array(x_train),
        jnp.array(y_train),
        jnp.array(x_validation),
        jnp.array(y_validation),
        jnp.array(x_test),
        jnp.array(y_test),
    )
