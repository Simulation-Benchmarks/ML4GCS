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
    try:
        path = BASE_DIR / f"dense/spe11b_co2mass_w1_diff_{year}y.csv"
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            names = header[1:]
            return {
                row[0]: {name: float(value) for name, value in zip(names, row[1:])}
                for row in reader
            }

    except:
        path = f"/home/jovyan/shared_folder/evaluation/spe11b/dense/spe11b_co2mass_w1_diff_{year}y.csv"
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


class PairDataset:
    """Same-year image pairs held as (i, j) indices into a shared image stack.

    ``images`` stores each used map exactly once, so the pair tensor is only
    materialized by ``gather``, one batch at a time, on whatever device
    ``images`` lives on. Storing the pairs directly instead would write every
    image once per pair it appears in -- about 2 * len(self) / n_images copies,
    ~67x for the SPE11B maps (0.39 GB of images -> a 26 GB pair tensor for 966
    images), which is what used to exhaust host RAM.

    Attributes:
        images: (n_images, n_rows, n_cols) stack of the scaled maps.
        pair_indices: (n_pairs, 2) rows into ``images``.
        y: (n_pairs,) scaled target distance per pair.
    """

    def __init__(self, images, pair_indices, y):
        self.images = images
        self.pair_indices = pair_indices
        self.y = y

    def __len__(self) -> int:
        return int(self.pair_indices.shape[0])

    @property
    def x_shape(self) -> tuple[int, ...]:
        """Shape of a single materialized pair, e.g. (2, 120, 840)."""
        return (2,) + tuple(self.images.shape[1:])

    def gather(self, batch_indices=None):
        """Materialize pairs as (n, 2, n_rows, n_cols); all pairs if None.

        This is a plain fancy-index into ``images``, so under jit it runs on
        the GPU and can be fused into the first layer.
        """
        pairs = self.pair_indices
        if batch_indices is not None:
            pairs = pairs[batch_indices]
        return self.images[pairs]


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
        train_dataset, validation_dataset, test_dataset as PairDataset objects
        sharing one image stack on the GPU. Only the (i, j) index pairs are
        split, so memory grows with the number of images, not with the (~67x
        larger) number of pairs.
    """
    global_array = np.asarray(_load_array_from_npz(data_path), dtype=np.float32)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    n_rows, n_cols = 120, 840
    expected_length = n_rows * n_cols
    distance_cache = {}

    stop = min(total_number_images, len(metadata), global_array.shape[1])
    used_indices = list(range(start, stop, step))
    indices_by_year = {}
    for index in used_indices:
        indices_by_year.setdefault(metadata[index][1], []).append(index)

    # global_array column index -> row in the compact image stack below.
    position = {column: row for row, column in enumerate(used_indices)}

    pair_indices, y = [], []
    for same_year_indices in indices_by_year.values():
        for i in same_year_indices:
            name1, year1 = metadata[i]
            print(f"Loading pairs for image {i} (year {year1})")
            for j in same_year_indices:
                name2, _ = metadata[j]
                distance = _distance_lookup(name1, name2, year1, distance_cache)
                pair_indices.append((position[i], position[j]))
                y.append(float(distance))

    if not pair_indices:
        raise ValueError("No same-year image pairs found for the selected range.")

    # One copy of each used map: (n_images, n_rows, n_cols).
    images = (
        global_array[:expected_length, used_indices]
        .T.reshape(len(used_indices), n_rows, n_cols)
    )
    pair_indices = np.array(pair_indices, dtype=np.int32)
    y = np.array(y, dtype=np.float32)

    # Scaling the unique images is equivalent to scaling the assembled pair
    # tensor: the pairs contain exactly these images, and duplicates cannot
    # change a min or a max.
    images, y = scale_data(images, y, input_scale_range, output_scale_range)

    (
        pair_indices_train,
        y_train,
        pair_indices_validation,
        y_validation,
        pair_indices_test,
        y_test,
    ) = split_data(pair_indices, y, train_split, validation_split)

    images = jnp.asarray(images)

    return (
        PairDataset(images, jnp.asarray(pair_indices_train), jnp.asarray(y_train)),
        PairDataset(
            images, jnp.asarray(pair_indices_validation), jnp.asarray(y_validation)
        ),
        PairDataset(images, jnp.asarray(pair_indices_test), jnp.asarray(y_test)),
    )
