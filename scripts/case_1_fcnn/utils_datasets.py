"""Dataset creation utilities.

Index vocabulary, from the coarsest to the finest (see
``doc/description_case_1_fcnn.tex`` for the same names in mathematical form):

An image is one 120 x 840 field of total CO2 mass in kg, produced by one
university at one reporting time. It is the only kind of 2D array here.

    index_university        0..33      which submission. In the code a university
                                       is carried by its *name*, because that
                                       name is the key into the distance tables.
    index_time              0..200     which reporting time. In the code a time
                                       is carried by its *year* (0, 5, ..., 1000),
                                       because the year names the distance file:
                                       year = 5 * index_time.
    index_image_in_archive  0..6832    one column of the packed archive, i.e. one
                                       (university, time) combination.
    index_image             0..n-1     one row of the image stack loaded here; n
                                       is how many images create_datasets keeps.
    index_pair              0..n_pairs-1  one row of ``index_image_pairs``, i.e.
                                       one (image, image) couple at a common time.
    index_image_row         0..119     row of a single image.
    index_image_column      0..839     column of a single image.
"""

from pdb import set_trace as st

import csv
import json
from pathlib import Path

import numpy as np
import jax.numpy as jnp

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "spe11b_tmco2_dt50y_images.npz"
IMAGE_LABELS_PATH = BASE_DIR / "spe11b_tmco2_dt50y_indices.json"
TRAIN_SPLIT = 0.7
VALIDATION_SPLIT = 0.15

N_IMAGE_ROWS = 120
N_IMAGE_COLUMNS = 840
N_PIXELS_PER_IMAGE = N_IMAGE_ROWS * N_IMAGE_COLUMNS


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


def _load_packed_images(
    npz_path: str | Path, array_key: str = "global_array"
) -> np.ndarray:
    """The (n_pixels_per_image x n_images) archive; one column is one image."""
    with np.load(npz_path, allow_pickle=True) as archive:
        return archive[array_key]


def _read_distance_table(path: str | Path) -> dict[str, dict[str, float]]:
    """Read one year's table as distances[university_1][university_2]."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        university_names = next(reader)[1:]
        return {
            csv_row[0]: {
                university_name: float(value)
                for university_name, value in zip(university_names, csv_row[1:])
            }
            for csv_row in reader
        }


def _load_distance_table(year: int) -> dict[str, dict[str, float]]:
    """Ground-truth distance table for one year, local copy or cluster path."""
    file_name = f"spe11b_co2mass_w1_diff_{year}y.csv"
    try:
        return _read_distance_table(BASE_DIR / "dense" / file_name)
    except OSError:
        return _read_distance_table(
            f"/home/jovyan/shared_folder/evaluation/spe11b/dense/{file_name}"
        )


def _distance_lookup(
    university_1: str,
    university_2: str,
    year: int,
    table_cache: dict[int, dict[str, dict[str, float]]],
) -> float:
    """Ground-truth distance between two universities at one year.

    The tables drop the trailing "1" of single-submission groups
    ("calgary" for "calgary1"), so both labels may need stripping.
    """
    if year not in table_cache:
        table_cache[year] = _load_distance_table(year)

    distances = table_cache[year]
    row_name = (
        university_1[:-1]
        if university_1.endswith("1") and university_1 not in distances
        else university_1
    )
    row = distances[row_name]
    column_name = (
        university_2[:-1]
        if university_2.endswith("1") and university_2 not in row
        else university_2
    )
    return row[column_name]


class PairDataset:
    """Same-time image couples held as index pairs into a shared image stack.

    ``images`` stores each used image exactly once, so the couple tensor is only
    materialized by ``gather``, one batch at a time, on whatever device
    ``images`` lives on.

    Attributes:
        images: (n, N_IMAGE_ROWS, N_IMAGE_COLUMNS) stack of the scaled images,
            addressed by index_image.
        index_image_pairs: (n_pairs, 2) array; row index_pair holds the two
            index_image values of that couple, both in [0, n). They are
            positions in this stack, never index_image_in_archive values.
        distances: (n_pairs,) scaled target distance, one per index_pair.
    """

    def __init__(self, images, index_image_pairs, distances):
        self.images = images
        self.index_image_pairs = index_image_pairs
        self.distances = distances

    def __len__(self) -> int:
        return int(self.index_image_pairs.shape[0])

    @property
    def x_shape(self) -> tuple[int, ...]:
        """Shape of a single materialized couple, e.g. (2, 120, 840)."""
        return (2,) + tuple(self.images.shape[1:])

    def gather(self, index_pair_batch=None):
        """Materialize couples as (n, 2, n_rows, n_cols); all of them if None.

        ``index_pair_batch`` selects index_pair values. This is a plain
        fancy-index into ``images``, so under jit it runs on the GPU and can be
        fused into the first layer.
        """
        pairs = self.index_image_pairs
        if index_pair_batch is not None:
            pairs = pairs[index_pair_batch]
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
    Load image couples and distances, split into train/validation/test sets.

    Images are selected as index_image_in_archive in range(start, stop, step) and
    stacked in
    that order, so index_image = (index_image_in_archive - start) // step. Couples are then
    formed within each time only, and carry index_image values, not index_image_in_archive.

    Args:
        total_number_images: Upper bound for index_image_in_archive.
        step: Stride over index_image_in_archive.
        start: First index_image_in_archive kept. With the year-major ordering of
            the image labels, start=34 drops the whole year-0 block, which
            has no distance table.
        data_path: Path to the .npz data file.
        train_split: Fraction of the couples used for training.
        validation_split: Fraction of the couples used for validation.
        input_scale_range: Target range for linear scaling of x (the couples).
        output_scale_range: Target range for linear scaling of y (the distances).

    Returns:
        train_dataset, validation_dataset, test_dataset as PairDataset objects
        sharing one image stack on the GPU. Only the index_pair rows are split,
        so memory grows with the number of images, not with the (~67x larger)
        number of couples.

    With the defaults of main.py: n = 6799 images and
    n_pairs = 199 * 34**2 + 33**2 = 231133 couples. The split is contiguous and
    unshuffled over time-ordered couples, hence temporal: train 5-700y,
    validation 700-855y, test 855-1000y.
    """
    packed_images = np.asarray(_load_packed_images(data_path), dtype=np.float32)
    with open(IMAGE_LABELS_PATH, encoding="utf-8") as f:
        image_labels = json.load(f)

    distance_table_cache = {}

    stop = min(total_number_images, len(image_labels), packed_images.shape[1])
    used_indices_in_archive = list(range(start, stop, step))

    indices_in_archive_by_time = {}
    for index_image_in_archive in used_indices_in_archive:
        year = image_labels[index_image_in_archive][1]
        indices_in_archive_by_time.setdefault(year, []).append(index_image_in_archive)

    index_image_of_archive = {
        index_image_in_archive: index_image
        for index_image, index_image_in_archive in enumerate(used_indices_in_archive)
    }

    index_image_pairs, distances = [], []
    for year, indices_at_time in indices_in_archive_by_time.items():
        for index_1_in_archive in indices_at_time:
            university_1 = image_labels[index_1_in_archive][0]
            print(f"Loading couples for image {index_1_in_archive} (year {year})")
            for index_2_in_archive in indices_at_time:
                university_2 = image_labels[index_2_in_archive][0]
                distance = _distance_lookup(
                    university_1, university_2, year, distance_table_cache
                )
                index_image_pairs.append(
                    (index_image_of_archive[index_1_in_archive], index_image_of_archive[index_2_in_archive])
                )
                distances.append(float(distance))

    if not index_image_pairs:
        raise ValueError("No same-time image couples found for the selected range.")

    images = (
        packed_images[:N_PIXELS_PER_IMAGE, used_indices_in_archive]
        .T.reshape(len(used_indices_in_archive), N_IMAGE_ROWS, N_IMAGE_COLUMNS)
    )
    index_image_pairs = np.array(index_image_pairs, dtype=np.int32)
    distances = np.array(distances, dtype=np.float32)

    images, distances = scale_data(
        images, distances, input_scale_range, output_scale_range
    )

    (
        index_image_pairs_train,
        distances_train,
        index_image_pairs_validation,
        distances_validation,
        index_image_pairs_test,
        distances_test,
    ) = split_data(index_image_pairs, distances, train_split, validation_split)

    train_dataset = PairDataset(
        jnp.asarray(images),
        jnp.asarray(index_image_pairs_train),
        jnp.asarray(distances_train),
    )
    validation_dataset = PairDataset(
        jnp.asarray(images),
        jnp.asarray(index_image_pairs_validation),
        jnp.asarray(distances_validation),
    )
    test_dataset = PairDataset(
        jnp.asarray(images),
        jnp.asarray(index_image_pairs_test),
        jnp.asarray(distances_test),
    )

    return (train_dataset, validation_dataset, test_dataset)
