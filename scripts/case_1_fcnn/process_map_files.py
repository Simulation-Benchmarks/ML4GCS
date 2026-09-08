"""Build the packed image archive from the raw SPE11B submission files.

Input  (originals, downloaded, not produced here):
    spe11b/<university>/spe11b_spatial_map_<year>y.csv   the raw images
    map_files.txt                                        the list of their paths

Output (processed, written here):
    spe11b_tmco2_dt50y_images.npz    all images, one per column, as one matrix
    spe11b_tmco2_dt50y_indices.json  the (university, year) label of each column

Nothing else survives the run: the chunked intermediates are deleted at the end.

Index vocabulary, matching utils_datasets.py:
    index_image_in_archive  one column of the packed archive = one image
                            = one (university, year) combination
    index_image_row         0..119   row of a single image
    index_image_column      0..839   column of a single image
"""

import os
import json
import numpy as np
import pandas as pd
import pathlib
import shutil
from fnmatch import fnmatch
from typing import Optional

# The single column of a raw map CSV that we keep: total CO2 mass per cell.
CO2_MASS_COLUMN = " tmCO2 [kg]"

# Key under which the packed matrix is stored inside the .npz. Kept as-is so
# that already-generated archives stay readable.
ARCHIVE_KEY = "global_array"


def main():
    script_dir = pathlib.Path(__file__).resolve().parent

    raw_image_roots = [
        script_dir / "../../spe11b",
        script_dir / "../../../shared_folder/data/spe11b",
    ]

    raw_image_paths_file = script_dir / "map_files.txt"
    packed_images_npz_path = script_dir / "spe11b_tmco2_dt50y_images.npz"
    image_labels_json_path = script_dir / "spe11b_tmco2_dt50y_indices.json"
    # Chunked so that np.column_stack never holds every map at once.
    chunk_dir = script_dir / "spe11b_tmco2_splits"
    chunk_size = 256

    with open(raw_image_paths_file, "r") as f:
        listed_paths = [line.strip() for line in f if line.strip()]

    raw_image_paths = [
        path
        for path in listed_paths
        if fnmatch(os.path.basename(path), "spe11b_spatial_map_*y.csv")
    ]

    image_columns_buffer = []
    image_labels = []
    chunk_npz_paths = []
    chunk_labels_paths = []

    def resolve_raw_image_path(listed_path: str) -> Optional[pathlib.Path]:
        relative_path = pathlib.Path(listed_path.lstrip("./"))
        for raw_image_root in raw_image_roots:
            full_path = raw_image_root / relative_path
            if full_path.exists():
                return full_path
        return None

    if chunk_dir.exists():
        shutil.rmtree(chunk_dir)
    chunk_dir.mkdir(parents=True)

    def save_chunk(index_chunk: int) -> None:
        if not image_columns_buffer:
            return

        chunk_array = np.column_stack(image_columns_buffer)
        chunk_npz_path = chunk_dir / f"spe11b_tmco2_part_{index_chunk:04d}.npz"
        chunk_labels_path = chunk_dir / f"metadata_part_{index_chunk:04d}.json"
        np.savez_compressed(chunk_npz_path, **{ARCHIVE_KEY: chunk_array})

        first_index_in_archive = len(image_labels) - len(image_columns_buffer)
        with open(chunk_labels_path, "w", encoding="utf-8") as f:
            json.dump(image_labels[first_index_in_archive:], f, indent=2)

        chunk_npz_paths.append(chunk_npz_path)
        chunk_labels_paths.append(chunk_labels_path)
        print(
            f"Saved chunk {index_chunk} with shape {chunk_array.shape} "
            f"to {chunk_npz_path}"
        )
        image_columns_buffer.clear()

    index_chunk = 0
    for listed_path in raw_image_paths:
        full_path = resolve_raw_image_path(listed_path)
        if full_path is None:
            attempted_paths = [
                str(raw_image_root / listed_path.lstrip("./"))
                for raw_image_root in raw_image_roots
            ]
            print(
                f"Warning: File {listed_path} does not exist in any configured data "
                f"directory. Tried: {attempted_paths}. Skipping"
            )
            continue

        try:
            print(f"Processing {full_path}")
            raw_image_table = pd.read_csv(full_path)
            if CO2_MASS_COLUMN not in raw_image_table.columns:
                print(
                    f"Warning: Column '{CO2_MASS_COLUMN}' not found in {full_path}, "
                    f"skipping"
                )
                continue

            image_column = pd.to_numeric(
                raw_image_table[CO2_MASS_COLUMN], errors="coerce"
            ).to_numpy()
            image_column = np.nan_to_num(image_column, nan=0.0)
            # Store as float32: the maps are CO2 masses that don't need float64
            # precision, and float32 halves the on-disk .npz and the host-RAM
            # footprint of the packed matrix (~2.75 GB instead of ~5.5 GB for the
            # full 100800 x 6833 array). utils_datasets already casts to float32
            # on load, so this just moves the cast upstream and avoids a
            # transient double allocation there.
            image_column = image_column.astype(np.float32)
            image_columns_buffer.append(image_column)

            # ./<university>/spe11b_spatial_map_<year>y.csv
            path_parts = listed_path.split("/")
            university = path_parts[1]
            file_name = path_parts[-1]
            year = int(file_name.split("_")[-1].replace("y.csv", ""))
            image_labels.append((university, year))

            if len(image_columns_buffer) >= chunk_size:
                save_chunk(index_chunk)
                index_chunk += 1

        except Exception as e:
            print(f"Error processing {full_path}: {e}")
            continue

    save_chunk(index_chunk)

    if not chunk_npz_paths:
        print("No valid data found. Ensure data is downloaded and files exist.")
        return

    with np.load(chunk_npz_paths[0]) as archive:
        first_chunk = archive[ARCHIVE_KEY]
        n_pixels_per_image = first_chunk.shape[0]
        dtype = first_chunk.dtype

    packed_images = np.lib.format.open_memmap(
        chunk_dir / "spe11b_tmco2_joined.npy",
        mode="w+",
        dtype=dtype,
        shape=(n_pixels_per_image, len(image_labels)),
    )

    index_start_in_archive = 0
    all_image_labels = []
    for chunk_npz_path, chunk_labels_path in zip(chunk_npz_paths, chunk_labels_paths):
        with np.load(chunk_npz_path) as archive:
            chunk_array = archive[ARCHIVE_KEY]
            index_end_in_archive = index_start_in_archive + chunk_array.shape[1]
            packed_images[:, index_start_in_archive:index_end_in_archive] = chunk_array
            index_start_in_archive = index_end_in_archive

        with open(chunk_labels_path, "r", encoding="utf-8") as f:
            all_image_labels.extend(json.load(f))

    packed_images.flush()

    np.savez_compressed(packed_images_npz_path, **{ARCHIVE_KEY: packed_images})

    with open(image_labels_json_path, "w", encoding="utf-8") as f:
        json.dump(all_image_labels, f, indent=2)

    n_images = packed_images.shape[1]
    del packed_images, chunk_array  # release the memmaps before removing the files
    shutil.rmtree(chunk_dir)

    print(f"Processed {len(all_image_labels)} files. Packed archive: {n_pixels_per_image} x {n_images}")
    print(f"Saved {packed_images_npz_path} and {image_labels_json_path}")


def get_university_and_year(
    index_image_in_archive: int, image_labels_path: str = "spe11b_tmco2_dt50y_indices.json"
) -> tuple[str, int]:
    """Label of one column of the packed archive.

    Args:
        index_image_in_archive: Column index (0-based).

    Returns:
        (university, year), e.g. ('calgary1', 5).

    Raises:
        IndexError: If index_image_in_archive is out of range.
        FileNotFoundError: If image_labels_path is not found.
    """
    try:
        with open(image_labels_path, "r", encoding="utf-8") as f:
            image_labels = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{image_labels_path} not found. Run the main function first to generate it."
        )

    if 0 <= index_image_in_archive < len(image_labels):
        return tuple(image_labels[index_image_in_archive])
    raise IndexError(
        f"index_image_in_archive {index_image_in_archive} is out of range. Valid range: 0 to {len(image_labels)-1}"
    )


def load_packed_images(
    npz_path: str = "spe11b_tmco2_dt50y_images.npz", array_key: str = ARCHIVE_KEY
) -> np.ndarray:
    """Load the packed (pixels x images) archive stored in a .npz archive."""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Archive not found: {npz_path}")
    with np.load(npz_path, allow_pickle=True) as archive:
        if array_key not in archive:
            raise KeyError(f"Array key '{array_key}' not found in {npz_path}")
        return archive[array_key]


def get_images(
    index_1_in_archive: int, index_2_in_archive: int, npz_path: str = "spe11b_tmco2_dt50y_images.npz"
) -> tuple[np.ndarray, np.ndarray]:
    """Return two columns of the packed archive as 120x840 images.

    Each column is flattened row-major: the first 840 entries are image row 0,
    the next 840 are image row 1, and so on.
    """
    packed_images = load_packed_images(npz_path)
    n_image_rows = 120
    n_image_columns = 840
    n_pixels_per_image = n_image_rows * n_image_columns

    if packed_images.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {packed_images.shape}")
    if packed_images.shape[0] < n_pixels_per_image:
        raise ValueError(
            f"Array has too few rows ({packed_images.shape[0]}); expected at least "
            f"{n_pixels_per_image} to reshape into {n_image_rows}x{n_image_columns}."
        )
    for index_image_in_archive in (index_1_in_archive, index_2_in_archive):
        if index_image_in_archive < 0 or index_image_in_archive >= packed_images.shape[1]:
            raise IndexError(
                f"index_image_in_archive {index_image_in_archive} is out of range. "
                f"Valid range: 0 to {packed_images.shape[1] - 1}"
            )

    def as_image(index_image_in_archive: int) -> np.ndarray:
        return (
            packed_images[:n_pixels_per_image, index_image_in_archive]
            .astype(float)
            .reshape((n_image_rows, n_image_columns))
        )

    return as_image(index_1_in_archive), as_image(index_2_in_archive)


def get_distance(year: int, university_1: str, university_2: str) -> float:
    """Ground-truth distance between two universities at one year."""
    file_name = (
        "/home/jovyan/shared_folder/evaluation/spe11b/dense/"
        f"spe11b_co2mass_w1_diff_{year}y.csv"
    )
    distances = pd.read_csv(file_name, index_col=0)

    try:
        row = distances.loc[university_1]
    except KeyError:
        # The tables drop the trailing "1" of single-submission groups.
        alternative = university_1[:-1] if university_1.endswith("1") else university_1
        if alternative == university_1:
            raise
        row = distances.loc[alternative]

    try:
        return row.loc[university_2]
    except KeyError:
        alternative = university_2[:-1] if university_2.endswith("1") else university_2
        if alternative == university_2:
            raise
        return row.loc[alternative]


def get_maps_and_distance(
    index_1_in_archive: int,
    index_2_in_archive: int,
    npz_path: str = "spe11b_tmco2_dt50y_images.npz",
    image_labels_path: str = "spe11b_tmco2_dt50y_indices.json",
) -> tuple[np.ndarray, np.ndarray, float]:
    image_1, image_2 = get_images(index_1_in_archive, index_2_in_archive, npz_path)

    university_1, year_1 = get_university_and_year(index_1_in_archive, image_labels_path)
    university_2, year_2 = get_university_and_year(index_2_in_archive, image_labels_path)

    if year_1 != year_2:
        raise ValueError(f"year_1 = {year_1} and year_2 = {year_2} have to coincide.")

    return image_1, image_2, get_distance(year_1, university_1, university_2)


def get_all_distances(
    image_labels_path: str = "spe11b_metadata.json",
    distance_npz_path: str = "spe11b_distances.npz",
) -> None:
    with open(image_labels_path, "r", encoding="utf-8") as f:
        image_labels = json.load(f)

    pairs = []
    values = []
    for index_1_in_archive in range(len(image_labels)):
        for index_2_in_archive in range(index_1_in_archive + 1, len(image_labels)):
            university_1, year_1 = image_labels[index_1_in_archive]
            university_2, year_2 = image_labels[index_2_in_archive]

            if year_1 != year_2 or year_1 < 1:
                continue

            try:
                distance = get_distance(year_1, university_1, university_2)
                pairs.append((index_1_in_archive, index_2_in_archive))
                values.append(distance)
            except KeyError:
                print(
                    f"Warning: Distance not found for {university_1} and "
                    f"{university_2} in year {year_1}, skipping."
                )
                continue

    np.savez_compressed(
        distance_npz_path,
        pairs=np.array(pairs, dtype=int),
        distances=np.array(values, dtype=float),
    )


if __name__ == "__main__":
    main()
