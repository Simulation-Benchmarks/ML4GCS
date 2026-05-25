import os
import json
import numpy as np
import pandas as pd
import pathlib
import pickle
import shutil
from fnmatch import fnmatch
from typing import Optional


def main():
    script_dir = pathlib.Path(__file__).resolve().parent

    base_dirs = [
        script_dir / "../../spe11b",
        script_dir / "../../../shared_folder/data/spe11b",
    ]

    map_file = script_dir / "map_files.txt"
    metadata_path = script_dir / "spe11b_metadata_dt50y.json"
    metadata_pickle_path = script_dir / "metadata.pkl"
    npz_path = script_dir / "spe11b_tmco2_dt50y.npz"
    split_dir = script_dir / "spe11b_tmco2_splits"
    split_size = 256

    # Read the list of files
    with open(map_file, "r") as f:
        files = [line.strip() for line in f if line.strip()]

    # Pattern to match
    pattern = "spe11b_spatial_map_*y.csv"

    # Filter matching files
    matching_files = [f for f in files if fnmatch(os.path.basename(f), pattern)]

    # Collect data and metadata in chunks to avoid one huge np.column_stack peak.
    data_list = []
    metadata = []
    split_paths = []
    split_metadata_paths = []
    col_name = " tmCO2 [kg]"

    def resolve_map_file(file_path: str) -> Optional[pathlib.Path]:
        relative_path = pathlib.Path(file_path.lstrip("./"))
        for base_dir in base_dirs:
            full_path = base_dir / relative_path
            if full_path.exists():
                return full_path
        return None

    if split_dir.exists():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True)

    def save_split(split_index: int) -> None:
        if not data_list:
            return

        split_array = np.column_stack(data_list)
        split_npz_path = split_dir / f"spe11b_tmco2_part_{split_index:04d}.npz"
        split_metadata_path = split_dir / f"metadata_part_{split_index:04d}.json"
        np.savez_compressed(split_npz_path, global_array=split_array)

        start = len(metadata) - len(data_list)
        split_metadata = metadata[start:]
        with open(split_metadata_path, "w", encoding="utf-8") as f:
            json.dump(split_metadata, f, indent=2)

        split_paths.append(split_npz_path)
        split_metadata_paths.append(split_metadata_path)
        print(
            f"Saved split {split_index} with shape {split_array.shape} to {split_npz_path}"
        )
        data_list.clear()

    split_index = 0
    for file_path in matching_files:
        full_path = resolve_map_file(file_path)
        if full_path is None:
            attempted_paths = [
                str(base_dir / file_path.lstrip("./")) for base_dir in base_dirs
            ]
            print(
                f"Warning: File {file_path} does not exist in any configured data directory. "
                f"Tried: {attempted_paths}. Skipping"
            )
            continue

        try:
            print(f"Processing {full_path}")
            df = pd.read_csv(full_path)
            if col_name not in df.columns:
                print(
                    f"Warning: Column '{col_name}' not found in {full_path}, skipping"
                )
                continue

            column_data = pd.to_numeric(df[col_name], errors="coerce").to_numpy()
            column_data = np.nan_to_num(column_data, nan=0.0)
            data_list.append(column_data)

            # Parse metadata
            parts = file_path.split("/")
            folder = parts[1]  # e.g., 'ifpen1'
            filename = parts[-1]  # e.g., 'spe11b_spatial_map_645y.csv'
            year_str = filename.split("_")[-1].replace("y.csv", "")
            year = int(year_str)
            metadata.append((folder, year))

            if len(data_list) >= split_size:
                save_split(split_index)
                split_index += 1

        except Exception as e:
            print(f"Error processing {full_path}: {e}")
            continue

    save_split(split_index)

    if not split_paths:
        print("No valid data found. Ensure data is downloaded and files exist.")
        return

    with np.load(split_paths[0]) as archive:
        first_split = archive["global_array"]
        n_rows = first_split.shape[0]
        dtype = first_split.dtype

    global_shape = (n_rows, len(metadata))
    global_array = np.lib.format.open_memmap(
        split_dir / "spe11b_tmco2_joined.npy",
        mode="w+",
        dtype=dtype,
        shape=global_shape,
    )

    column_start = 0
    joined_metadata = []
    for split_npz_path, split_metadata_path in zip(split_paths, split_metadata_paths):
        with np.load(split_npz_path) as archive:
            split_array = archive["global_array"]
            column_end = column_start + split_array.shape[1]
            global_array[:, column_start:column_end] = split_array
            column_start = column_end

        with open(split_metadata_path, "r", encoding="utf-8") as f:
            joined_metadata.extend(json.load(f))

    global_array.flush()

    # Save the joined array in compressed format
    np.savez_compressed(npz_path, global_array=global_array)

    # Save metadata as JSON
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(joined_metadata, f, indent=2)

    # Save metadata in the format expected by utils_datasets.py
    with open(metadata_pickle_path, "wb") as f:
        pickle.dump(joined_metadata, f)

    print(
        f"Processed {len(joined_metadata)} files. Global array shape: {global_array.shape}"
    )
    print(f"Saved to {npz_path}, {metadata_path}, and {metadata_pickle_path}")


def get_result_name_and_year(
    column_index: int, metadata_path: str = "spe11b_metadata_dt50y.json"
):
    """
    Given a column index in the global array, return the result name (folder) and year.

    Args:
        column_index (int): The column index (0-based).

    Returns:
        tuple: (result_name, year) where result_name is the folder name (e.g., 'calgary1'),
               and year is the integer year extracted from the filename.

    Raises:
        IndexError: If column_index is out of range.
        FileNotFoundError: If metadata_path is not found.
    """
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{metadata_path} not found. Run the main function first to generate it."
        )

    if 0 <= column_index < len(metadata):
        return tuple(metadata[column_index])
    else:
        raise IndexError(
            f"Column index {column_index} is out of range. Valid range: 0 to {len(metadata)-1}"
        )


def load_array_from_npz(
    npz_path: str = "spe11b_tmco2_dt50y.npz", array_key: str = "global_array"
) -> np.ndarray:
    """Load the global array stored in a .npz archive."""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Archive not found: {npz_path}")
    with np.load(npz_path, allow_pickle=True) as archive:
        if array_key not in archive:
            raise KeyError(f"Array key '{array_key}' not found in {npz_path}")
        return archive[array_key]


def get_spatial_maps(
    column_index1: int, column_index2: int, npz_path: str = "spe11b_tmco2_dt50y.npz"
) -> tuple[np.ndarray, np.ndarray]:
    """Return two columns from the global array as 120x840 images.

    The first 840 entries of each column form the first row of the image,
    the next 840 entries form the second row, and so on.
    """
    global_array = load_array_from_npz(npz_path)
    n_rows = 120
    n_cols = 840
    expected_length = n_rows * n_cols

    if global_array.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {global_array.shape}")
    if global_array.shape[0] < expected_length:
        raise ValueError(
            f"Array has too few rows ({global_array.shape[0]}); expected at least {expected_length} to reshape into {n_rows}x{n_cols}."
        )
    for idx in (column_index1, column_index2):
        if idx < 0 or idx >= global_array.shape[1]:
            raise IndexError(
                f"Column index {idx} is out of range. Valid range: 0 to {global_array.shape[1] - 1}"
            )

    image1 = (
        global_array[:expected_length, column_index1]
        .astype(float)
        .reshape((n_rows, n_cols))
    )
    image2 = (
        global_array[:expected_length, column_index2]
        .astype(float)
        .reshape((n_rows, n_cols))
    )
    return image1, image2


def get_distance(year: int, name1: str, name2: str) -> float:
    filename = f"/home/jovyan/shared_folder/evaluation/spe11b/dense/spe11b_co2mass_w1_diff_{year}y.csv"
    distances = pd.read_csv(filename, index_col=0)

    try:
        row = distances.loc[name1]
    except KeyError:
        alt_name1 = name1[:-1] if name1.endswith("1") else name1

        if alt_name1 != name1:
            try:
                row = distances.loc[alt_name1]
                name1 = alt_name1
            except KeyError:
                raise
        else:
            raise

    try:
        distance = row.loc[name2]
    except KeyError:
        alt_name2 = name2[:-1] if name2.endswith("1") else name2

        if alt_name2 != name2:
            try:
                distance = row.loc[alt_name2]
                name2 = alt_name2
            except KeyError:
                raise
        else:
            raise
    return distance


def get_maps_and_distance(
    column_index1: int,
    column_index2: int,
    npz_path: str = "spe11b_tmco2_dt50y.npz",
    metadata_path: str = "spe11b_metadata_dt50y.json",
) -> tuple[np.ndarray, np.ndarray, float]:
    image1, image2 = get_spatial_maps(column_index1, column_index2, npz_path)

    name1, year1 = get_result_name_and_year(column_index1, metadata_path)
    name2, year2 = get_result_name_and_year(column_index2, metadata_path)

    if year1 != year2:
        raise ValueError(f"year1 = {year1} and year2 = {year2} have to coincide.")

    distance = get_distance(year1, name1, name2)

    return image1, image2, distance


def get_all_distances(
    metadata_path: str = "spe11b_metadata.json",
    distance_npz_path: str = "spe11b_distances.npz",
) -> None:
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    pairs = []
    values = []
    for i in range(len(metadata)):
        for j in range(i + 1, len(metadata)):
            name1, year1 = metadata[i]
            name2, year2 = metadata[j]

            if year1 != year2 or year1 < 1:
                continue

            try:
                distance = get_distance(year1, name1, name2)
                pairs.append((i, j))
                values.append(distance)
            except KeyError:
                print(
                    f"Warning: Distance not found for {name1} and {name2} in year {year1}, skipping."
                )
                continue

    np.savez_compressed(
        distance_npz_path,
        pairs=np.array(pairs, dtype=int),
        distances=np.array(values, dtype=float),
    )


if __name__ == "__main__":
    main()
