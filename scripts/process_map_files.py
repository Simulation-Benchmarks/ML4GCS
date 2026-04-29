import json
import numpy as np
import pandas as pd
import os
from fnmatch import fnmatch

def main():
    base_dir = 'spe11b'
    map_file = 'map_files.txt'
    metadata_path = 'metadata.json'
    npz_path = 'spe11b_tmco2.npz'
    
    # Read the list of files
    with open(map_file, 'r') as f:
        files = [line.strip() for line in f if line.strip()]
    
    # Pattern to match
    pattern = 'spe11b_spatial_map_*y.csv'
    
    # Filter matching files
    matching_files = [f for f in files if fnmatch(os.path.basename(f), pattern)]
    
    # Collect data and metadata
    data_list = []
    metadata = []
    
    for file_path in matching_files:
        full_path = os.path.join(base_dir, file_path.lstrip('./'))
        if not os.path.exists(full_path):
            print(f"Warning: File {full_path} does not exist, skipping")
            continue
        
        try:
            df = pd.read_csv(full_path)
            col_name = ' tmCO2 [kg]'
            if col_name not in df.columns:
                print(f"Warning: Column '{col_name}' not found in {full_path}, skipping")
                continue
            
            column_data = df[col_name].values
            data_list.append(column_data)
            
            # Parse metadata
            parts = file_path.split('/')
            folder = parts[1]  # e.g., 'ifpen1'
            filename = parts[-1]  # e.g., 'spe11b_spatial_map_645y.csv'
            year_str = filename.split('_')[-1].replace('y.csv', '')
            year = int(year_str)
            metadata.append((folder, year))
        
        except Exception as e:
            print(f"Error processing {full_path}: {e}")
            continue
    
    if not data_list:
        print("No valid data found. Ensure data is downloaded and files exist.")
        return
    
    # Create the global 2D array
    global_array = np.column_stack(data_list)
    
    # Save the array in compressed format
    np.savez_compressed(npz_path, global_array=global_array)
    
    # Save metadata as JSON
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Processed {len(data_list)} files. Global array shape: {global_array.shape}")
    print(f"Saved to {npz_path} and {metadata_path}")

def get_result_name_and_year(column_index: int, metadata_path: str = 'metadata.json'):
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
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"{metadata_path} not found. Run the main function first to generate it.")
    
    if 0 <= column_index < len(metadata):
        return tuple(metadata[column_index])
    else:
        raise IndexError(f"Column index {column_index} is out of range. Valid range: 0 to {len(metadata)-1}")


def load_array_from_npz(npz_path: str = 'spe11b_tmco2.npz', array_key: str = 'global_array') -> np.ndarray:
    """Load the global array stored in a .npz archive."""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Archive not found: {npz_path}")
    with np.load(npz_path, allow_pickle=True) as archive:
        if array_key not in archive:
            raise KeyError(f"Array key '{array_key}' not found in {npz_path}")
        return archive[array_key]


def get_spatial_maps(column_index1: int, column_index2: int, npz_path: str = 'spe11b_tmco2.npz') -> tuple[np.ndarray, np.ndarray]:
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
            raise IndexError(f"Column index {idx} is out of range. Valid range: 0 to {global_array.shape[1] - 1}")

    image1 = global_array[:expected_length, column_index1].reshape((n_rows, n_cols))
    image2 = global_array[:expected_length, column_index2].reshape((n_rows, n_cols))
    return image1, image2

def get_maps_and_distance(column_index1: int, column_index2: int, npz_path: str = 'spe11b_tmco2.npz') -> tuple[np.ndarray, np.ndarray, float]:
    name1, year1 = get_result_name_and_year(column_index1)
    name2, year2 = get_result_name_and_year(column_index2)

    if year1 != year2:
        raise ValueError(f"year1 = {year1} and year2 = {year2} have to coincide.")

    filename = f"/home/jovyan/shared_folder/evaluation/spe11b/dense/spe11b_co2mass_w1_diff_{year1}y.csv"
    distances = pd.read_csv(filename, index_col=0)

    try:
        row = distances.loc[name1]
    except KeyError:
        alt_name1 = name1[:-1] if name1.endswith('1') else name1

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
        alt_name2 = name2[:-1] if name2.endswith('1') else name2

        if alt_name2 != name2:
            try:
                distance = row.loc[alt_name2]
                name2 = alt_name2
            except KeyError:
                raise
        else:
            raise

    image1, image2 = get_spatial_maps(column_index1, column_index2, npz_path)

    return image1, image2, distance
    
if __name__ == "__main__":
    main()
