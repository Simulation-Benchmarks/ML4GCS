import numpy as np
import pandas as pd
import os
import pickle
from fnmatch import fnmatch

def main():
    base_dir = 'spe11b'
    map_file = 'map_files.txt'
    
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
            col_name = 'tmCO2 [kg]'
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
    np.savez_compressed('spe11b_tmco2.npz', global_array=global_array)
    
    # Save metadata
    with open('metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"Processed {len(data_list)} files. Global array shape: {global_array.shape}")
    print("Saved to 'spe11b_tmco2.npz' and 'metadata.pkl'")

def get_result_name_and_year(column_index):
    """
    Given a column index in the global array, return the result name (folder) and year.
    
    Args:
        column_index (int): The column index (0-based).
    
    Returns:
        tuple: (result_name, year) where result_name is the folder name (e.g., 'calgary1'),
               and year is the integer year extracted from the filename.
    
    Raises:
        IndexError: If column_index is out of range.
        FileNotFoundError: If metadata.pkl is not found.
    """
    try:
        with open('metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("metadata.pkl not found. Run the main function first to generate it.")
    
    if 0 <= column_index < len(metadata):
        return metadata[column_index]
    else:
        raise IndexError(f"Column index {column_index} is out of range. Valid range: 0 to {len(metadata)-1}")

if __name__ == "__main__":
    main()
