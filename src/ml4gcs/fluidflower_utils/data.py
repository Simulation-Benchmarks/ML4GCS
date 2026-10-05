import numpy as np
from pathlib import Path
from typing import Tuple
from dataclasses import dataclass

ALL_TIMESTAMPS = sorted(
    ["0{k}_{idx}{m}" for k in range(3) for idx in range(6) for m in [0, 2, 4, 6, 8]]
    + ["0{k}_{idx}0" for k in range(2, 10) for idx in range(6)]
    + ["1{k}_{idx}0" for k in range(3) for idx in range(6)]
    + ["{k}_{idx}0" for k in range(13, 73) for idx in [0, 3]]
)

TIMESTAMPS = ALL_TIMESTAMPS  # Alias for convenience
SELECTED_TIMESTAMPS = ["02_30", "05_00", "10_00", "20_00", "50_00", "72_00"]


@dataclass
class FluidFlowerData:
    coordinates_x: np.ndarray
    coordinates_y: np.ndarray
    data_dict: dict  # Dictionary to hold flexible data columns


def load(
    path: Path, data_type: str = "spatial_map"
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Generalized loader for FluidFlower CSV files.

    Parameters
    ----------
    path : Path
        Path to the CSV file to load.
    data_type : str
        Type of data to load: "facies", "spatial_map", or other.
        Determines which columns are extracted.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, dict]
        coordinates_x, coordinates_y, and a dict with data columns.
        - For "facies": dict contains 'facies'
        - For "spatial_map": dict contains 'saturation' and 'mass'
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV file {path} does not exist.")

    # Read csv
    delimiter: str = ","
    try:
        data = np.loadtxt(path, delimiter=delimiter)
    except ValueError:
        data = np.loadtxt(path, delimiter=delimiter, skiprows=1)

    # Extract coordinates (x, y) - always first two columns
    coordinates_x = data[:, 0]
    coordinates_y = data[:, 1]

    # Extract data based on type
    data_dict = {}
    if data_type == "facies":
        # Facies file has: x, y, facies
        data_dict["facies"] = data[:, 2]
    elif data_type == "spatial_map":
        # Spatial map has: x, y, saturation, mass
        data_dict["saturation"] = data[:, 2]
        data_dict["mass"] = data[:, 3]
    else:
        # Generic: store all columns after x, y
        for i in range(2, data.shape[1]):
            data_dict[f"column_{i}"] = data[:, i]

    return coordinates_x, coordinates_y, data_dict


def to_2d_array(
    coordinates_x: np.ndarray, coordinates_y: np.ndarray, values: np.ndarray
) -> np.ndarray:
    """
    Convert 1D coordinate-value data to 2D grid array.

    Parameters
    ----------
    coordinates_x : np.ndarray
        1D array of x coordinates
    coordinates_y : np.ndarray
        1D array of y coordinates
    values : np.ndarray
        1D array of values at each coordinate

    Returns
    -------
    np.ndarray
        2D array with shape (n_y, n_x) in image coordinates
    """
    # Determine the shape = frequency of x_coordinates (fastest changing) and y_coordinates (slowest changing)
    unique_x = np.unique(coordinates_x)
    unique_y = np.unique(coordinates_y)

    row = len(unique_y)
    col = len(unique_x)
    shape = (row, col)

    # Reshape values to the determined shape, remember that the values are ordered wrt. Euclidean coordinates,
    # with x changing fastest, so we need to reshape accordingly. Also, we need to switch to row-col order,
    # with origin at the top-left corner, so we need to flip the y-coordinates and reshape accordingly.
    values_reshaped = values.reshape(
        shape, order="F"
    )  # Reshape to (row, col) with Fortran order (y changes fastest)
    values_reshaped = np.flip(values_reshaped, axis=0)  # Flip

    return values_reshaped


def to_1d_array(values: np.ndarray) -> np.ndarray:
    # Flip values back to original order and reshape to 1D array
    values_flipped = np.flip(values, axis=0)  # Flip back
    return values_flipped.flatten()


def save(
    coordinates_x: np.ndarray,
    coordinates_y: np.ndarray,
    data_dict: dict,
    path: Path,
    header: str = "x [m],y [m],saturation,mass [kg]",
) -> None:
    """
    Save FluidFlower data to CSV file.

    Parameters
    ----------
    coordinates_x : np.ndarray
        1D array of x coordinates
    coordinates_y : np.ndarray
        1D array of y coordinates
    data_dict : dict
        Dictionary with data columns (keys are column names, values are 1D arrays)
    path : Path
        Output file path
    header : str
        CSV header line
    """
    # Assemble the data in the original format, with x and y coordinates as the first two columns,
    # followed by other data columns in order of the dict
    columns = [coordinates_x, coordinates_y] + list(data_dict.values())

    data = np.column_stack(columns)

    # Write to CSV
    np.savetxt(
        path,
        data,
        delimiter=",",
        header=header,
        comments="",
    )
