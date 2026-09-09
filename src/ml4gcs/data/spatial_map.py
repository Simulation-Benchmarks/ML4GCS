"""Load and represent SPE11B spatial-map snapshots."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import re
from pathlib import Path
from typing import Callable

import numpy as np


SPE11B_SPATIAL_MAP_TIME_RE = re.compile(
    r"^spe11b_spatial_map_(?P<time>-?\d+(?:\.\d+)?)y\.csv$"
)
FLUIDFLOWER_SPATIAL_MAP_TIME_RE = re.compile(
    r"^ml4gcs_spatial_map_(?P<hours>\d+)_(?P<minutes>\d{2})\.csv$"
)


@dataclass(frozen=True, slots=True)
class SpatialMapSnapshot:
    """One spatial-map file for one participant at one time."""

    participant: str
    time_years: float
    path: Path
    columns: tuple[str, ...]
    data: np.ndarray

    @property
    def x(self) -> np.ndarray:
        return self.data[:, 0]

    @property
    def z(self) -> np.ndarray:
        return self.data[:, 1]

    @property
    def field_names(self) -> tuple[str, ...]:
        return self.columns[2:]

    @property
    def field_values(self) -> np.ndarray:
        return self.data[:, 2:]

    @property
    def unique_x(self) -> np.ndarray:
        return np.unique(self.x)

    @property
    def unique_z(self) -> np.ndarray:
        return np.unique(self.z)

    @property
    def grid_shape(self) -> tuple[int, int]:
        return (self.unique_z.size, self.unique_x.size)

    def reshape_field(self, field_name: str) -> np.ndarray:
        """Reshape a scalar field onto the 2D x-z grid."""

        try:
            field_index = self.columns.index(field_name)
        except ValueError as exc:
            raise KeyError(f"Unknown field name: {field_name}") from exc
        if field_index < 2:
            raise ValueError(f"{field_name} is a coordinate column, not a field.")

        values = self.data[:, field_index]
        z_values = self.unique_z
        x_values = self.unique_x
        grid = np.empty((z_values.size, x_values.size), dtype=values.dtype)

        x_to_index = {float(x): i for i, x in enumerate(x_values)}
        z_to_index = {float(z): i for i, z in enumerate(z_values)}

        for x, z, value in zip(self.x, self.z, values, strict=True):
            grid[z_to_index[float(z)], x_to_index[float(x)]] = value
        return grid


def parse_spatial_map_time(
    path: Path | str,
    *,
    filename_regex: re.Pattern[str] = SPE11B_SPATIAL_MAP_TIME_RE,
    time_group: str = "time",
) -> float:
    """Extract the time encoded in a spatial-map filename using a regex pattern."""

    filename = Path(path).name
    match = filename_regex.match(filename)
    if not match:
        raise ValueError(f"Not a spatial-map filename: {filename}")
    if time_group in match.groupdict():
        return float(match.group(time_group))
    if len(match.groups()) == 1:
        return float(match.group(1))
    raise ValueError(
        f"Could not find time group '{time_group}' in filename pattern for {filename}"
    )


def parse_fluidflower_spatial_map_time(path: Path | str) -> float:
    """Extract FluidFlower snapshot time in hours from ``ml4gcs_spatial_map_HH_MM.csv``."""

    filename = Path(path).name
    match = FLUIDFLOWER_SPATIAL_MAP_TIME_RE.match(filename)
    if not match:
        raise ValueError(f"Not a FluidFlower spatial-map filename: {filename}")
    hours = int(match.group("hours"))
    minutes = int(match.group("minutes"))
    return float(hours) + float(minutes) / 60.0


def _read_header_and_rows(path: Path) -> tuple[tuple[str, ...], np.ndarray]:
    with path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        header_row = next(reader)
        if not header_row:
            raise ValueError(f"Missing header in {path}")
        header = tuple(item.strip().lstrip("#").strip() for item in header_row)
        rows = [[float(value) for value in row] for row in reader if row]
    return header, np.asarray(rows, dtype=np.float64)


def load_spatial_map_csv(
    path: Path | str,
    *,
    time_parser: Callable[[Path | str], float] = parse_spatial_map_time,
) -> SpatialMapSnapshot:
    """Load one spatial-map CSV file."""

    csv_path = Path(path)
    header, data = _read_header_and_rows(csv_path)
    participant = csv_path.parent.name
    time_years = time_parser(csv_path)
    return SpatialMapSnapshot(
        participant=participant,
        time_years=time_years,
        path=csv_path,
        columns=header,
        data=data,
    )
