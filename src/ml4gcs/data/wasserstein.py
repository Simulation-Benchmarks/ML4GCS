"""Shared data management for pairwise Wasserstein-distance regression."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


SPE11B_MAP_RE = re.compile(
    r"^spe11b_spatial_map_(?P<time>-?\d+(?:\.\d+)?)y\.csv$"
)


@dataclass(frozen=True, slots=True)
class MapRecord:
    """One spatial map and the metadata needed to identify its label."""

    participant: str
    time_years: float
    path: Path


@dataclass(frozen=True, slots=True)
class PairRecord:
    """One ordered input pair and its Wasserstein-distance target."""

    left: MapRecord
    right: MapRecord
    distance: float


@dataclass(frozen=True, slots=True)
class WassersteinDataset:
    """Dense map-pair inputs, labels, and serializable pair metadata."""

    X: np.ndarray
    y: np.ndarray
    pairs: tuple[PairRecord, ...]


def parse_spe11b_time(path: Path | str) -> float:
    """Parse years from a SPE11B spatial-map filename."""

    match = SPE11B_MAP_RE.match(Path(path).name)
    if match is None:
        raise ValueError(f"Not a SPE11B spatial-map filename: {Path(path).name}")
    return float(match.group("time"))


def _strip_relative_prefix(path: str) -> Path:
    value = path.strip()
    while value.startswith("./"):
        value = value[2:]
    return Path(value)


def read_map_records(
    map_list_path: Path | str,
    data_roots: Sequence[Path | str],
    *,
    filename_pattern: str = "spe11b_spatial_map_*.csv",
    strict: bool = True,
) -> tuple[MapRecord, ...]:
    """Resolve listed map files and return records in list-file order."""

    roots = tuple(Path(root) for root in data_roots)
    records: list[MapRecord] = []
    missing: list[str] = []
    for raw_line in Path(map_list_path).read_text().splitlines():
        relative = _strip_relative_prefix(raw_line)
        if not raw_line.strip() or not relative.match(filename_pattern):
            continue
        full_path = next((root / relative for root in roots if (root / relative).exists()), None)
        if full_path is None:
            missing.append(raw_line.strip())
            continue
        records.append(
            MapRecord(
                participant=relative.parts[0],
                time_years=parse_spe11b_time(relative.name),
                path=full_path,
            )
        )

    if strict and missing:
        examples = ", ".join(missing[:3])
        raise FileNotFoundError(
            f"Could not resolve {len(missing)} map files. Examples: {examples}"
        )
    if not records:
        raise ValueError(f"No map files found in {map_list_path}.")
    return tuple(records)


def load_map_values(
    records: Sequence[MapRecord],
    *,
    field_name: str = " tmCO2 [kg]",
    expected_size: int | None = 120 * 840,
) -> np.ndarray:
    """Load one numeric field from every map, validating a common size."""

    values: list[np.ndarray] = []
    columns: list[str] | None = None
    for record in records:
        with record.path.open(newline="") as handle:
            reader = csv.reader(handle)
            header = [item.strip() for item in next(reader)]
            if columns is None:
                columns = header
            elif header != columns:
                raise ValueError(f"Inconsistent columns in {record.path}")
            try:
                field_index = header.index(field_name.strip())
            except ValueError as exc:
                raise KeyError(f"Field {field_name!r} not found in {record.path}") from exc
            column = []
            for row in reader:
                if row:
                    column.append(float(row[field_index]))
        array = np.asarray(column, dtype=np.float32)
        if expected_size is not None and array.size != expected_size:
            raise ValueError(
                f"Expected {expected_size} values in {record.path}, got {array.size}"
            )
        values.append(np.nan_to_num(array, nan=0.0))
    return np.stack(values)


def _read_distance_table(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        names = [name.strip() for name in header[1:]]
        return {
            row[0].strip(): {
                name: float(value) for name, value in zip(names, row[1:], strict=False)
            }
            for row in reader
            if row
        }


class WassersteinDistanceStore:
    """Lazy cache for SPE11B dense Wasserstein distance tables."""

    def __init__(self, roots: Sequence[Path | str]) -> None:
        self.roots = tuple(Path(root) for root in roots)
        self._tables: dict[float, dict[str, dict[str, float]]] = {}

    def _path_for_time(self, time_years: float) -> Path:
        filename = f"spe11b_co2mass_w1_diff_{time_years:g}y.csv"
        for root in self.roots:
            path = root / filename
            if path.exists():
                return path
        raise FileNotFoundError(
            f"Could not find Wasserstein table {filename} in: {self.roots}"
        )

    @staticmethod
    def _candidate_names(name: str) -> Iterable[str]:
        yield name
        if name.endswith("1"):
            yield name[:-1]

    def lookup(self, left: str, right: str, time_years: float) -> float:
        if time_years not in self._tables:
            self._tables[time_years] = _read_distance_table(
                self._path_for_time(time_years)
            )
        table = self._tables[time_years]
        row_name = next((name for name in self._candidate_names(left) if name in table), None)
        if row_name is None:
            raise KeyError(f"No distance row for participant {left!r}")
        row = table[row_name]
        col_name = next((name for name in self._candidate_names(right) if name in row), None)
        if col_name is None:
            raise KeyError(f"No distance column for participant {right!r}")
        return row[col_name]


def build_pair_dataset(
    records: Sequence[MapRecord],
    map_values: np.ndarray,
    distances: WassersteinDistanceStore,
    *,
    include_self: bool = False,
    unique_pairs: bool = True,
    same_time_only: bool = True,
    grid_shape: tuple[int, int] | None = None,
) -> WassersteinDataset:
    """Build map pairs and labels with explicit duplicate/self-pair policy."""

    if len(records) != map_values.shape[0]:
        raise ValueError("records and map_values must contain the same number of maps")
    if grid_shape is not None:
        expected_size = int(np.prod(grid_shape))
        if map_values.shape[1] != expected_size:
            raise ValueError(
                f"grid_shape {grid_shape} expects {expected_size} values, "
                f"got {map_values.shape[1]}"
            )
        map_values = map_values.reshape((map_values.shape[0], *grid_shape))

    pairs: list[PairRecord] = []
    X: list[np.ndarray] = []
    for i, left in enumerate(records):
        for j, right in enumerate(records):
            if unique_pairs and j <= i:
                continue
            if not include_self and i == j:
                continue
            if same_time_only and left.time_years != right.time_years:
                continue
            distance = distances.lookup(left.participant, right.participant, left.time_years)
            pairs.append(PairRecord(left=left, right=right, distance=distance))
            X.append(np.stack([map_values[i], map_values[j]]))
    if not X:
        raise ValueError("No valid Wasserstein map pairs were generated")
    return WassersteinDataset(
        X=np.stack(X),
        y=np.asarray([pair.distance for pair in pairs], dtype=np.float32),
        pairs=tuple(pairs),
    )


def save_wasserstein_dataset(dataset: WassersteinDataset, npz_path: Path | str, metadata_path: Path | str) -> None:
    """Save dense arrays and human-readable pair metadata."""

    npz_path = Path(npz_path)
    metadata_path = Path(metadata_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, X=dataset.X, y=dataset.y)
    metadata = [
        {
            "left": {**asdict(pair.left), "path": str(pair.left.path)},
            "right": {**asdict(pair.right), "path": str(pair.right.path)},
            "distance": pair.distance,
        }
        for pair in dataset.pairs
    ]
    metadata_path.write_text(json.dumps(metadata, indent=2))


def load_wasserstein_arrays(npz_path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """Load the map-pair inputs and distance labels from a prepared dataset."""

    with np.load(npz_path) as archive:
        for key in ("X", "y"):
            if key not in archive:
                raise KeyError(f"Prepared dataset is missing array '{key}'")
        return np.asarray(archive["X"], dtype=np.float32), np.asarray(archive["y"], dtype=np.float32)
