"""Lightweight catalogues for SPE11B spatial-map files."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Callable, Sequence

from .discovery import discover_spatial_map_paths
from .spatial_map import SpatialMapSnapshot, load_spatial_map_csv, parse_spatial_map_time


@dataclass(frozen=True, slots=True)
class SpatialMapRef:
    """A lightweight reference to one spatial-map CSV file."""

    participant: str
    time_years: float
    path: Path
    time_parser: Callable[[Path | str], float] = parse_spatial_map_time

    def load(self) -> SpatialMapSnapshot:
        """Load the CSV file into a full in-memory snapshot."""

        return load_spatial_map_csv(self.path, time_parser=self.time_parser)


def build_spatial_map_index(
    start: Path | str = "spe11b",
    *,
    spatial_map_glob: str = "*/spe11b_spatial_map_*.csv",
    nested_dir_name: str | None = "spe11b",
    time_parser: Callable[[Path | str], float] = parse_spatial_map_time,
) -> tuple[SpatialMapRef, ...]:
    """Return a lightweight catalog of all spatial-map files."""

    return build_spatial_map_index_filtered(
        start,
        spatial_map_glob=spatial_map_glob,
        nested_dir_name=nested_dir_name,
        time_parser=time_parser,
    )


def build_spatial_map_index_limited(
    start: Path | str = "spe11b",
    participants: Sequence[str] | None = None,
    max_files: int | None = None,
    *,
    spatial_map_glob: str = "*/spe11b_spatial_map_*.csv",
    nested_dir_name: str | None = "spe11b",
    time_parser: Callable[[Path | str], float] = parse_spatial_map_time,
) -> tuple[SpatialMapRef, ...]:
    """Compatibility alias for the filtered index builder."""

    return build_spatial_map_index_filtered(
        start,
        participants=participants,
        max_files=max_files,
        spatial_map_glob=spatial_map_glob,
        nested_dir_name=nested_dir_name,
        time_parser=time_parser,
    )


def build_spatial_map_index_filtered(
    start: Path | str = "spe11b",
    participants: Sequence[str] | None = None,
    max_files: int | None = None,
    *,
    spatial_map_glob: str = "*/spe11b_spatial_map_*.csv",
    nested_dir_name: str | None = "spe11b",
    time_parser: Callable[[Path | str], float] = parse_spatial_map_time,
) -> tuple[SpatialMapRef, ...]:
    """Return a filtered lightweight catalog of spatial-map files."""

    participant_set = set(participants) if participants is not None else None
    refs = [
        SpatialMapRef(
            participant=path.parent.name,
            time_years=time_parser(path),
            path=path,
            time_parser=time_parser,
        )
        for path in discover_spatial_map_paths(
            start,
            spatial_map_glob=spatial_map_glob,
            nested_dir_name=nested_dir_name,
        )
        if participant_set is None or path.parent.name in participant_set
    ]
    refs.sort(key=lambda ref: (ref.participant, ref.time_years))
    if max_files is not None:
        refs = list(islice(refs, max_files))
    return tuple(refs)


def group_spatial_map_index(
    start: Path | str = "spe11b",
    participants: Sequence[str] | None = None,
    max_files: int | None = None,
    *,
    spatial_map_glob: str = "*/spe11b_spatial_map_*.csv",
    nested_dir_name: str | None = "spe11b",
    time_parser: Callable[[Path | str], float] = parse_spatial_map_time,
) -> dict[str, tuple[SpatialMapRef, ...]]:
    """Group the spatial-map catalog by participant."""

    grouped: dict[str, list[SpatialMapRef]] = {}
    for ref in build_spatial_map_index_filtered(
        start,
        participants=participants,
        max_files=max_files,
        spatial_map_glob=spatial_map_glob,
        nested_dir_name=nested_dir_name,
        time_parser=time_parser,
    ):
        grouped.setdefault(ref.participant, []).append(ref)
    return {participant: tuple(refs) for participant, refs in grouped.items()}
