"""Lightweight catalogues for SPE11B spatial-map files."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Sequence

from .discovery import discover_spatial_map_paths
from .spatial_map import SpatialMapSnapshot, load_spatial_map_csv, parse_spatial_map_time


@dataclass(frozen=True, slots=True)
class SpatialMapRef:
    """A lightweight reference to one spatial-map CSV file."""

    participant: str
    time_years: float
    path: Path

    def load(self) -> SpatialMapSnapshot:
        """Load the CSV file into a full in-memory snapshot."""

        return load_spatial_map_csv(self.path)


def build_spatial_map_index(start: Path | str = "spe11b") -> tuple[SpatialMapRef, ...]:
    """Return a lightweight catalog of all spatial-map files."""

    return build_spatial_map_index_filtered(start)


def build_spatial_map_index_limited(
    start: Path | str = "spe11b",
    participants: Sequence[str] | None = None,
    max_files: int | None = None,
) -> tuple[SpatialMapRef, ...]:
    """Compatibility alias for the filtered index builder."""

    return build_spatial_map_index_filtered(
        start,
        participants=participants,
        max_files=max_files,
    )


def build_spatial_map_index_filtered(
    start: Path | str = "spe11b",
    participants: Sequence[str] | None = None,
    max_files: int | None = None,
) -> tuple[SpatialMapRef, ...]:
    """Return a filtered lightweight catalog of spatial-map files."""

    participant_set = set(participants) if participants is not None else None
    refs = [
        SpatialMapRef(
            participant=path.parent.name,
            time_years=parse_spatial_map_time(path),
            path=path,
        )
        for path in discover_spatial_map_paths(start)
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
) -> dict[str, tuple[SpatialMapRef, ...]]:
    """Group the spatial-map catalog by participant."""

    grouped: dict[str, list[SpatialMapRef]] = {}
    for ref in build_spatial_map_index_filtered(
        start,
        participants=participants,
        max_files=max_files,
    ):
        grouped.setdefault(ref.participant, []).append(ref)
    return {participant: tuple(refs) for participant, refs in grouped.items()}
