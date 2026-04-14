"""Group spatial-map snapshots into participant time series."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .discovery import discover_spatial_map_paths
from .spatial_map import SpatialMapSnapshot, load_spatial_map_csv


@dataclass(frozen=True, slots=True)
class ParticipantSeries:
    """All spatial-map snapshots for one participant."""

    participant: str
    snapshots: tuple[SpatialMapSnapshot, ...]

    @property
    def times_years(self) -> tuple[float, ...]:
        return tuple(snapshot.time_years for snapshot in self.snapshots)

    @property
    def paths(self) -> tuple[Path, ...]:
        return tuple(snapshot.path for snapshot in self.snapshots)

    def by_time(self, time_years: float) -> SpatialMapSnapshot:
        for snapshot in self.snapshots:
            if snapshot.time_years == time_years:
                return snapshot
        raise KeyError(f"No snapshot at time {time_years} for participant {self.participant}.")


def load_spatial_map_series(start: Path | str = "spe11b") -> tuple[ParticipantSeries, ...]:
    """Load all participant trajectories found under the SPE11B data root."""

    paths = discover_spatial_map_paths(start)
    grouped: dict[str, list[SpatialMapSnapshot]] = {}

    for path in paths:
        snapshot = load_spatial_map_csv(path)
        grouped.setdefault(snapshot.participant, []).append(snapshot)

    series = []
    for participant, snapshots in sorted(grouped.items()):
        snapshots.sort(key=lambda snapshot: snapshot.time_years)
        series.append(ParticipantSeries(participant=participant, snapshots=tuple(snapshots)))
    return tuple(series)

