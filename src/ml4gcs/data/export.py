"""Export SPE11B pressure predictions back to CSV files."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from .series import ParticipantSeries
from .spatial_map import SpatialMapSnapshot


def _format_row(values: np.ndarray) -> str:
    return ", ".join(f"{float(value):.6e}" for value in values)


def save_spatial_map_csv(
    template: SpatialMapSnapshot,
    output_path: Path | str,
    pressure_grid: np.ndarray | None = None,
) -> Path:
    """Write a spatial-map CSV using a template snapshot.

    If ``pressure_grid`` is provided, it replaces the pressure column while the
    remaining columns are copied from the template.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = template.data.copy()
    if pressure_grid is not None:
        pressure_index = template.columns.index("pressure [Pa]")
        if pressure_grid.shape != template.grid_shape:
            raise ValueError(
                "Pressure grid shape does not match the template grid shape: "
                f"{pressure_grid.shape} vs {template.grid_shape}"
            )
        data[:, pressure_index] = np.asarray(pressure_grid, dtype=data.dtype).reshape(-1)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("# " + ", ".join(template.columns) + "\n")
        for row in data:
            handle.write(_format_row(row) + "\n")

    return output_path


def save_spatial_map_csv_data(
    template: SpatialMapSnapshot,
    output_path: Path | str,
    data: np.ndarray | None = None,
) -> Path:
    """Write a spatial-map CSV using a full row matrix."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if data is None:
        data = template.data.copy()
    else:
        data = np.asarray(data, dtype=template.data.dtype)
        if data.shape != template.data.shape:
            raise ValueError(
                "Predicted data shape does not match the template shape: "
                f"{data.shape} vs {template.data.shape}"
            )

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("# " + ", ".join(template.columns) + "\n")
        for row in data:
            handle.write(_format_row(row) + "\n")

    return output_path


def export_participant_timeline(
    participant_series: ParticipantSeries,
    predict_pressure: Callable[[SpatialMapSnapshot], np.ndarray],
    output_root: Path | str,
) -> tuple[Path, ...]:
    """Export a full participant timeline as CSV files.

    ``predict_pressure`` is called once per snapshot and must return a pressure
    grid with the same shape as the template snapshot.
    """

    output_root = Path(output_root) / participant_series.participant
    output_paths: list[Path] = []

    for snapshot in participant_series.snapshots:
        pressure_grid = predict_pressure(snapshot)
        output_path = output_root / snapshot.path.name
        output_paths.append(save_spatial_map_csv(snapshot, output_path, pressure_grid))

    return tuple(output_paths)


def export_next_step_timeline(
    participant_series: ParticipantSeries,
    predict_pressure: Callable[[SpatialMapSnapshot, SpatialMapSnapshot], np.ndarray],
    output_root: Path | str,
) -> tuple[Path, ...]:
    """Export next-step forecasts for a participant timeline.

    The predictor receives an input snapshot and the corresponding target
    snapshot template, which is useful for time-conditioned forecasting.
    """

    output_root = Path(output_root) / participant_series.participant
    output_paths: list[Path] = []
    snapshots = participant_series.snapshots

    if snapshots:
        first_snapshot = snapshots[0]
        output_paths.append(
            save_spatial_map_csv_data(first_snapshot, output_root / first_snapshot.path.name)
        )

    for input_snapshot, target_snapshot in zip(snapshots[:-1], snapshots[1:], strict=True):
        pressure_grid = predict_pressure(input_snapshot, target_snapshot)
        output_path = output_root / target_snapshot.path.name
        output_paths.append(save_spatial_map_csv(target_snapshot, output_path, pressure_grid))

    return tuple(output_paths)


def export_next_step_timeline_data(
    participant_series: ParticipantSeries,
    predict_data: Callable[[SpatialMapSnapshot, SpatialMapSnapshot], np.ndarray],
    output_root: Path | str,
) -> tuple[Path, ...]:
    """Export next-step forecasts using full row matrices."""

    output_root = Path(output_root) / participant_series.participant
    output_paths: list[Path] = []
    snapshots = participant_series.snapshots

    if snapshots:
        first_snapshot = snapshots[0]
        output_paths.append(save_spatial_map_csv_data(first_snapshot, output_root / first_snapshot.path.name))

    for input_snapshot, target_snapshot in zip(snapshots[:-1], snapshots[1:], strict=True):
        predicted = predict_data(input_snapshot, target_snapshot)
        output_path = output_root / target_snapshot.path.name
        output_paths.append(save_spatial_map_csv_data(target_snapshot, output_path, predicted))

    return tuple(output_paths)
