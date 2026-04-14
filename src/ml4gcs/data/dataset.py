"""Transition datasets for SPE11B benchmark tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .index import SpatialMapRef, build_spatial_map_index_filtered, group_spatial_map_index
from .spatial_map import SpatialMapSnapshot


COORDINATE_COLUMNS = ("x [m]", "z [m]")


@dataclass(frozen=True, slots=True)
class SpatialMapTransition:
    """One supervised transition between two consecutive snapshots."""

    participant: str
    input_time_years: float
    target_time_years: float
    input_ref: SpatialMapRef
    target_ref: SpatialMapRef
    input_snapshot: SpatialMapSnapshot
    target_snapshot: SpatialMapSnapshot

    @property
    def input_path(self) -> Path:
        return self.input_ref.path

    @property
    def target_path(self) -> Path:
        return self.target_ref.path

    @property
    def delta_time_years(self) -> float:
        return self.target_time_years - self.input_time_years

    @property
    def field_names(self) -> tuple[str, ...]:
        return self.target_snapshot.columns


@dataclass(frozen=True, slots=True)
class SpatialMapTransitionBatch:
    """Dense row-wise matrices built from a list of transitions."""

    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    X: np.ndarray
    Y: np.ndarray
    delta_time_years: np.ndarray


class SpatialMapTransitionDataset:
    """Lazy dataset of consecutive SPE11B spatial-map transitions.

    Each item contains the next available snapshot for one participant, with
    optional filtering by participant and time window.
    """

    def __init__(
        self,
        start: Path | str = "spe11b",
        participants: Sequence[str] | None = None,
        time_years_range: tuple[float, float] | None = None,
        max_transitions_per_participant: int | None = None,
        max_transitions: int | None = None,
    ) -> None:
        self.start = Path(start)
        self.participants_filter = tuple(participants) if participants is not None else None
        self.time_years_range = time_years_range
        self.max_transitions_per_participant = max_transitions_per_participant
        self.max_transitions = max_transitions
        self._transitions = self._build_transitions()

    def _build_transitions(self) -> tuple[tuple[SpatialMapRef, SpatialMapRef], ...]:
        grouped = group_spatial_map_index(
            self.start,
            participants=self.participants_filter,
        )
        transitions: list[tuple[SpatialMapRef, SpatialMapRef]] = []

        for participant, refs in grouped.items():
            ordered = sorted(refs, key=lambda ref: ref.time_years)
            if self.time_years_range is not None:
                start_year, end_year = self.time_years_range
                ordered = [
                    ref
                    for ref in ordered
                    if start_year <= ref.time_years <= end_year
                ]
            if len(ordered) < 2:
                continue

            participant_transitions = list(zip(ordered[:-1], ordered[1:], strict=True))
            if self.max_transitions_per_participant is not None:
                participant_transitions = participant_transitions[
                    : self.max_transitions_per_participant
                ]
            transitions.extend(participant_transitions)

        transitions.sort(key=lambda pair: (pair[0].participant, pair[0].time_years, pair[1].time_years))
        if self.max_transitions is not None:
            transitions = transitions[: self.max_transitions]
        return tuple(transitions)

    def __len__(self) -> int:
        return len(self._transitions)

    def __getitem__(self, index: int) -> SpatialMapTransition:
        input_ref, target_ref = self._transitions[index]
        input_snapshot = input_ref.load()
        target_snapshot = target_ref.load()

        if input_snapshot.columns != target_snapshot.columns:
            raise ValueError(
                "Input and target snapshots do not share the same columns: "
                f"{input_snapshot.columns} vs {target_snapshot.columns}"
            )
        if input_snapshot.data.shape != target_snapshot.data.shape:
            raise ValueError(
                "Input and target snapshots do not share the same shape: "
                f"{input_snapshot.data.shape} vs {target_snapshot.data.shape}"
            )

        return SpatialMapTransition(
            participant=input_ref.participant,
            input_time_years=input_ref.time_years,
            target_time_years=target_ref.time_years,
            input_ref=input_ref,
            target_ref=target_ref,
            input_snapshot=input_snapshot,
            target_snapshot=target_snapshot,
        )

    @property
    def participants(self) -> tuple[str, ...]:
        return tuple(sorted({input_ref.participant for input_ref, _ in self._transitions}))

    @property
    def time_pairs(self) -> tuple[tuple[float, float], ...]:
        return tuple(
            (input_ref.time_years, target_ref.time_years)
            for input_ref, target_ref in self._transitions
        )


def build_spatial_map_transition_batch(
    transitions: Sequence[SpatialMapTransition],
    rows_per_transition: int | None = None,
    seed: int = 0,
) -> SpatialMapTransitionBatch:
    """Convert a sequence of transitions into dense training matrices."""

    if not transitions:
        raise ValueError("Cannot build a training batch from an empty transition sequence.")

    first_snapshot = transitions[0].input_snapshot
    state_columns = tuple(
        name for name in first_snapshot.columns if name not in COORDINATE_COLUMNS
    )
    state_indices = tuple(first_snapshot.columns.index(name) for name in state_columns)

    rng = np.random.default_rng(seed)
    x_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    delta_chunks: list[np.ndarray] = []

    for transition in transitions:
        input_data = transition.input_snapshot.data
        target_data = transition.target_snapshot.data
        if input_data.shape != target_data.shape:
            raise ValueError(
                "Input and target transition rows have different shapes: "
                f"{input_data.shape} vs {target_data.shape}"
            )

        n_rows = input_data.shape[0]
        if rows_per_transition is not None and rows_per_transition < n_rows:
            keep = rng.choice(n_rows, size=rows_per_transition, replace=False)
            input_data = input_data[keep]
            target_data = target_data[keep]

        x_state = input_data[:, state_indices]
        y_state = target_data[:, state_indices]
        delta = np.full(
            (x_state.shape[0], 1),
            float(transition.delta_time_years),
            dtype=np.float64,
        )

        x_chunks.append(np.hstack([x_state, delta]))
        y_chunks.append(y_state)
        delta_chunks.append(delta)

    X = np.vstack(x_chunks)
    Y = np.vstack(y_chunks)
    delta_time_years = np.vstack(delta_chunks)
    feature_columns = state_columns + ("delta_time_years",)
    return SpatialMapTransitionBatch(
        feature_columns=feature_columns,
        target_columns=state_columns,
        X=X,
        Y=Y,
        delta_time_years=delta_time_years,
    )


# Backwards-compatible aliases for the earlier pressure-only prototype.
PRESSURE_FIELD_NAME = "pressure [Pa]"
PressureTransition = SpatialMapTransition
PressureTransitionDataset = SpatialMapTransitionDataset
