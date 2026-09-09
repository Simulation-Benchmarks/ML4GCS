"""Discover SPE11B data files on disk."""

from __future__ import annotations

from pathlib import Path


def find_data_root(
    start: Path | str,
    *,
    nested_dir_name: str | None = None,
) -> Path:
    """Return the directory that directly contains participant folders."""

    root = Path(start)
    if not root.exists():
        raise FileNotFoundError(f"Data root does not exist: {root}")

    if nested_dir_name is not None:
        nested = root / nested_dir_name
        if nested.exists() and nested.is_dir():
            return nested

    if any(path.is_dir() for path in root.iterdir()):
        return root

    raise FileNotFoundError(
        f"Could not find a data root under {root}. Expected participant folders."
    )


def find_spe11b_data_root(start: Path | str = "spe11b") -> Path:
    """Return the directory that directly contains the participant folders.

    The downloaded archive in this repo currently ends up as ``spe11b/spe11b``.
    This helper resolves that automatically so notebooks and scripts can use a
    single root regardless of where the data was unpacked.
    """

    return find_data_root(start, nested_dir_name="spe11b")


def find_fluidflower_data_root(start: Path | str = "ml4gcs_fluidflower") -> Path:
    """Return the directory that directly contains the FluidFlower participant folders."""

    return find_data_root(start, nested_dir_name=None)


def discover_spatial_map_paths(
    start: Path | str = "spe11b",
    *,
    spatial_map_glob: str = "*/spe11b_spatial_map_*.csv",
    nested_dir_name: str | None = "spe11b",
) -> list[Path]:
    """Return all spatial-map CSV files under a configurable data root."""

    root = find_data_root(start, nested_dir_name=nested_dir_name)
    return sorted(root.glob(spatial_map_glob))


def discover_fluidflower_spatial_map_paths(
    start: Path | str = "ml4gcs_fluidflower",
) -> list[Path]:
    """Return all FluidFlower spatial-map CSV files."""

    return discover_spatial_map_paths(
        start,
        spatial_map_glob="*/ml4gcs_spatial_map_*.csv",
        nested_dir_name=None,
    )
