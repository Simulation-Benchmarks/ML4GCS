"""Discover SPE11B data files on disk."""

from __future__ import annotations

from pathlib import Path


def find_spe11b_data_root(start: Path | str = "spe11b") -> Path:
    """Return the directory that directly contains the participant folders.

    The downloaded archive in this repo currently ends up as ``spe11b/spe11b``.
    This helper resolves that automatically so notebooks and scripts can use a
    single root regardless of where the data was unpacked.
    """

    root = Path(start)
    if not root.exists():
        raise FileNotFoundError(f"Data root does not exist: {root}")

    nested = root / "spe11b"
    if nested.exists() and nested.is_dir():
        return nested

    if any(path.is_dir() for path in root.iterdir()):
        return root

    raise FileNotFoundError(
        f"Could not find a SPE11B data root under {root}. Expected participant folders."
    )


def discover_spatial_map_paths(start: Path | str = "spe11b") -> list[Path]:
    """Return all spatial-map CSV files under the SPE11B data root."""

    root = find_spe11b_data_root(start)
    return sorted(root.glob("*/spe11b_spatial_map_*.csv"))
