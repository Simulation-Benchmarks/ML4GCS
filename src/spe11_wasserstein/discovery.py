"""Discovery of the SPE11B data on disk (see scripts/pipeline_overview.md)."""

from pathlib import Path


def find_spe11b_data_root(start="spe11b"):
    """Folder that directly contains the participant folders.

    start/spe11b/ (nested archive) if it contains them, else start/.
    A participant folder contains spe11b_spatial_map_*.csv files.
    """
    start = Path(start)
    for root in (start / "spe11b", start):
        if any(root.glob("*/spe11b_spatial_map_*.csv")):
            return root
    raise FileNotFoundError(f"No participant folders in {start / 'spe11b'} or {start}.")
