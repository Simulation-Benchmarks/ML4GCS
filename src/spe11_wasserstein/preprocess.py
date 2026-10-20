"""SPE11B preprocessing: spe11b/ -> HDF5 file (see scripts/pipeline_overview.md).

Run once from the repository root:
    PYTHONPATH=src python3 -m spe11_wasserstein.preprocess
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from spe11_wasserstein.data import H5_PATH, REPO_ROOT, Lambda_inv
from spe11_wasserstein.discovery import find_spe11b_data_root

SPE11B_DIR = REPO_ROOT / "spe11b"

MAP_SHAPE = (120, 840)


def preprocess(spe11b_dir=SPE11B_DIR, h5_path=H5_PATH):
    """spe11b/ -> HDF5 file."""
    spe11b_dir = Path(spe11b_dir)
    table = pd.read_csv(
        spe11b_dir / "assembled_sorted_full.csv", usecols=["key1", "key2", "distance"]
    )

    # P and T from the keys "<p>_<t>y".
    keys = [key.rsplit("_", 1) for key in table["key1"].unique()]
    participants = sorted({p for p, _ in keys})
    years = sorted({int(t.removesuffix("y")) for _, t in keys})
    n_participants = len(participants)
    K = n_participants * len(years)

    k_of_key = {
        f"{p}_{t}y": Lambda_inv(a, b, n_participants)
        for a, p in enumerate(participants)
        for b, t in enumerate(years)
    }
    distances = np.full((K, K), np.nan, dtype=np.float32)
    distances[
        table["key1"].map(k_of_key).to_numpy(), table["key2"].map(k_of_key).to_numpy()
    ] = table["distance"].to_numpy()
    if np.isnan(distances).any():
        raise ValueError("assembled_sorted_full.csv lacks some pairs of maps.")

    root = find_spe11b_data_root(spe11b_dir)
    maps = np.empty((K, *MAP_SHAPE), dtype=np.float32)
    for b, t in enumerate(years):
        print(f"Reading the maps of year {t}")
        for a, p in enumerate(participants):
            M = pd.read_csv(
                root / p / f"spe11b_spatial_map_{t}y.csv",
                skipinitialspace=True,
                usecols=["tmCO2 [kg]"],
            )["tmCO2 [kg]"]
            # NaN cells (ctc-cne1, ut-csee2) -> 0 kg
            maps[Lambda_inv(a, b, n_participants)] = (
                M.fillna(0.0).to_numpy().reshape(MAP_SHAPE)
            )

    h5_path = Path(h5_path)
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "w") as f:
        f["maps"] = maps
        f["distances"] = distances
        f.create_dataset("participants", data=participants, dtype=h5py.string_dtype())
        f["years"] = years
    print(f"Saved {h5_path}")


if __name__ == "__main__":
    preprocess()
