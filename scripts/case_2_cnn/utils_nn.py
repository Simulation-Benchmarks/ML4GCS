"""Neural-network persistence helpers."""

from pdb import set_trace as st

import pickle
from pathlib import Path

import jax


def save_params(params, path: str | Path = "results/params.pkl") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(jax.device_get(params), f)


def load_params(path: str | Path = "results/params.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)
