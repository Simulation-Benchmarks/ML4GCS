"""SPE11B data management (see scripts/pipeline_overview.md).

g = ((S_M × S_M) × S_d) ∘ φ ∘ (Λ × Λ) ∘ π:

- `pi`: π, training index -> pair of unrolled indices.
- `Lambda`, `Lambda_inv`: Λ, Λ⁻¹, unrolled index <-> label (a, b).
- `phi`: φ, label pair -> training sample.
- `S`, `S_inv`: standardization and its inverse.
- `Standardization`: S_M, S_d, i.e. S with fixed μ, σ.
- `Dataset.__getitem__`: g.

π, Λ, Λ⁻¹, φ, S_M, S_d and g accept NumPy, JAX and PyTorch arrays.

`Minibatches`: the Minibatching section, for JAX; PyTorch users use DataLoader.

The HDF5 file is written by preprocess.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import h5py
import numpy as np

try:
    import jax
except ImportError:  # PyTorch users without JAX
    jax = None

REPO_ROOT = Path(__file__).resolve().parents[2]
H5_PATH = REPO_ROOT / "data" / "spe11b.h5"

EPS = 1e-8  # ε


# Indexing


def Lambda(k, n_participants):
    """Λ: unrolled index k -> label (a, b)."""
    return k % n_participants, k // n_participants


def Lambda_inv(a, b, n_participants):
    """Λ⁻¹: label (a, b) -> unrolled index k."""
    return n_participants * b + a


def pi(pairs, i):
    """π: training index i -> (k_i, k̃_i), row i of pairs."""
    return pairs[i, 0], pairs[i, 1]


def phi(maps, distances, lam, lam_tilde, n_participants):
    """φ: label pair (λ, λ̃) -> ((M_λ, M_λ̃), d(M_λ, M_λ̃))."""
    k = Lambda_inv(*lam, n_participants)
    k_tilde = Lambda_inv(*lam_tilde, n_participants)
    return (maps[k], maps[k_tilde]), distances[k, k_tilde]


# Scaling


def S(x, mu, sigma):
    """Standardization: S(x) = (x - μ) / (σ + ε)."""
    return (x - mu) / (sigma + EPS)


def S_inv(y, mu, sigma):
    """S⁻¹(y) = (σ + ε) y + μ."""
    return (sigma + EPS) * y + mu


class Standardization(NamedTuple):
    """S with fixed μ, σ: S_M or S_d."""

    mu: float  # μ
    sigma: float  # σ

    def __call__(self, x):
        return S(x, self.mu, self.sigma)

    def inv(self, y):
        return S_inv(y, self.mu, self.sigma)


# Datasets


@dataclass(frozen=True, eq=False)
class Dataset:
    """One split: shared maps and distances, its own π, the scalings of D_train."""

    maps: Any  # (K, 120, 840), M in kg
    distances: Any  # (K, K), d(M, M̃) in kg·m
    pairs: Any  # (N, 2), row i = π(i)
    S_M: Standardization
    S_d: Standardization
    n_participants: int  # |P|

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        """g(i). i: one index, or an array of indices (a minibatch)."""
        k, k_tilde = pi(self.pairs, i)
        lam = Lambda(k, self.n_participants)
        lam_tilde = Lambda(k_tilde, self.n_participants)
        (M, M_tilde), d = phi(
            self.maps, self.distances, lam, lam_tilde, self.n_participants
        )
        return (self.S_M(M), self.S_M(M_tilde)), self.S_d(d)


if jax is not None:  # a Dataset can be an argument of jitted functions
    jax.tree_util.register_dataclass(
        Dataset,
        data_fields=["maps", "distances", "pairs", "S_M", "S_d"],
        meta_fields=["n_participants"],
    )


def split(n_maps, seed=0, ratios=(0.7, 0.15, 0.15)):
    """Random partition of the datum indices q.

    q = K k + k̃ identifies the ordered pair of maps (k, k̃).
    Returns the pairs of D_train, D_val, D_test: arrays of shape (N_split, 2).
    """
    n_data = n_maps**2  # |D|
    q = np.random.default_rng(seed).permutation(n_data)
    ends = np.round(np.cumsum(ratios)[:-1] * n_data).astype(int)
    return tuple(
        np.stack(np.divmod(part, n_maps), axis=1).astype(np.int32)
        for part in np.split(q, ends)
    )


def load(h5_path=H5_PATH):
    """maps, distances, participants, years of the HDF5 file, as NumPy arrays."""
    with h5py.File(h5_path, "r") as f:
        return {
            "maps": f["maps"][:],
            "distances": f["distances"][:],
            "participants": f["participants"].asstr()[:],
            "years": f["years"][:],
        }


def make_datasets(h5_path=H5_PATH, seed=0, asarray=np.asarray):
    """D_train, D_val, D_test.

    asarray moves an array to the framework and device, e.g. jax.numpy.asarray.
    """
    arrays = load(h5_path)
    pairs = split(len(arrays["maps"]), seed)
    # S_M from all entries of all maps; S_d from the distances in D_train.
    d_train = arrays["distances"][pairs[0][:, 0], pairs[0][:, 1]]
    S_M = Standardization(
        float(arrays["maps"].mean(dtype=np.float64)),
        float(arrays["maps"].std(dtype=np.float64)),
    )
    S_d = Standardization(
        float(d_train.mean(dtype=np.float64)), float(d_train.std(dtype=np.float64))
    )
    maps, distances = asarray(arrays["maps"]), asarray(arrays["distances"])
    n_participants = len(arrays["participants"])
    return tuple(
        Dataset(maps, distances, asarray(p), S_M, S_d, n_participants) for p in pairs
    )


# Minibatching


class Minibatches:
    """Minibatches of one split, from its index sequence I = (0, ..., N-1).

    Iterating gives one epoch: consecutive slices of I; the last may be smaller.
    """

    def __init__(self, n_samples, batch_size, seed=0):
        self.I = np.arange(n_samples, dtype=np.int32)
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

    def reshuffle(self):
        """I <- reshuffle(I): the entries of I in a uniformly random order."""
        self.I = self.rng.permutation(self.I)

    def __iter__(self):
        for start in range(0, len(self.I), self.batch_size):
            yield self.I[start : start + self.batch_size]
