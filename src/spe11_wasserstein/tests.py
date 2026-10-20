"""Tests of data.py, discovery.py and preprocess.py.

From the repository root: PYTHONPATH=src python3 -m spe11_wasserstein.tests
"""

import tempfile
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import torch

from spe11_wasserstein import data, discovery, preprocess


def random_dataset(n_participants=3, n_years=2, seed=0):
    """D_train of small random maps and distances, from make_datasets."""
    rng = np.random.default_rng(seed)
    K = n_participants * n_years
    with tempfile.TemporaryDirectory() as tmp:
        h5_path = Path(tmp) / "spe11b.h5"
        with h5py.File(h5_path, "w") as f:
            f["maps"] = rng.random((K, 2, 3), dtype=np.float32)
            f["distances"] = rng.random((K, K), dtype=np.float32)
            f.create_dataset(
                "participants",
                data=[f"p{a}" for a in range(n_participants)],
                dtype=h5py.string_dtype(),
            )
            f["years"] = 50 * np.arange(1, n_years + 1)
        return data.make_datasets(h5_path, seed)[0]


def test_Lambda():
    assert data.Lambda(0, 34) == (0, 0)
    assert data.Lambda(1, 34) == (1, 0)
    k = np.arange(680)
    np.testing.assert_array_equal(data.Lambda_inv(*data.Lambda(k, 34), 34), k)


def test_split():
    K = 680
    parts = data.split(K)
    assert [len(part) for part in parts] == [323680, 69360, 69360]
    q = np.concatenate(parts) @ np.array([K, 1])
    np.testing.assert_array_equal(np.sort(q), np.arange(K**2))


def test_standardization():
    S = data.Standardization(mu=3.0, sigma=2.0)
    x = np.array([-1.0, 3.0, 5.0])
    np.testing.assert_allclose(S(x), [-2.0, 0.0, 1.0])
    np.testing.assert_allclose(S.inv(S(x)), x)


def test_scalings():
    """S_M from all entries of all maps; S_d from the distances in D_train."""
    dataset = random_dataset()
    d_train = dataset.distances[dataset.pairs[:, 0], dataset.pairs[:, 1]]
    np.testing.assert_allclose(
        dataset.S_M, (dataset.maps.mean(), dataset.maps.std()), rtol=1e-6
    )
    np.testing.assert_allclose(dataset.S_d, (d_train.mean(), d_train.std()), rtol=1e-6)


def test_g():
    dataset = random_dataset()
    k, k_tilde = dataset.pairs[:, 0], dataset.pairs[:, 1]
    (M, M_tilde), y = dataset[np.arange(len(dataset))]
    np.testing.assert_allclose(M, dataset.S_M(dataset.maps[k]))
    np.testing.assert_allclose(M_tilde, dataset.S_M(dataset.maps[k_tilde]))
    np.testing.assert_allclose(y, dataset.S_d(dataset.distances[k, k_tilde]))

    (M_1, M_tilde_1), y_1 = dataset[1]
    np.testing.assert_allclose(M_1, M[1])
    np.testing.assert_allclose(M_tilde_1, M_tilde[1])
    np.testing.assert_allclose(y_1, y[1])


def test_g_jax_jit():
    dataset = random_dataset()
    i = np.array([2, 0, 5])
    (M, M_tilde), y = dataset[i]
    (M_jax, M_tilde_jax), y_jax = jax.jit(lambda dataset, i: dataset[i])(
        jax.tree_util.tree_map(jnp.asarray, dataset), i
    )
    np.testing.assert_allclose(M_jax, M, rtol=1e-6)
    np.testing.assert_allclose(M_tilde_jax, M_tilde, rtol=1e-6)
    np.testing.assert_allclose(y_jax, y, rtol=1e-6)


def test_g_torch_dataloader():
    dataset = random_dataset()
    dataset_torch = data.Dataset(
        torch.as_tensor(dataset.maps),
        torch.as_tensor(dataset.distances),
        torch.as_tensor(dataset.pairs),
        dataset.S_M,
        dataset.S_d,
        dataset.n_participants,
    )
    loader = torch.utils.data.DataLoader(dataset_torch, batch_size=4)
    (M, M_tilde), y = next(iter(loader))
    (M_ref, M_tilde_ref), y_ref = dataset[np.arange(4)]
    np.testing.assert_allclose(M, M_ref, rtol=1e-6)
    np.testing.assert_allclose(M_tilde, M_tilde_ref, rtol=1e-6)
    np.testing.assert_allclose(y, y_ref, rtol=1e-6)


def test_minibatches():
    """Each epoch uses every index once; reshuffle changes the order."""
    minibatches = data.Minibatches(10, 4)
    assert [len(minibatch) for minibatch in minibatches] == [4, 4, 2]
    np.testing.assert_array_equal(np.concatenate(list(minibatches)), np.arange(10))

    minibatches.reshuffle()
    I_1 = np.concatenate(list(minibatches))
    minibatches.reshuffle()
    I_2 = np.concatenate(list(minibatches))
    np.testing.assert_array_equal(np.sort(I_1), np.arange(10))
    assert not np.array_equal(I_1, I_2)


def test_find_spe11b_data_root():
    """spe11b/<p>/, then also the nested archive spe11b/spe11b/<p>/."""
    with tempfile.TemporaryDirectory() as tmp:
        spe11b_dir = Path(tmp) / "spe11b"
        for root in (spe11b_dir, spe11b_dir / "spe11b"):
            (root / "calgary1").mkdir(parents=True)
            (root / "calgary1" / "spe11b_spatial_map_50y.csv").touch()
            assert discovery.find_spe11b_data_root(spe11b_dir) == root


def test_preprocess():
    """Fake spe11b/ with 2 participants and 2 years: k = |P| b + a."""
    participants, years = ["calgary1", "opm1"], [50, 100]
    n_cells = preprocess.MAP_SHAPE[0] * preprocess.MAP_SHAPE[1]
    rng = np.random.default_rng(0)

    with tempfile.TemporaryDirectory() as tmp:
        spe11b_dir = Path(tmp) / "spe11b"
        expected_maps = np.empty((4, n_cells), dtype=np.float32)
        expected_distances = np.empty((4, 4), dtype=np.float32)
        rows = []
        for a, p in enumerate(participants):
            (spe11b_dir / p).mkdir(parents=True)
            for b, t in enumerate(years):
                M = 1e6 * (10 * a + b) + np.arange(n_cells, dtype=np.float64)
                M[7] = np.nan
                np.savetxt(
                    spe11b_dir / p / f"spe11b_spatial_map_{t}y.csv",
                    np.column_stack([np.zeros(n_cells), M]),
                    delimiter=", ",
                    header="# x [m], tmCO2 [kg]",
                    comments="",
                )
                expected_maps[2 * b + a] = np.nan_to_num(M)
                for a_tilde, p_tilde in enumerate(participants):
                    for b_tilde, t_tilde in enumerate(years):
                        d = 100 * (10 * a + b) + 10 * a_tilde + b_tilde
                        rows.append((f"{p}_{t}y", f"{p_tilde}_{t_tilde}y", d))
                        expected_distances[2 * b + a, 2 * b_tilde + a_tilde] = d
        rows = [rows[j] for j in rng.permutation(len(rows))]
        pd.DataFrame(rows, columns=["key1", "key2", "distance"]).to_csv(
            spe11b_dir / "assembled_sorted_full.csv", index=False
        )

        h5_path = Path(tmp) / "data" / "spe11b.h5"
        preprocess.preprocess(spe11b_dir, h5_path)
        arrays = data.load(h5_path)

    np.testing.assert_array_equal(arrays["maps"].reshape(4, n_cells), expected_maps)
    np.testing.assert_array_equal(arrays["distances"], expected_distances)
    assert list(arrays["participants"]) == participants
    assert list(arrays["years"]) == years


def run_tests():
    tests = (
        test_Lambda,
        test_split,
        test_standardization,
        test_scalings,
        test_g,
        test_g_jax_jit,
        test_g_torch_dataloader,
        test_minibatches,
        test_find_spe11b_data_root,
        test_preprocess,
    )

    for test in tests:
        try:
            test()
        except Exception as error:
            print(f"{test.__name__}: test FAILED ({error})\n")
        else:
            print(f"{test.__name__}: test PASSED\n")


if __name__ == "__main__":
    run_tests()
