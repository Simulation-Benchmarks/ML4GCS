from pdb import set_trace as st
from pathlib import Path

import numpy as np

import model
import utils_nn


def compute_test_metrics(params, x_test, y_test):
    y_pred = np.asarray(model.forward(params, x_test)).squeeze(axis=-1)
    y_test = np.asarray(y_test, dtype=y_pred.dtype)

    discrepancy = y_pred - y_test

    mse = float(np.mean(discrepancy**2) / np.mean(y_test**2))
    rmse = float(np.sqrt(np.sum(discrepancy**2)) / np.sqrt(np.sum(y_test**2)))
    mae = float(np.mean(np.abs(discrepancy)) / np.mean(y_test))

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
    }


def main():
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "results"
    params_path = results_dir / "params.pkl"
    test_dataset_path = results_dir / "test_dataset.npz"

    params = utils_nn.load_params(params_path)
    with np.load(test_dataset_path) as data:
        x_test = data["x_test"]
        y_test = data["y_test"]

    metrics = compute_test_metrics(params, x_test, y_test)

    print("\nTest metrics")
    print(f"MSE: {metrics['mse']:.6e}")
    print(f"RMSE: {metrics['rmse']:.6e}")
    print(f"MAE: {metrics['mae']:.6e}")


if __name__ == "__main__":
    main()
