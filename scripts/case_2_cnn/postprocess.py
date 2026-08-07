from pdb import set_trace as st
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"text.usetex": True, "font.family": "serif"})

import model
import utils_nn


def compute_test_metrics(params, x_test, y_test):
    y_pred = np.asarray(model.forward(params, x_test)).squeeze(axis=-1)
    y_test = np.asarray(y_test, dtype=y_pred.dtype)

    discrepancy = y_pred - y_test

    mse = float(np.mean(discrepancy**2) / np.mean(y_test**2))
    rmse = float(np.sqrt(np.sum(discrepancy**2)) / np.sqrt(np.sum(y_test**2)))
    mae = float(np.mean(np.abs(discrepancy)) / np.mean(y_test))

    ss_res = float(np.sum(discrepancy**2))
    ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0.0 else float(ss_res == 0.0)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }, y_pred, y_test


def plot_r_squared(y_pred, y_test, r2, output_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, s=10, alpha=0.5)
    lo = float(min(y_test.min(), y_pred.min()))
    hi = float(max(y_test.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    ax.set_xlabel("Target")
    ax.set_ylabel("Prediction")
    ax.set_title(f"$R^2$ = {r2:.4f}")
    fig.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)


def main():
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "results"
    params_path = results_dir / "params.pkl"
    test_dataset_path = results_dir / "test_dataset.npz"

    params = utils_nn.load_params(params_path)
    with np.load(test_dataset_path) as data:
        x_test = data["x_test"]
        y_test = data["y_test"]

    metrics, y_pred, y_test = compute_test_metrics(params, x_test, y_test)

    print("\nTest metrics")
    print(f"MSE: {metrics['mse']:.6e}")
    print(f"RMSE: {metrics['rmse']:.6e}")
    print(f"MAE: {metrics['mae']:.6e}")
    print(f"R2: {metrics['r2']:.6f}")

    plot_r_squared(y_pred, y_test, metrics["r2"], results_dir / "r_squared.pdf")


if __name__ == "__main__":
    main()
