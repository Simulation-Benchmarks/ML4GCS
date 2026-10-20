from pdb import set_trace as st
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from spe11_wasserstein import data

import model
import utils_nn


@jax.jit
def predict(params, dataset, minibatch):
    """Prediction and target in kg·m: S_d⁻¹(f_θ(x_i)), S_d⁻¹(y_i)."""
    (M, M_tilde), y = dataset[minibatch]
    y_pred = jnp.squeeze(
        model.forward(params, jnp.stack([M, M_tilde], axis=1)), axis=-1
    )
    return dataset.S_d.inv(y_pred), dataset.S_d.inv(y)


def compute_test_predictions(params, test_dataset, batch_size=256):
    y_pred, y_test = [], []
    for minibatch in data.Minibatches(len(test_dataset), batch_size):
        prediction, target = predict(params, test_dataset, minibatch)
        y_pred.append(prediction)
        y_test.append(target)
    y_pred = np.concatenate(y_pred).astype(np.float64)
    y_test = np.concatenate(y_test).astype(np.float64)

    return y_test, y_pred


def compute_test_metrics(y_test, y_pred):
    discrepancy = y_pred - y_test

    mse = float(np.mean(discrepancy**2) / np.mean(y_test**2))
    rmse = float(np.sqrt(np.sum(discrepancy**2)) / np.sqrt(np.sum(y_test**2)))
    mae = float(np.mean(np.abs(discrepancy)) / np.mean(y_test))
    r2 = float(1 - np.sum(discrepancy**2) / np.sum((y_test - np.mean(y_test)) ** 2))

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def main():
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "results"
    params_path = results_dir / "params.pkl"

    params = jax.device_put(utils_nn.load_params(params_path))
    _, _, test_dataset = data.make_datasets(asarray=jnp.asarray)

    y_test, y_pred = compute_test_predictions(params, test_dataset)
    metrics = compute_test_metrics(y_test, y_pred)

    report = (
        "Test metrics (unscaled distances)\n"
        f"MSE: {metrics['mse']:.6e}\n"
        f"RMSE: {metrics['rmse']:.6e}\n"
        f"MAE: {metrics['mae']:.6e}\n"
        f"R2: {metrics['r2']:.6f}\n"
    )
    print("\n" + report, end="")
    (results_dir / "metrics.txt").write_text(report)
    np.savetxt(
        results_dir / "test_predictions.txt",
        np.column_stack([y_test, y_pred]),
        header="target prediction [kg m]",
    )


if __name__ == "__main__":
    main()
