from pdb import set_trace as st
from pathlib import Path

import numpy as np
import jax.numpy as jnp

import model
import utils_datasets
import utils_nn
from plot_postprocess import plot_r_squared  # also applies the shared LaTeX/font rc


def compute_test_metrics(params, test_dataset, batch_size=256):
    """Metrics over a PairDataset, gathering pairs on the GPU in batches so the
    full pair tensor is never materialized."""
    n_pairs = len(test_dataset)
    y_pred = np.concatenate(
        [
            np.asarray(
                model.forward(params, test_dataset.gather(slice(start, start + batch_size)))
            ).squeeze(axis=-1)
            for start in range(0, n_pairs, batch_size)
        ]
    )
    y_test = np.asarray(test_dataset.distances, dtype=y_pred.dtype)

    discrepancy = y_pred - y_test

    nmse = float(np.mean(discrepancy**2) / np.mean(y_test**2))
    nrmse = float(np.sqrt(np.sum(discrepancy**2)) / np.sqrt(np.sum(y_test**2)))
    nmae = float(np.mean(np.abs(discrepancy)) / np.mean(y_test))

    ss_res = float(np.sum(discrepancy**2))
    ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0.0 else float(ss_res == 0.0)

    return {
        "nmse": nmse,
        "nrmse": nrmse,
        "nmae": nmae,
        "r2": r2,
    }, y_pred, y_test


def write_examples(test_dataset, y_pred, y_test, output_path, n_examples=20):
    """Dump a few input/output/reference triples so the fit can be eyed by hand.

    The network input is a pair of 120x840 images, too large to print, so each
    example lists the image indices that identify the pair instead of the pixels.
    """
    index_image_pairs = np.asarray(test_dataset.index_image_pairs)
    n_pairs = len(y_pred)
    picks = np.linspace(0, n_pairs - 1, num=min(n_examples, n_pairs), dtype=int)
    picks = np.unique(picks)

    # The network sees and returns distances scaled onto output_scale_range;
    # report the physical W1 values, in kg m, that those stand for.
    y_test_kgm = test_dataset.unscale_distances(y_test)
    y_pred_kgm = test_dataset.unscale_distances(y_pred)

    header = (
        f"{'index_pair':>10}  {'image_a':>8}  {'image_b':>8}  "
        f"{'reference':>14}  {'prediction':>14}  "
        f"{'reference[kg m]':>16}  {'prediction[kg m]':>17}  "
        f"{'abs_err[kg m]':>15}  {'rel_error':>12}"
    )
    lines = [
        "Examples of network input (image pair) vs output vs reference target.",
        "The 'reference' and 'prediction' columns are the scaled values the",
        f"network works with (range {tuple(test_dataset.output_scale_range)}); the "
        "[kg m] columns are the",
        "same values mapped back onto the original Wasserstein-1 distances.",
        "",
        header,
        "-" * len(header),
    ]
    for i in picks:
        ref = float(y_test[i])
        pred = float(y_pred[i])
        ref_kgm = float(y_test_kgm[i])
        pred_kgm = float(y_pred_kgm[i])
        abs_err_kgm = abs(pred_kgm - ref_kgm)
        rel_err = abs_err_kgm / abs(ref_kgm) if ref_kgm != 0.0 else float("nan")
        image_a, image_b = index_image_pairs[i]
        lines.append(
            f"{int(i):>10}  {int(image_a):>8}  {int(image_b):>8}  "
            f"{ref:>14.6e}  {pred:>14.6e}  "
            f"{ref_kgm:>16.6e}  {pred_kgm:>17.6e}  "
            f"{abs_err_kgm:>15.6e}  {rel_err:>12.4f}"
        )

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved {output_path}")


def main():
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "results"
    params_path = results_dir / "params.pkl"
    test_dataset_path = results_dir / "test_dataset.npz"

    params = utils_nn.load_params(params_path)
    with np.load(test_dataset_path) as data:
        if "distance_bounds" in data:
            distance_bounds = tuple(data["distance_bounds"])
            output_scale_range = tuple(data["output_scale_range"])
        else:
            # Archive written before main.py saved the scaling. Rebuild the
            # bounds from the distance tables alone, with the same arguments
            # main.py passes to create_datasets. No retraining involved.
            print("Recomputing the training distance bounds from the CSV tables...")
            distance_bounds = utils_datasets.training_distance_bounds(
                total_number_images=6833, step=1, start=34
            )
            output_scale_range = (0.0, 1.0)

        test_dataset = utils_datasets.PairDataset(
            jnp.asarray(data["images"]),
            jnp.asarray(data["index_image_pairs_test"]),
            jnp.asarray(data["y_test"]),
            distance_bounds=distance_bounds,
            output_scale_range=output_scale_range,
        )

    print(
        f"Target scaling: [{distance_bounds[0]:.6e}, {distance_bounds[1]:.6e}] kg m "
        f"-> {tuple(float(v) for v in output_scale_range)}"
    )

    metrics, y_pred, y_test = compute_test_metrics(params, test_dataset)

    metrics_lines = [
        f"Normalized Mean Squared Error (NMSE): {metrics['nmse']:.6e}",
        f"Normalized Root Mean Squared Error (NRMSE): {metrics['nrmse']:.6e}",
        f"Normalized Mean Absolute Error (NMAE): {metrics['nmae']:.6e}",
        f"Coefficient of Determination (R2): {metrics['r2']:.6f}",
    ]

    print("\nTest metrics")
    for line in metrics_lines:
        print(line)

    with open(results_dir / "test_metrics.txt", "w") as f:
        f.write("\n".join(metrics_lines) + "\n")

    write_examples(test_dataset, y_pred, y_test, results_dir / "test_examples.txt")

    plot_r_squared(y_pred, y_test, metrics["r2"], results_dir / "r_squared.pdf")


if __name__ == "__main__":
    main()
