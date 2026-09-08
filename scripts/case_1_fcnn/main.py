"""
Trains a fully connected network (FCNN) to predict the distance between pairs of
SPE11B CO2 mass images, as a cheap surrogate for the ground-truth pairwise distance
table used in the benchmark's dense evaluation.

Data: `utils_datasets.create_datasets` loads CO2 mass images (120x840) from the SPE11B
dataset and forms all same-year pairs (img1, img2). For each pair, x is the pair
stacked as a (2, 120, 840) array and y is the corresponding scalar distance looked up
from the precomputed `dense/spe11b_co2mass_w1_diff_<year>y.csv` tables. Both x and y
are linearly rescaled independently, via `input_scale_range` and `output_scale_range`
(both [0, 1] here), before splitting into train/validation/test sets.

Network: x is flattened to a vector of length input_dim = 2 * 120 * 840 and fed
through an FCNN with layer widths [input_dim, 64, 64, 1] (see `model.py`), so the
network maps a flattened image pair directly to a single scalar output approximating
the (scaled) distance between the two images.
"""


from pdb import set_trace as st
from pathlib import Path


import numpy as np
import optax

import utils_datasets
import utils_nn
import model
import train

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, validation_dataset, test_dataset = utils_datasets.create_datasets(
        total_number_images=6833,  # number of images to use; capped at 6833
        step=1,  # stride over image indices; >= 1
        start=34,  # first image index; >= 34 to skip year-0 images (no distance table)
        data_path=base_dir / "spe11b_tmco2_dt50y_images.npz",
        input_scale_range=(0, 1),
        output_scale_range=(0, 1),
    )

    images = train_dataset.images
    print(
        f"Images: {images.shape} {images.dtype} = {images.nbytes / 1e9:.2f} GB on device "
        f"(stored once each; the equivalent materialized pair tensor would be "
        f"{len(train_dataset) * np.prod(train_dataset.x_shape) * 4 / 1e9:.1f} GB for train alone)."
    )
    print(
        f"Pairs: {len(train_dataset)} train / {len(validation_dataset)} validation / "
        f"{len(test_dataset)} test"
    )

    input_dim = int(np.prod(train_dataset.x_shape))
    layer_widths = [input_dim, 64, 64, 1]
    seed = 0
    params = model.initialize_model(layer_widths, seed=seed)

    # train using Adam
    lr = 1e-2
    epochs_tot = 100
    batch_size = 128
    optimizer = optax.adam(lr)

    params_opt, loss_epochs, loss_train, loss_validation = train.train_model(
        params,
        train_dataset,
        validation_dataset,
        optimizer=optimizer,
        epochs_tot=epochs_tot,
        batch_size=batch_size,
    )

    # save results
    utils_nn.save_params(params_opt, results_dir / "params.pkl")
    np.savetxt(results_dir / "loss_epochs.txt", np.asarray(loss_epochs), fmt="%d")
    np.savetxt(results_dir / "loss_train.txt", np.asarray(loss_train))
    np.savetxt(results_dir / "loss_validation.txt", np.asarray(loss_validation))
    np.savez_compressed(
        results_dir / "test_dataset.npz",
        images=np.asarray(test_dataset.images),
        index_image_pairs_test=np.asarray(test_dataset.index_image_pairs),
        y_test=np.asarray(test_dataset.distances),
    )

    print("\nTraining finished")
