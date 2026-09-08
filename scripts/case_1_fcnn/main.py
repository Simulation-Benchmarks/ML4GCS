"""
Trains a fully connected network (FCNN) to predict the distance between pairs of
SPE11B CO2 mass maps, as a cheap surrogate for the ground-truth pairwise distance
table used in the benchmark's dense evaluation.

Data: `utils_datasets.create_datasets` loads CO2 mass maps (120x840) from the SPE11B
dataset and forms all same-year pairs (img1, img2). For each pair, x is the pair
stacked as a (2, 120, 840) array and y is the corresponding scalar distance looked up
from the precomputed `dense/spe11b_co2mass_w1_diff_<year>y.csv` tables. Both x and y
are linearly rescaled independently, via `input_scale_range` and `output_scale_range`
(both [0, 1] here), before splitting into train/validation/test sets.

Network: x is flattened to a vector of length input_dim = 2 * 120 * 840 and fed
through an FCNN with layer widths [input_dim, 64, 64, 1] (see `model.py`), so the
network maps a flattened image pair directly to a single scalar output approximating
the (scaled) distance between the two maps.
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
        total_number_images=1000,  # images to use from global_array (indices start..this, clamped to 6833). Only the unique images are held (n_images * 120 * 840 * 4 bytes); pairs are gathered from them on the GPU, so this scales with images, not with the ~67x larger number of pairs.
        step=1,  # stride over image indices in range(start, total_number_images, step); must be >= 1
        start=34,  # starting column index into global_array/metadata; must be >= 34 to skip the year-0 images (indices 0-33), which have no dense/spe11b_co2mass_w1_diff_0y.csv distance table
        data_path=base_dir / "spe11b_tmco2_dt50y.npz",
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
    epochs_tot = 200
    # Pairs per optimizer step. None = one full-batch step per epoch, which
    # needs the whole pair tensor on the GPU (len(train) * 806 KB) and only
    # works for a few hundred images. An int streams minibatches gathered from
    # the shared image stack on the GPU -- no host<->device traffic per step.
    batch_size = 256
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
    # Save the shared image stack plus the test (i, j) pairs rather than the
    # materialized pairs, which would duplicate every image ~67 times.
    np.savez_compressed(
        results_dir / "test_dataset.npz",
        images=np.asarray(test_dataset.images),
        pair_indices_test=np.asarray(test_dataset.pair_indices),
        y_test=np.asarray(test_dataset.y),
    )

    print("\nTraining finished")
