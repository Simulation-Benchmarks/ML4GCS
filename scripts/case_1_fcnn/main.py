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

    (
        x_train,
        y_train,
        x_validation,
        y_validation,
        x_test,
        y_test,
    ) = utils_datasets.create_datasets(
        total_number_images=100,
        step=1,
        start=35,
        data_path=base_dir / "spe11b_tmco2_dt50y.npz",
        scale_range=(0, 1),
    )

    input_dim = int(np.prod(x_train.shape[1:]))
    layer_widths = [input_dim, 64, 64, 1]
    seed = 0
    params = model.initialize_model(layer_widths, seed=seed)

    # train using Adam
    lr = 1e-2
    epochs_tot = 10000
    optimizer = optax.adam(lr)

    params_opt, loss_epochs, loss_train, loss_validation = train.train_model(
        params,
        x_train,
        y_train,
        x_validation,
        y_validation,
        optimizer=optimizer,
        epochs_tot=epochs_tot,
    )

    # save results
    utils_nn.save_params(params_opt, results_dir / "params.pkl")
    np.savetxt(results_dir / "loss_epochs.txt", np.asarray(loss_epochs), fmt="%d")
    np.savetxt(results_dir / "loss_train.txt", np.asarray(loss_train))
    np.savetxt(results_dir / "loss_validation.txt", np.asarray(loss_validation))
    np.savez_compressed(
        results_dir / "test_dataset.npz",
        x_test=np.asarray(x_test),
        y_test=np.asarray(y_test),
    )

    print("\nTraining finished")
