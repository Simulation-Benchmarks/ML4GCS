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

    x_train, y_train, x_validation, y_validation = utils_datasets.create_datasets(
        total_number_images=45,
        step=1,
        start=35,
        data_path=base_dir / "spe11b_tmco2_dt50y.npz",
    )

    input_dim = int(np.prod(x_train.shape[1:]))
    layer_widths = [input_dim, 50, 50, 1]
    params = model.initialize_model(layer_widths)

    # train using Adam
    lr = 1e-1
    epochs_tot = 1000
    optimizer = optax.adam(lr)

    params_opt, loss_train, loss_validation = train.train_model(
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
    np.savetxt(results_dir / "loss_train.txt", np.asarray(loss_train))
    np.savetxt(results_dir / "loss_validation.txt", np.asarray(loss_validation))

    print("\nTraining finished")
