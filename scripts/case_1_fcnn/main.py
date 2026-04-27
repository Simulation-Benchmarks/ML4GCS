import jax
import optax
from pdb import set_trace as st

import utils_datasets
import utils_nn
import model
import train


if __name__ == "__main__":

    # initialize model and datasets
    layer_widths = [120 * 840, 10, 1]
    params = model.initialize_model(layer_widths)
    x_train, y_train, x_validation, y_validation = utils_datasets.create_datasets()

    # train using Adam
    lr = 1e-3
    epochs_tot = 10
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
    utils_nn.save_params(params_opt)
    np.savetxt("./results/loss_train", loss_train)
    np.savetxt("./results/loss_test", loss_test)

print("\nTraining finished")
