from pdb import set_trace as st
from pathlib import Path


import jax.numpy as jnp
import numpy as np
import optax
from spe11_wasserstein import data

import utils_nn
import model
import train

base_dir = Path(__file__).resolve().parent
results_dir = base_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)

train_dataset, validation_dataset, test_dataset = data.make_datasets(
    asarray=jnp.asarray
)
print(
    f"Samples: {len(train_dataset)} train / {len(validation_dataset)} validation / "
    f"{len(test_dataset)} test"
)

seed = 0
params = model.initialize_model(
    input_channels=2,  # M and M̃
    conv_channels=(4, 8),
    kernel_size=(5, 5),
    conv_strides=(2, 2),
    dense_width=16,
    output_dim=1,
    seed=seed,
)

# train using Adam
lr = 1e-4
epochs_tot = 10
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

print("\nTraining finished")
