from pdb import set_trace as st
import jax
import jax.numpy as jnp
import optax

import model


def loss_fn(params, x, y):
    """Mean squared error loss."""
    y_pred = jnp.squeeze(model.forward(params, x), axis=-1)
    return jnp.mean((y_pred - y) ** 2)


def train_model(
    params,
    x_train,
    y_train,
    x_validation,
    y_validation,
    optimizer,
    lr=1e-3,
    epochs_tot=10,
    test_spacing=100,
):
    """
    Train a convolutional neural network with optax Adam.

    Args:
        params: Initial model parameters.
        x_train: Training inputs.
        y_train: Training targets.
        x_validation: Validation inputs.
        y_validation: Validation targets.
        optimizer: Optional optax optimizer. Defaults to optax.adam(lr).
        lr: Learning rate used when optimizer is not provided.
        epochs_tot: Number of training epochs.

    Returns:
        params: Trained parameters.
        loss_epochs: Epoch numbers where losses were evaluated.
        loss_train: Training losses evaluated every ``test_spacing`` epochs.
        loss_validation: Validation losses evaluated every ``test_spacing`` epochs.
    """

    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @jax.jit
    def eval_step(params, x, y):
        return loss_fn(params, x, y)

    loss_epochs = []
    loss_train = []
    loss_validation = []

    for epoch in range(epochs_tot):
        params, opt_state, train_loss = train_step(params, opt_state, x_train, y_train)
        val_loss = eval_step(params, x_validation, y_validation)

        if epoch % test_spacing == 0:
            loss_epochs.append(epoch + 1)
            loss_train.append(float(train_loss))
            loss_validation.append(float(val_loss))

            print(
                f"Epoch {epoch}/{epochs_tot} | train loss: {train_loss:.3e} | val loss: {val_loss:.3e}"
            )

    return params, loss_epochs, loss_train, loss_validation
