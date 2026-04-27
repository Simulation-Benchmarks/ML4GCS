import jax
import jax.numpy as jnp
import optax

import utils_nn
import model


def loss_fn(params, x, y):
    """Mean squared error loss."""
    y_pred = model.forward(params, x)
    return jnp.mean((y_pred - y) ** 2)


def train_model(
    params, x_train, y_train, x_validation, y_validation, optimizer, epochs_tot=10
):
    """
    Train a fully connected neural network.

    Args:
        params:           Initial model parameters (list of dicts with "W" and "b")
        x_train:          Training inputs
        y_train:          Training targets
        x_validation:     Validation inputs
        y_validation:     Validation targets
        optimizer:        Any optax optimizer (e.g. optax.adam(1e-3))
        epochs_tot:       Number of training epochs

    Returns:
        params:           Trained parameters
        loss_train:       Training loss per epoch
        loss_validation:  Validation loss per epoch
    """
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @jax.jit
    def eval_step(params, x, y):
        return loss_fn(params, x, y)

    loss_train = []
    loss_validation = []

    for epoch in range(epochs_tot):
        params, opt_state, train_loss = train_step(params, opt_state, x_train, y_train)
        val_loss = eval_step(params, x_validation, y_validation)

        loss_train.append(float(train_loss))
        loss_validation.append(float(val_loss))

        print(
            f"Epoch {epoch + 1}/{epochs_tot} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f}"
        )

    return params, loss_train, loss_validation
