from pdb import set_trace as st
import jax
import jax.numpy as jnp
import optax
from spe11_wasserstein import data

import model


def loss_fn(params, x, y):
    """Mean squared error loss."""
    y_pred = jnp.squeeze(model.forward(params, x), axis=-1)
    return jnp.mean((y_pred - y) ** 2)


def minibatch_loss(params, dataset, minibatch):
    """Loss on a minibatch, built on the fly by g: dataset[minibatch]."""
    (M, M_tilde), y = dataset[minibatch]
    return loss_fn(params, jnp.stack([M, M_tilde], axis=1), y)


def train_model(
    params,
    train_dataset,
    validation_dataset,
    optimizer,
    epochs_tot=10,
    batch_size=128,
    reshuffle=True,
    validation_spacing=1,
    seed=0,
):
    """
    Train a convolutional neural network with an optax optimizer, on minibatches.

    Args:
        params: Initial model parameters.
        train_dataset: D_train, a spe11_wasserstein.data.Dataset.
        validation_dataset: D_val, a spe11_wasserstein.data.Dataset.
        optimizer: optax optimizer.
        epochs_tot: Number of training epochs.
        batch_size: Batch size B.
        reshuffle: Reshuffle the training index sequence at each epoch.
        validation_spacing: Evaluate losses every ``validation_spacing`` epochs.
        seed: Random seed of the reshuffle.

    Returns:
        params: Trained parameters.
        loss_epochs: Epoch numbers where losses were evaluated.
        loss_train: Mean minibatch loss over each of those epochs.
        loss_validation: Validation loss at the end of each of those epochs.
    """

    opt_state = optimizer.init(params)
    train_minibatches = data.Minibatches(len(train_dataset), batch_size, seed)

    @jax.jit
    def train_step(params, opt_state, dataset, minibatch):
        loss, grads = jax.value_and_grad(minibatch_loss)(params, dataset, minibatch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @jax.jit
    def evaluate(params, dataset):
        """Loss on the whole dataset."""
        # In chunks of batch_size samples to avoid memory issues.
        squared_errors = jax.lax.map(
            lambda i: minibatch_loss(params, dataset, i[None]),  # sample i only
            jnp.arange(len(dataset)),
            batch_size=batch_size,
        )
        return jnp.mean(squared_errors)

    loss_epochs = []
    loss_train = []
    loss_validation = []

    for epoch in range(epochs_tot):
        if reshuffle:
            train_minibatches.reshuffle()

        loss_sum = 0.0
        for minibatch in train_minibatches:
            params, opt_state, loss = train_step(
                params, opt_state, train_dataset, minibatch
            )
            loss_sum += loss * len(minibatch)

        if epoch % validation_spacing == 0:
            train_loss = float(loss_sum) / len(train_dataset)
            val_loss = float(evaluate(params, validation_dataset))

            loss_epochs.append(epoch + 1)
            loss_train.append(train_loss)
            loss_validation.append(val_loss)

            print(
                f"Epoch {epoch}/{epochs_tot} | train loss: {train_loss:.3e} | val loss: {val_loss:.3e}"
            )

    return params, loss_epochs, loss_train, loss_validation
