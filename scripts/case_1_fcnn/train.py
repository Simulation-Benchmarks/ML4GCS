from pdb import set_trace as st
import jax
import jax.numpy as jnp
import numpy as np
import optax

import model


def loss_fn(params, x, y):
    """Mean squared error loss."""
    y_pred = jnp.squeeze(model.forward(params, x), axis=-1)
    return jnp.mean((y_pred - y) ** 2)


def pair_loss_fn(params, images, pairs, y):
    """Mean squared error over pairs gathered from a shared image stack."""
    return loss_fn(params, images[pairs], y)


def _batch_slices(n_samples, batch_size, rng=None):
    """Index arrays covering 0..n_samples, shuffled when rng is given.

    batch_size None means a single full-batch slice.
    """
    order = np.arange(n_samples) if rng is None else rng.permutation(n_samples)
    if batch_size is None:
        return [order]
    return [
        order[start : start + batch_size] for start in range(0, n_samples, batch_size)
    ]


def train_model(
    params,
    train_dataset,
    validation_dataset,
    optimizer,
    lr=1e-3,
    epochs_tot=10,
    test_spacing=100,
    batch_size=None,
    validation_batch_size=256,
    seed=0,
):
    """
    Train a fully connected neural network with optax Adam.

    Args:
        params: Initial model parameters.
        train_dataset: utils_datasets.PairDataset with the training pairs.
        validation_dataset: utils_datasets.PairDataset with the validation pairs.
        optimizer: optax optimizer.
        lr: Learning rate used when optimizer is not provided.
        epochs_tot: Number of training epochs.
        test_spacing: Evaluate and log losses every this many epochs.
        batch_size: Pairs per optimizer step. None runs one full-batch step per
                    epoch (fewest steps, but the whole pair tensor must fit on
                    the GPU: n_pairs * 2 * n_rows * n_cols * 4 bytes). An int
                    enables minibatching, which is what lets the number of
                    images grow past that limit.
        validation_batch_size: Pairs per validation forward pass; validation is
                    always streamed so it never needs the full pair tensor.
        seed: Seed for the epoch shuffling when minibatching.

    Returns:
        params: Trained parameters.
        loss_epochs: Epoch numbers where losses were evaluated.
        loss_train: Training losses evaluated every ``test_spacing`` epochs.
        loss_validation: Validation losses evaluated every ``test_spacing`` epochs.
    """

    opt_state = optimizer.init(params)
    rng = np.random.default_rng(seed)

    @jax.jit
    def train_step(params, opt_state, images, pairs, y):
        loss, grads = jax.value_and_grad(pair_loss_fn)(params, images, pairs, y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @jax.jit
    def eval_step(params, images, pairs, y):
        return pair_loss_fn(params, images, pairs, y)

    def evaluate(params, dataset):
        """Streamed mean loss over a dataset, weighted by batch size."""
        loss_sum, count = 0.0, 0
        for batch in _batch_slices(len(dataset), validation_batch_size):
            idx = jnp.asarray(batch)
            batch_loss = eval_step(
                params, dataset.images, dataset.pair_indices[idx], dataset.y[idx]
            )
            loss_sum += float(batch_loss) * len(batch)
            count += len(batch)
        return loss_sum / count

    loss_epochs = []
    loss_train = []
    loss_validation = []
    n_train = len(train_dataset)

    for epoch in range(epochs_tot):
        log_epoch = epoch % test_spacing == 0
        batch_losses, batch_counts = [], []

        for batch in _batch_slices(n_train, batch_size, rng):
            idx = jnp.asarray(batch)
            params, opt_state, batch_loss = train_step(
                params,
                opt_state,
                train_dataset.images,
                train_dataset.pair_indices[idx],
                train_dataset.y[idx],
            )
            if log_epoch:
                batch_losses.append(float(batch_loss))
                batch_counts.append(len(batch))

        if log_epoch:
            train_loss = float(np.average(batch_losses, weights=batch_counts))
            val_loss = evaluate(params, validation_dataset)

            loss_epochs.append(epoch + 1)
            loss_train.append(train_loss)
            loss_validation.append(val_loss)

            print(
                f"Epoch {epoch}/{epochs_tot} | train loss: {train_loss:.3e} | val loss: {val_loss:.3e}"
            )

    return params, loss_epochs, loss_train, loss_validation
