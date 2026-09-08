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


def pair_loss_fn(params, images, index_image_pairs_batch, distances_batch):
    """Mean squared error over one batch of pairs from a shared image stack.

    ``images`` is the whole stack, passed unsliced so the gather stays inside
    jit; ``index_image_pairs_batch`` and ``distances_batch`` are already
    restricted to the batch by the caller.
    """
    return loss_fn(params, images[index_image_pairs_batch], distances_batch)


def _index_pair_batches(n_pairs, batch_size, rng=None):
    """Yield the index_pair values of each minibatch, as device arrays.

    These are *not* data: each yielded array holds row numbers into a
    PairDataset's ``index_image_pairs`` and ``distances``, i.e. index_pair
    values in [0, n_pairs). Together the batches cover every pair exactly once,
    in order when rng is None and shuffled otherwise.

    batch_size None means one single full-size batch.
    """
    index_pair_order = np.arange(n_pairs) if rng is None else rng.permutation(n_pairs)
    if batch_size is None:
        yield jnp.asarray(index_pair_order)
        return
    for start in range(0, n_pairs, batch_size):
        yield jnp.asarray(index_pair_order[start : start + batch_size])


def train_model(
    params,
    train_dataset,
    validation_dataset,
    optimizer,
    epochs_tot=10,
    test_spacing=10,
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
        epochs_tot: Number of training epochs.
        test_spacing: Evaluate and log losses every this many epochs.
        batch_size: index_pair values per optimizer step. None = full-batch
                    (fits only a few hundred images); an int enables minibatching.
        validation_batch_size: index_pair values per validation forward pass.
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
    def train_step(params, opt_state, images, index_image_pairs_batch, distances_batch):
        loss, grads = jax.value_and_grad(pair_loss_fn)(
            params, images, index_image_pairs_batch, distances_batch
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @jax.jit
    def eval_step(params, images, index_image_pairs_batch, distances_batch):
        return pair_loss_fn(params, images, index_image_pairs_batch, distances_batch)

    def evaluate(params, dataset):
        """Streamed mean loss over a dataset, weighted by batch size."""
        loss_sum, n_pairs_seen = 0.0, 0
        for index_pair_batch in _index_pair_batches(
            len(dataset), validation_batch_size
        ):
            batch_loss = eval_step(
                params,
                dataset.images,
                dataset.index_image_pairs[index_pair_batch],
                dataset.distances[index_pair_batch],
            )
            loss_sum += float(batch_loss) * len(index_pair_batch)
            n_pairs_seen += len(index_pair_batch)
        return loss_sum / n_pairs_seen

    loss_epochs = []
    loss_train = []
    loss_validation = []
    n_pairs_train = len(train_dataset)

    for epoch in range(epochs_tot):
        log_epoch = epoch % test_spacing == 0
        batch_losses, batch_counts = [], []

        for index_pair_batch in _index_pair_batches(n_pairs_train, batch_size, rng):
            
            params, opt_state, batch_loss = train_step(
                params,
                opt_state,
                train_dataset.images,
                train_dataset.index_image_pairs[index_pair_batch],
                train_dataset.distances[index_pair_batch],
            )
            if log_epoch:
                batch_losses.append(float(batch_loss))
                batch_counts.append(len(index_pair_batch))

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
