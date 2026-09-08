import numpy as np
import jax.numpy as jnp
import optax

import model
import postprocess
import train
import utils_datasets


def test_scaling_dataset():
    scale_range = (-1.0, 1.0)
    x = np.array([0.0, 2.5, 5.0, 7.5, 10.0], dtype=np.float32)
    y = np.array([2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)

    x, y = utils_datasets.scale_data(
        x, y, input_scale_range=scale_range, output_scale_range=scale_range
    )

    np.testing.assert_allclose(x, [-1.0, -0.5, 0.0, 0.5, 1.0])
    np.testing.assert_allclose(y, [-1.0, -0.5, 0.0, 0.5, 1.0])


def test_split_data():
    x = np.arange(20, dtype=np.float32).reshape(10, 2)
    y = np.arange(10, dtype=np.float32)

    x_train, y_train, x_validation, y_validation, x_test, y_test = (
        utils_datasets.split_data(x, y, train_split=0.6, validation_split=0.2)
    )

    assert x_train.shape == (6, 2)
    assert y_train.shape == (6,)
    assert x_validation.shape == (2, 2)
    assert y_validation.shape == (2,)
    assert x_test.shape == (2, 2)
    assert y_test.shape == (2,)
    np.testing.assert_allclose(y_train, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(y_validation, [6.0, 7.0])
    np.testing.assert_allclose(y_test, [8.0, 9.0])


def test_initialize_model_shapes():
    params = model.initialize_model([4, 3, 2, 1], seed=0)

    assert params["layer_0"]["W"].shape == (4, 3)
    assert params["layer_0"]["b"].shape == (3,)
    assert params["layer_1"]["W"].shape == (3, 2)
    assert params["layer_1"]["b"].shape == (2,)
    assert params["layer_2"]["W"].shape == (2, 1)
    assert params["layer_2"]["b"].shape == (1,)


def test_initialize_model_is_deterministic():
    params_a = model.initialize_model([4, 3, 1], seed=0)
    params_b = model.initialize_model([4, 3, 1], seed=0)

    for layer_name in params_a:
        np.testing.assert_allclose(params_a[layer_name]["W"], params_b[layer_name]["W"])
        np.testing.assert_allclose(params_a[layer_name]["b"], params_b[layer_name]["b"])


def test_forward_output_shape():
    params = model.initialize_model([4, 3, 1], seed=0)

    x_single = jnp.ones((4,), dtype=jnp.float32)
    y_single = model.forward(params, x_single)
    assert y_single.shape == (1,)

    x_batch = jnp.ones((5, 4), dtype=jnp.float32)
    y_batch = model.forward(params, x_batch)
    assert y_batch.shape == (5, 1)


def test_loss_zero_for_exact_prediction():
    params = {
        "layer_0": {
            "W": jnp.array([[1.0], [1.0]], dtype=jnp.float32),
            "b": jnp.array([0.0], dtype=jnp.float32),
        }
    }
    x = jnp.array([[0.25, 0.75], [0.4, 0.6]], dtype=jnp.float32)
    y = jnp.array([1.0, 1.0], dtype=jnp.float32)

    assert float(train.loss_fn(params, x, y)) == 0.0


def test_loss_positive_for_wrong_prediction():
    params = {
        "layer_0": {
            "W": jnp.array([[1.0], [1.0]], dtype=jnp.float32),
            "b": jnp.array([0.0], dtype=jnp.float32),
        }
    }
    x = jnp.array([[0.25, 0.75]], dtype=jnp.float32)
    y = jnp.array([2.0], dtype=jnp.float32)

    assert float(train.loss_fn(params, x, y)) > 0.0


def _pair_dataset(pair_values, distances):
    """PairDataset whose gathered pairs equal `pair_values`, one 1-pixel image
    per entry, so a gathered pair flattens to a length-2 model input."""
    flat = [value for pair in pair_values for value in pair]
    images = jnp.array(flat, dtype=jnp.float32).reshape(len(flat), 1)
    index_image_pairs = jnp.arange(len(flat), dtype=jnp.int32).reshape(
        len(pair_values), 2
    )
    return utils_datasets.PairDataset(
        images, index_image_pairs, jnp.array(distances, dtype=jnp.float32)
    )


def test_pair_dataset_gathers_without_duplicating():
    images = jnp.arange(12, dtype=jnp.float32).reshape(3, 2, 2)
    index_image_pairs = jnp.array([[0, 1], [1, 2], [0, 2], [2, 0]], dtype=jnp.int32)
    dataset = utils_datasets.PairDataset(images, index_image_pairs, jnp.zeros(4))

    pairs = dataset.gather()

    assert len(dataset) == 4
    assert dataset.x_shape == (2, 2, 2)
    assert pairs.shape == (4, 2, 2, 2)
    # Each pair is the right two images, and the stack itself never grew.
    for index_pair, (index_image_1, index_image_2) in enumerate(
        np.asarray(index_image_pairs)
    ):
        np.testing.assert_allclose(pairs[index_pair, 0], images[index_image_1])
        np.testing.assert_allclose(pairs[index_pair, 1], images[index_image_2])
    assert images.shape[0] == 3
    np.testing.assert_allclose(dataset.gather(slice(1, 3)), pairs[1:3])


def test_postprocess_metrics():
    params = {
        "layer_0": {
            "W": jnp.array([[1.0], [1.0]], dtype=jnp.float32),
            "b": jnp.array([0.0], dtype=jnp.float32),
        }
    }
    test_dataset = _pair_dataset([(0.25, 0.75), (0.2, 0.7)], [1.0, 1.0])

    metrics, _, _ = postprocess.compute_test_metrics(params, test_dataset)

    np.testing.assert_allclose(metrics["nmse"], 0.005, atol=1e-7)
    np.testing.assert_allclose(metrics["nrmse"], np.sqrt(0.005), atol=1e-7)
    np.testing.assert_allclose(metrics["nmae"], 0.05, atol=1e-7)
    assert "accuracy" not in metrics


def test_postprocess_metrics_batching_is_exact():
    """Batched inference must give the same metrics as a single forward pass."""
    params = model.initialize_model([2, 3, 1], seed=0)
    pair_values = [(0.1 * i, 0.2 * i) for i in range(7)]
    dataset = _pair_dataset(pair_values, [0.3 * i for i in range(7)])

    full, y_pred_full, _ = postprocess.compute_test_metrics(params, dataset, batch_size=64)
    split, y_pred_split, _ = postprocess.compute_test_metrics(params, dataset, batch_size=2)

    np.testing.assert_allclose(y_pred_full, y_pred_split, rtol=1e-6)
    for key in full:
        np.testing.assert_allclose(full[key], split[key], rtol=1e-6)


def test_train_model_records_losses():
    dataset = _pair_dataset([(1.0, 0.0)], [1.0])
    params = model.initialize_model([2, 1], seed=0)

    _, loss_epochs, loss_train, loss_validation = train.train_model(
        params,
        dataset,
        dataset,
        optimizer=optax.adam(0.01),
        epochs_tot=5,
        test_spacing=2,
    )

    assert loss_epochs == [1, 3, 5]
    assert len(loss_train) == 3
    assert len(loss_validation) == 3


def test_train_single_input():
    dataset = _pair_dataset([(0.25, -0.75)], [0.5])
    params = model.initialize_model([2, 1], seed=0)

    params, _, loss_train, loss_validation = train.train_model(
        params,
        dataset,
        dataset,
        optimizer=optax.adam(0.1),
        epochs_tot=300,
        test_spacing=299,
    )

    final_loss = float(
        train.pair_loss_fn(
            params, dataset.images, dataset.index_image_pairs, dataset.distances
        )
    )
    assert final_loss < 1e-8
    assert loss_train[-1] < loss_train[0]
    assert loss_validation[-1] < loss_validation[0]


def test_minibatch_training_matches_full_batch_loss():
    """Minibatching must cover every pair: same data, same starting loss."""
    pair_values = [(0.1 * i, -0.05 * i) for i in range(10)]
    targets = [0.2 * i for i in range(10)]
    dataset = _pair_dataset(pair_values, targets)
    params = model.initialize_model([2, 4, 1], seed=0)

    _, _, full_loss, _ = train.train_model(
        params, dataset, dataset, optimizer=optax.adam(0.0), epochs_tot=1, test_spacing=1
    )
    _, _, mini_loss, _ = train.train_model(
        params,
        dataset,
        dataset,
        optimizer=optax.adam(0.0),
        epochs_tot=1,
        test_spacing=1,
        batch_size=3,
    )

    # lr 0 means params never move, so both must report the same mean loss.
    np.testing.assert_allclose(full_loss[0], mini_loss[0], rtol=1e-5)


def run_tests():
    tests = (
        test_scaling_dataset,
        test_split_data,
        test_initialize_model_shapes,
        test_initialize_model_is_deterministic,
        test_forward_output_shape,
        test_loss_zero_for_exact_prediction,
        test_loss_positive_for_wrong_prediction,
        test_pair_dataset_gathers_without_duplicating,
        test_postprocess_metrics,
        test_postprocess_metrics_batching_is_exact,
        test_train_model_records_losses,
        test_train_single_input,
        test_minibatch_training_matches_full_batch_loss,
    )

    for test in tests:
        try:
            test()
        except Exception as error:
            print(f"{test.__name__}: test FAILED ({error})\n")
        else:
            print(f"{test.__name__}: test PASSED\n")


if __name__ == "__main__":
    run_tests()
