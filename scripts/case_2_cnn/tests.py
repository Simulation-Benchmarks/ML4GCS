import numpy as np
import jax.numpy as jnp
import optax
from spe11_wasserstein import data

import model
import postprocess
import train


def pair_dataset(maps, pairs, distances):
    """Dataset of the given maps, pairs and distances, with identity scalings."""
    identity = data.Standardization(mu=0.0, sigma=1.0)
    return data.Dataset(
        jnp.asarray(maps, dtype=jnp.float32),
        jnp.asarray(distances, dtype=jnp.float32),
        jnp.asarray(pairs, dtype=jnp.int32),
        identity,
        identity,
        n_participants=len(maps),
    )


def constant_cnn_params(value=1.0):
    """Constant CNN: f_θ(x) = value. All parameters are zero, except the output bias."""
    layers = {
        "conv_0": {
            "W": jnp.zeros((1, 2, 1, 1), dtype=jnp.float32),
            "b": jnp.zeros((1,), dtype=jnp.float32),
        },
        "dense_0": {
            "W": jnp.zeros((1, 1), dtype=jnp.float32),
            "b": jnp.zeros((1,), dtype=jnp.float32),
        },
        "dense_1": {
            "W": jnp.zeros((1, 1), dtype=jnp.float32),
            "b": jnp.array([value], dtype=jnp.float32),
        },
    }
    return model.CNNParameters(layers=layers, conv_strides=(1, 1))


def test_initialize_model_shapes():
    params = model.initialize_model(
        input_channels=2,
        conv_channels=(4, 8),
        kernel_size=(3, 5),
        conv_strides=(2, 2),
        dense_width=6,
        output_dim=1,
        seed=0,
    )

    assert params["conv_0"]["W"].shape == (4, 2, 3, 5)
    assert params["conv_0"]["b"].shape == (4,)
    assert params["conv_1"]["W"].shape == (8, 4, 3, 5)
    assert params["conv_1"]["b"].shape == (8,)
    assert params["dense_0"]["W"].shape == (8, 6)
    assert params["dense_0"]["b"].shape == (6,)
    assert params["dense_1"]["W"].shape == (6, 1)
    assert params["dense_1"]["b"].shape == (1,)


def test_initialize_model_is_deterministic():
    params_a = model.initialize_model(
        input_channels=2,
        conv_channels=(4,),
        kernel_size=(3, 3),
        conv_strides=(1, 1),
        dense_width=3,
        output_dim=1,
        seed=0,
    )
    params_b = model.initialize_model(
        input_channels=2,
        conv_channels=(4,),
        kernel_size=(3, 3),
        conv_strides=(1, 1),
        dense_width=3,
        output_dim=1,
        seed=0,
    )

    for layer_name in params_a:
        np.testing.assert_allclose(params_a[layer_name]["W"], params_b[layer_name]["W"])
        np.testing.assert_allclose(params_a[layer_name]["b"], params_b[layer_name]["b"])


def test_forward_output_shape():
    params = model.initialize_model(
        input_channels=2,
        conv_channels=(4,),
        kernel_size=(3, 3),
        conv_strides=(2, 2),
        dense_width=3,
        output_dim=1,
        seed=0,
    )

    x_single = jnp.ones((2, 16, 20), dtype=jnp.float32)
    y_single = model.forward(params, x_single)
    assert y_single.shape == (1,)

    x_batch = jnp.ones((5, 2, 16, 20), dtype=jnp.float32)
    y_batch = model.forward(params, x_batch)
    assert y_batch.shape == (5, 1)


def test_loss_zero_for_exact_prediction():
    params = constant_cnn_params(value=1.0)
    x = jnp.array([[0.25, 0.75], [0.4, 0.6]], dtype=jnp.float32).reshape(2, 2, 1, 1)
    y = jnp.array([1.0, 1.0], dtype=jnp.float32)

    assert float(train.loss_fn(params, x, y)) == 0.0


def test_loss_positive_for_wrong_prediction():
    params = constant_cnn_params(value=1.0)
    x = jnp.array([[0.25, 0.75]], dtype=jnp.float32).reshape(1, 2, 1, 1)
    y = jnp.array([2.0], dtype=jnp.float32)

    assert float(train.loss_fn(params, x, y)) > 0.0


def test_postprocess_metrics():
    params = constant_cnn_params(value=1.0)
    # Inputs (0.25, 0.75) and (0.2, 0.7), targets 1.0 and 2.0.
    maps = np.array([0.25, 0.75, 0.2, 0.7]).reshape(4, 1, 1)
    distances = np.ones((4, 4))
    distances[2, 3] = 2.0
    test_dataset = pair_dataset(maps, [[0, 1], [2, 3]], distances)

    y_test, y_pred = postprocess.compute_test_predictions(params, test_dataset)
    metrics = postprocess.compute_test_metrics(y_test, y_pred)

    np.testing.assert_allclose(metrics["mse"], 0.2, atol=1e-7)
    np.testing.assert_allclose(metrics["rmse"], np.sqrt(0.2), atol=1e-7)
    np.testing.assert_allclose(metrics["mae"], 1.0 / 3.0, atol=1e-7)
    np.testing.assert_allclose(metrics["r2"], -1.0, atol=1e-6)
    assert "accuracy" not in metrics


def test_train_model_records_losses():
    # Input (1.0, 0.0), target 1.0.
    maps = np.array([1.0, 0.0]).reshape(2, 1, 1)
    dataset = pair_dataset(maps, [[0, 1]], np.ones((2, 2)))
    params = constant_cnn_params(value=0.0)

    _, loss_epochs, loss_train, loss_validation = train.train_model(
        params,
        dataset,
        dataset,
        optimizer=optax.adam(0.01),
        epochs_tot=5,
        validation_spacing=2,
    )

    assert loss_epochs == [1, 3, 5]
    assert len(loss_train) == 3
    assert len(loss_validation) == 3


def test_train_single_input():
    # Input (0.25, -0.75), target 0.5.
    maps = np.array([0.25, -0.75]).reshape(2, 1, 1)
    dataset = pair_dataset(maps, [[0, 1]], np.full((2, 2), 0.5))
    params = constant_cnn_params(value=0.0)

    params, _, loss_train, loss_validation = train.train_model(
        params,
        dataset,
        dataset,
        optimizer=optax.adam(0.1),
        epochs_tot=300,
        validation_spacing=299,
    )

    final_loss = float(train.minibatch_loss(params, dataset, np.array([0])))
    assert final_loss < 1e-8
    assert loss_train[-1] < loss_train[0]
    assert loss_validation[-1] < loss_validation[0]


def run_tests():
    tests = (
        test_initialize_model_shapes,
        test_initialize_model_is_deterministic,
        test_forward_output_shape,
        test_loss_zero_for_exact_prediction,
        test_loss_positive_for_wrong_prediction,
        test_postprocess_metrics,
        test_train_model_records_losses,
        test_train_single_input,
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
