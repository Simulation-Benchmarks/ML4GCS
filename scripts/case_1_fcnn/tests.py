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


def test_postprocess_metrics():
    params = {
        "layer_0": {
            "W": jnp.array([[1.0], [1.0]], dtype=jnp.float32),
            "b": jnp.array([0.0], dtype=jnp.float32),
        }
    }
    x_test = jnp.array([[0.25, 0.75], [0.2, 0.7]], dtype=jnp.float32)
    y_test = jnp.array([1.0, 1.0], dtype=jnp.float32)

    metrics = postprocess.compute_test_metrics(params, x_test, y_test)

    np.testing.assert_allclose(metrics["mse"], 0.005, atol=1e-7)
    np.testing.assert_allclose(metrics["rmse"], np.sqrt(0.005), atol=1e-7)
    np.testing.assert_allclose(metrics["mae"], 0.05, atol=1e-7)
    assert "accuracy" not in metrics


def test_train_model_records_losses():
    x = jnp.array([[1.0, 0.0]], dtype=jnp.float32)
    y = jnp.array([1.0], dtype=jnp.float32)
    params = model.initialize_model([2, 1], seed=0)

    _, loss_epochs, loss_train, loss_validation = train.train_model(
        params,
        x,
        y,
        x,
        y,
        optimizer=optax.adam(0.01),
        epochs_tot=5,
        test_spacing=2,
    )

    assert loss_epochs == [1, 3, 5]
    assert len(loss_train) == 3
    assert len(loss_validation) == 3


def test_train_single_input():
    x = jnp.array([[0.25, -0.75]], dtype=jnp.float32)
    y = jnp.array([0.5], dtype=jnp.float32)
    params = model.initialize_model([2, 1], seed=0)

    params, _, loss_train, loss_validation = train.train_model(
        params,
        x,
        y,
        x,
        y,
        optimizer=optax.adam(0.1),
        epochs_tot=300,
        test_spacing=299,
    )

    final_loss = float(train.loss_fn(params, x, y))
    assert final_loss < 1e-8
    assert loss_train[-1] < loss_train[0]
    assert loss_validation[-1] < loss_validation[0]


def run_tests():
    tests = (
        test_scaling_dataset,
        test_split_data,
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
