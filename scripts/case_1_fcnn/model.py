from pdb import set_trace as st
import jax.numpy as jnp
from jax import nn, random
from typing import List


def initialize_model(layer_widths: List[int]) -> dict[str, dict[str, jnp.ndarray]]:
    """
    Initialize parameters of a fully connected neural network.

    Args:
        layer_widths: List of integers defining the width of each layer,
                      e.g. [784, 128, 64, 10] creates a 3-layer network.

    Returns:
        Nested dict of the form {"layer_0": {"W": ..., "b": ...}, ...},
        compatible with jax.tree_util, jax.grad, and optax optimizers.
    """
    params = {}
    key = random.PRNGKey(0)

    for i, (fan_in, fan_out) in enumerate(zip(layer_widths[:-1], layer_widths[1:])):
        key, w_key = random.split(key)
        std = jnp.sqrt(2.0 / fan_in)
        W = random.normal(w_key, shape=(fan_in, fan_out), dtype=jnp.float32) * std
        b = jnp.zeros((fan_out,), dtype=jnp.float32)

        params[f"layer_{i}"] = {"W": W, "b": b}

    return params


def activation(x: jnp.ndarray) -> jnp.ndarray:
    return nn.relu(x)


def forward(
    params: dict[str, dict[str, jnp.ndarray]],
    x: jnp.ndarray,
) -> jnp.ndarray:
    """
    Forward pass through a fully connected neural network.

    Args:
        params: Nested param dict from initialize_model().
        x: Input array of shape (batch_size, input_dim) or (input_dim,).
        activation: Hidden layer activation function (default: ReLU).
                    The final layer is always linear (no activation).

    Returns:
        Output array of shape (batch_size, output_dim) or (output_dim,).
    """
    layers = [
        params[name]
        for name in sorted(params, key=lambda name: int(name.split("_")[1]))
    ]
    input_dim = layers[0]["W"].shape[0]
    if x.ndim != 1:
        x = jnp.reshape(x, (-1, input_dim))

    for layer in layers[:-1]:
        x = activation(x @ layer["W"] + layer["b"])

    last_layer = layers[-1]
    x = x @ last_layer["W"] + last_layer["b"]

    return x
