from pdb import set_trace as st
import jax.numpy as jnp
from jax import nn, random
from typing import List

DEFAULT_SEED = 0


def initialize_model(
    layer_widths: List[int], seed: int = DEFAULT_SEED
) -> dict[str, dict[str, jnp.ndarray]]:
    """
    Initialize parameters of a fully connected neural network.
    From "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification" 10.1109/iccv.2015.123

    Args:
        layer_widths: List of integers defining the width of each layer,
                      e.g. [784, 128, 64, 10] creates a 3-layer network.
        seed: Random seed used for deterministic parameter initialization.

    Returns:
        Nested dict of the form {"layer_0": {"W": ..., "b": ...}, ...},
        compatible with jax.tree_util, jax.grad, and optax optimizers.
    """
    params = {}
    key = random.PRNGKey(seed)

    for i, (fan_in, fan_out) in enumerate(zip(layer_widths[:-1], layer_widths[1:])):
        key, w_key = random.split(key)
        std = jnp.sqrt(2.0 / fan_in)
        W = random.normal(w_key, shape=(fan_in, fan_out), dtype=jnp.float32) * std
        b = jnp.zeros((fan_out,), dtype=jnp.float32)

        params[f"layer_{i}"] = {"W": W, "b": b}

    return params


def activation(x: jnp.ndarray) -> jnp.ndarray:
    return nn.silu(x)


def forward(
    params: dict[str, dict[str, jnp.ndarray]],
    x: jnp.ndarray,
) -> jnp.ndarray:
    """
    Forward pass through a fully connected neural network.

    Args:
        params: Nested param dict from initialize_model().
        x: Input array of shape (batch_size, input_dim) or (input_dim,).

    Returns:
        Output array of shape (batch_size, output_dim) or (output_dim,).
    """
    num_layers = len(params)
    input_dim = params["layer_0"]["W"].shape[0]
    if x.ndim != 1:
        x = jnp.reshape(x, (-1, input_dim))

    for i in range(num_layers - 1):
        layer = params[f"layer_{i}"]
        x = activation(x @ layer["W"] + layer["b"])

    last_layer = params[f"layer_{num_layers - 1}"]
    x = x @ last_layer["W"] + last_layer["b"]

    return x
