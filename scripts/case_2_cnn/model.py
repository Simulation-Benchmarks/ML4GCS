from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jnp
from jax import lax, nn, random, tree_util

DEFAULT_SEED = 0


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CNNParameters(Mapping[str, dict[str, jnp.ndarray]]):
    layers: dict[str, dict[str, jnp.ndarray]]
    conv_strides: tuple[int, int]

    def __getitem__(self, layer_name: str) -> dict[str, jnp.ndarray]:
        return self.layers[layer_name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.layers)

    def __len__(self) -> int:
        return len(self.layers)

    def tree_flatten(self):
        return (self.layers,), self.conv_strides

    @classmethod
    def tree_unflatten(cls, conv_strides, children):
        (layers,) = children
        return cls(layers=layers, conv_strides=conv_strides)


def _he_normal(
    key: jnp.ndarray,
    shape: tuple[int, ...],
    fan_in: int,
) -> jnp.ndarray:
    std = jnp.sqrt(2.0 / fan_in)
    return random.normal(key, shape=shape, dtype=jnp.float32) * std


def initialize_model(
    input_channels: int,
    conv_channels: Sequence[int],
    kernel_size: tuple[int, int],
    conv_strides: tuple[int, int],
    dense_width: int,
    output_dim: int,
    seed: int = DEFAULT_SEED,
) -> CNNParameters:
    """
    Initialize parameters of a convolutional neural network for image-pair regression.

    Args:
        input_channels: Number of input image channels.
        conv_channels: Output channels for each convolutional layer.
        kernel_size: Spatial kernel size for all convolutional layers.
        conv_strides: Spatial stride for all convolutional layers.
        dense_width: Hidden width of the dense regression head.
        output_dim: Number of regression outputs.
        seed: Random seed used for deterministic parameter initialization.

    Returns:
        CNN parameters compatible with jax.tree_util, jax.grad, and optax.
    """
    layers = {}
    key = random.PRNGKey(seed)
    in_channels = input_channels
    kernel_h, kernel_w = kernel_size
    conv_strides = tuple(conv_strides)

    for i, out_channels in enumerate(conv_channels):
        key, w_key = random.split(key)
        fan_in = in_channels * kernel_h * kernel_w
        layers[f"conv_{i}"] = {
            "W": _he_normal(
                w_key,
                (out_channels, in_channels, kernel_h, kernel_w),
                fan_in,
            ),
            "b": jnp.zeros((out_channels,), dtype=jnp.float32),
        }
        in_channels = out_channels

    key, dense_key, output_key = random.split(key, 3)
    layers["dense_0"] = {
        "W": _he_normal(dense_key, (in_channels, dense_width), in_channels),
        "b": jnp.zeros((dense_width,), dtype=jnp.float32),
    }
    layers["dense_1"] = {
        "W": _he_normal(output_key, (dense_width, output_dim), dense_width),
        "b": jnp.zeros((output_dim,), dtype=jnp.float32),
    }

    return CNNParameters(layers=layers, conv_strides=conv_strides)


def activation(x: jnp.ndarray) -> jnp.ndarray:
    return nn.silu(x)


def forward(
    params: CNNParameters,
    x: jnp.ndarray,
) -> jnp.ndarray:
    """
    Forward pass through a convolutional neural network.

    Args:
        params: CNN parameters from initialize_model().
        x: Input array of shape (batch_size, channels, height, width) or
           (channels, height, width).

    Returns:
        Output array of shape (batch_size, output_dim) or (output_dim,).
    """
    single_input = x.ndim == 3
    if single_input:
        x = x[jnp.newaxis, ...]

    num_conv_layers = len(params) - 2
    for i in range(num_conv_layers):
        layer = params[f"conv_{i}"]
        x = lax.conv_general_dilated(
            x,
            layer["W"],
            window_strides=params.conv_strides,
            padding="SAME",
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )
        x = activation(x + layer["b"][jnp.newaxis, :, jnp.newaxis, jnp.newaxis])

    x = jnp.mean(x, axis=(2, 3))
    dense = params["dense_0"]
    x = activation(x @ dense["W"] + dense["b"])
    output = params["dense_1"]
    x = x @ output["W"] + output["b"]

    return x[0] if single_input else x
