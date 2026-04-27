"""Dataset creation utilities."""

import numpy as np
import jax.numpy as jnp

from process_map_files_mod import get_maps_and_distance


DATA_PATH = "./spe11b_tmco2_dt50y.npz"
TRAIN_SPLIT = 0.7


def create_datasets(
    total_number_images: int = 45,
    step: int = 5,
    start: int = 35,
    data_path: str = DATA_PATH,
    train_split: float = TRAIN_SPLIT,
):
    """
    Load image pairs and distances, split into train/validation sets.

    Args:
        total_number_images: Upper bound for image index range.
        step: Step size when iterating over image indices.
        start: Starting index for image range.
        data_path: Path to the .npz data file.
        train_split: Fraction of data to use for training.

    Returns:
        x_train, y_train, x_validation, y_validation as JAX arrays.
    """
    x, y = [], []

    indices = range(start, total_number_images, step)
    for i in indices:
        for j in indices:
            print(f"Loading pair ({i}, {j})")
            image1, image2, distance = get_maps_and_distance(i, j, data_path)

            # Cast to float32 to ensure JAX compatibility (guards against object arrays)
            img1 = np.array(image1, dtype=np.float32)
            img2 = np.array(image2, dtype=np.float32)

            x.append(np.stack([img1, img2]))  # shape: (2, H, W)
            y.append(float(distance))

    x = jnp.array(np.array(x))  # shape: (N, 2, H, W)
    y = jnp.array(np.array(y))  # shape: (N,)

    split = int(train_split * x.shape[0])

    return x[:split], y[:split], x[split:], y[split:]
