from collections.abc import Sequence
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor


def show_image(
    image: Tensor,
    norm_means: Sequence[float] = (0.0,),
    norm_stds: Sequence[float] = (1.0,),
) -> None:
    """
    Display image.
    """
    # Matplotlib expects HWC images, while PyTorch image tensors use CHW.
    # Detaching and moving to the CPU makes the tensor safe to convert to NumPy.
    np_image = image.detach().cpu().permute(1, 2, 0).numpy()

    # Reverse the channel-wise normalization applied by the dataset transform.
    means = np.asarray(norm_means, dtype=np_image.dtype)
    stds = np.asarray(norm_stds, dtype=np_image.dtype)
    np_image = np_image * stds + means

    # Keep floating-point rounding from producing invalid display values.
    np_image = np.clip(np_image, 0.0, 1.0)

    # Matplotlib represents grayscale images as HW rather than HW1.
    if np_image.shape[-1] == 1:
        np_image = np_image.squeeze(-1)

    plt.imshow(np_image)
    plt.show()


def show_feature_maps(
    features: Tensor,
    max_maps: int = 16,
    columns: int = 4,
    title: str | None = None,
) -> None:
    """Display convolutional feature-map channels in a grayscale grid."""
    if features.ndim == 4:
        # Select the first image when a complete BCHW batch is provided.
        features = features[0]
    if features.ndim != 3:
        raise ValueError("features must have CHW or BCHW layout")
    if max_maps <= 0:
        raise ValueError("max_maps must be greater than zero")
    if columns <= 0:
        raise ValueError("columns must be greater than zero")

    features = features.detach().cpu()
    num_maps = min(max_maps, features.shape[0])
    if num_maps == 0:
        raise ValueError("features must contain at least one channel")

    columns = min(columns, num_maps)
    rows = ceil(num_maps / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3 * columns, 3 * rows),
        squeeze=False,
    )

    if title is not None:
        figure.suptitle(title)

    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index < num_maps:
            axis.imshow(features[index].numpy(), cmap="gray")
            axis.set_title(f"Channel {index}")

    figure.tight_layout()
    plt.show()
