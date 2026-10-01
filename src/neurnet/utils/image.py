from collections.abc import Sequence
from math import ceil
from typing import cast

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np
from torch import Tensor

# ===--------------------------------------------------------------------------===
# Embed image
# ===--------------------------------------------------------------------------===


def embed_image[T: np.ndarray | mx.array | Tensor](
    image: T,
    dim_x: int,
    dim_y: int,
    center_x: int,
    center_y: int,
) -> T:
    """Embed an image in a zero canvas, preserving its array type and layout."""
    if isinstance(image, np.ndarray):
        return cast(T, _embed_image_np(image, dim_x, dim_y, center_x, center_y))
    elif isinstance(image, mx.array):
        embedded = _embed_image_np(np.array(image), dim_x, dim_y, center_x, center_y)
        return cast(T, mx.array(embedded))
    elif isinstance(image, Tensor):
        return cast(T, _embed_image_torch(image, dim_x, dim_y, center_x, center_y))
    raise TypeError(f"unsupported image type: {type(image).__name__}")


def _embed_image_np(
    image: np.ndarray,
    dim_x: int,
    dim_y: int,
    center_x: int,
    center_y: int,
) -> np.ndarray:
    if image.ndim < 2:
        raise ValueError("image must have at least two dimensions")

    canvas = np.zeros((dim_y, dim_x, *image.shape[2:]), dtype=image.dtype)
    source_y, source_x, canvas_y, canvas_x = _embedding_slices(
        image.shape[0], image.shape[1], dim_x, dim_y, center_x, center_y
    )
    canvas[canvas_y, canvas_x, ...] = image[source_y, source_x, ...]
    return canvas


def _embed_image_torch(
    image: Tensor,
    dim_x: int,
    dim_y: int,
    center_x: int,
    center_y: int,
) -> Tensor:
    if image.ndim < 2:
        raise ValueError("image must have at least two dimensions")

    canvas = image.new_zeros((*image.shape[:-2], dim_y, dim_x))
    source_y, source_x, canvas_y, canvas_x = _embedding_slices(
        image.shape[-2], image.shape[-1], dim_x, dim_y, center_x, center_y
    )
    canvas[..., canvas_y, canvas_x] = image[..., source_y, source_x]
    return canvas


def _embedding_slices(
    image_height: int,
    image_width: int,
    dim_x: int,
    dim_y: int,
    center_x: int,
    center_y: int,
) -> tuple[slice, slice, slice, slice]:
    image_top = center_y - image_height // 2
    image_left = center_x - image_width // 2
    canvas_top = max(image_top, 0)
    canvas_left = max(image_left, 0)
    canvas_bottom = min(image_top + image_height, dim_y)
    canvas_right = min(image_left + image_width, dim_x)

    if canvas_top >= canvas_bottom or canvas_left >= canvas_right:
        empty = slice(0, 0)
        return empty, empty, empty, empty

    source_y = slice(canvas_top - image_top, canvas_bottom - image_top)
    source_x = slice(canvas_left - image_left, canvas_right - image_left)
    canvas_y = slice(canvas_top, canvas_bottom)
    canvas_x = slice(canvas_left, canvas_right)
    return source_y, source_x, canvas_y, canvas_x


# ===--------------------------------------------------------------------------===
# Show image
# ===--------------------------------------------------------------------------===


def show_image(
    image: np.ndarray | mx.array | Tensor,
    norm_means: Sequence[float] = (0.0,),
    norm_stds: Sequence[float] = (1.0,),
    *,
    title: str | None = None,
) -> None:
    """Display image."""
    if isinstance(image, np.ndarray):
        _show_image(image, norm_means, norm_stds, title=title)
    elif isinstance(image, Tensor):
        # Matplotlib expects HWC images, while PyTorch image tensors use CHW.
        # Detaching and moving to the CPU makes the tensor safe to convert to NumPy.
        np_image = image.detach().cpu().permute(1, 2, 0).numpy()
        _show_image(np_image, norm_means, norm_stds, title=title)
    elif isinstance(image, mx.array):
        _show_image(np.array(image), norm_means, norm_stds, title=title)


def _show_image(
    image: np.ndarray,
    norm_means: Sequence[float] = (0.0,),
    norm_stds: Sequence[float] = (1.0,),
    *,
    title: str | None = None,
) -> None:
    """Display image."""
    # Reverse the channel-wise normalization applied by the dataset transform.
    means = np.asarray(norm_means, dtype=image.dtype)
    stds = np.asarray(norm_stds, dtype=image.dtype)
    image = image * stds + means

    # Keep floating-point rounding from producing invalid display values.
    image = np.clip(image, 0.0, 1.0)

    # Matplotlib represents grayscale images as HW rather than HW1.
    if image.shape[-1] == 1:
        image = image.squeeze(-1)

    if title is not None:
        plt.title(title)

    plt.imshow(image)
    plt.show()


# ===--------------------------------------------------------------------------===
# Show feature maps
# ===--------------------------------------------------------------------------===


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
