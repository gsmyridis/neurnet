from collections.abc import Sequence

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
