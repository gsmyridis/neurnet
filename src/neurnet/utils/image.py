import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor


def show_image(image: Tensor, norm_mean: float = 0, norm_std: float = 1):
    """
    Display image.
    """
    image = image * norm_std + norm_mean
    np_image = image.numpy()
    plt.imshow(np.transpose(np_image, (1, 2, 0)))
    plt.show()
