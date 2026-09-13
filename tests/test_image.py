import unittest
from unittest.mock import patch

import numpy as np
import torch

from neurnet.utils.image import show_image


class ShowImageTests(unittest.TestCase):
    def test_denormalizes_channel_first_rgb_image(self) -> None:
        means = (0.490, 0.449, 0.411)
        stds = (0.231, 0.221, 0.230)
        image = torch.zeros((3, 2, 4))

        with (
            patch("neurnet.utils.image.plt.imshow") as imshow,
            patch("neurnet.utils.image.plt.show"),
        ):
            show_image(image, means, stds)

        shown_image = imshow.call_args.args[0]
        expected = np.broadcast_to(np.asarray(means), (2, 4, 3))
        np.testing.assert_allclose(shown_image, expected)


if __name__ == "__main__":
    unittest.main()
