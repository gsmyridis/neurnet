import unittest
from unittest.mock import patch

import numpy as np
import torch
from matplotlib import pyplot as plt

from neurnet.utils.image import show_feature_maps, show_image


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

    def test_displays_selected_feature_maps_in_a_grid(self) -> None:
        features = torch.arange(4 * 2 * 3).reshape(4, 2, 3)

        with patch("neurnet.utils.image.plt.show"):
            show_feature_maps(features, max_maps=3, columns=2, title="Convolution 1")

        figure = plt.gcf()
        self.addCleanup(plt.close, figure)

        self.assertEqual(len(figure.axes), 4)
        self.assertEqual([len(axis.images) for axis in figure.axes], [1, 1, 1, 0])
        np.testing.assert_array_equal(
            figure.axes[0].images[0].get_array(), features[0].numpy()
        )


if __name__ == "__main__":
    unittest.main()
