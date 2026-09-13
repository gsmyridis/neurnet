import unittest

import torch

from neurnet.cnn import LeNet


class LeNetVisualizationTests(unittest.TestCase):
    def test_extracts_both_convolutional_feature_maps(self) -> None:
        model = LeNet()
        images = torch.zeros((1, 3, 32, 32))

        feature_maps = model.extract_feature_maps(images)

        self.assertEqual(
            {name: tuple(features.shape) for name, features in feature_maps.items()},
            {
                "Convolution 1": (1, 6, 28, 28),
                "Convolution 2": (1, 16, 10, 10),
            },
        )


if __name__ == "__main__":
    unittest.main()
