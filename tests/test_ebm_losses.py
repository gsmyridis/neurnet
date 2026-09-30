import math
import unittest

import mlx.core as mx

from neurnet.nn.loss import (
    hinge_loss,
    negative_log_likelihood,
    perceptron_loss,
)


class MulticlassEnergyLossTests(unittest.TestCase):
    def test_perceptron_and_hinge_use_the_lowest_competing_energy(self) -> None:
        energies = mx.array([[0.0, 2.0, 3.0], [3.0, 1.0, 0.0], [0.4, 1.0, 2.0]])
        labels = mx.array([0, 1, 0], dtype=mx.uint8)

        self.assertAlmostEqual(perceptron_loss(energies, labels).item(), 1 / 3)
        self.assertAlmostEqual(hinge_loss(energies, labels).item(), 0.8, places=6)
        self.assertAlmostEqual(
            hinge_loss(energies, labels, margin=0).item(),
            perceptron_loss(energies, labels).item(),
        )

    def test_nll_matches_cross_entropy_of_negative_energies(self) -> None:
        energies = mx.array([[0.0, 2.0, 3.0], [3.0, 1.0, 0.0]])
        labels = mx.array([0, 1], dtype=mx.uint8)

        expected = (
            math.log(1 + math.exp(-2) + math.exp(-3))
            + 1
            + math.log(math.exp(-3) + math.exp(-1) + 1)
        ) / 2
        self.assertAlmostEqual(
            negative_log_likelihood(energies, labels).item(), expected
        )

    def test_losses_ignore_a_common_energy_shift(self) -> None:
        energies = mx.array([[0.0, 2.0, 3.0], [3.0, 1.0, 0.0]])
        labels = mx.array([0, 1])
        shifted = energies + mx.array([[5.0], [-4.0]])

        for loss in (perceptron_loss, hinge_loss, negative_log_likelihood):
            with self.subTest(loss=loss.__name__):
                self.assertAlmostEqual(
                    loss(energies, labels).item(), loss(shifted, labels).item()
                )

    def test_invalid_hinge_margin_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "margin must be nonnegative"):
            hinge_loss(mx.array([[0.0, 1.0]]), mx.array([0]), margin=-1)


if __name__ == "__main__":
    unittest.main()
