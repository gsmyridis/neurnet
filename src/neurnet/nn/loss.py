"""MLX loss functions shared by neural-network architectures."""

from collections.abc import Callable

import mlx.core as mx
from mlx import nn

type MLXLossFunction = Callable[[mx.array, mx.array], mx.array]


def _correct_energies(energies: mx.array, labels: mx.array) -> mx.array:
    return mx.squeeze(mx.take_along_axis(energies, labels[:, None], axis=1), axis=1)


def perceptron_loss(energies: mx.array, labels: mx.array) -> mx.array:
    """Mean of E(x, correct) - min_y E(x, y)."""
    return mx.mean(_correct_energies(energies, labels) - mx.min(energies, axis=1))


def hinge_loss(energies: mx.array, labels: mx.array, margin: float = 1.0) -> mx.array:
    """Require the correct energy to beat every incorrect energy by margin."""
    if margin < 0:
        raise ValueError("margin must be nonnegative")
    if energies.shape[1] < 2:
        raise ValueError("hinge loss requires at least two classes")

    class_ids = mx.arange(energies.shape[1])
    incorrect_energies = mx.where(
        class_ids[None, :] == labels[:, None], float("inf"), energies
    )
    gap = (
        margin
        + _correct_energies(energies, labels)
        - mx.min(incorrect_energies, axis=1)
    )
    return mx.mean(mx.maximum(gap, 0))


def negative_log_likelihood(energies: mx.array, labels: mx.array) -> mx.array:
    """Exact conditional NLL over the finite set of classes."""
    return mx.mean(nn.losses.cross_entropy(-energies, labels))
