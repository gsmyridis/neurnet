"""RBM-shaped energy model and contrastive energy training."""

from typing import cast

import mlx.core as mx
from mlx import nn
from mlx.optimizers import Optimizer

from neurnet.datasets import MLXDataLoader
from neurnet.nn import contrastive_energy_loss


class RestrictedBoltzmannMachine(nn.Module):
    """Bipartite visible-hidden energy model with deterministic hidden units.

    The hidden units are minimized out of the joint energy, so this model does
    not compute Boltzmann probabilities or use probabilistic Gibbs sampling.
    """

    def __init__(self, visible_units: int, hidden_units: int):
        super().__init__()
        if visible_units <= 0 or hidden_units <= 0:
            raise ValueError("visible_units and hidden_units must be positive")

        self.visible_units = visible_units
        self.hidden_units = hidden_units
        self.weights = mx.random.normal(shape=(visible_units, hidden_units), scale=0.01)
        self.visible_bias = mx.zeros((visible_units,))
        self.hidden_bias = mx.zeros((hidden_units,))

    def energy(self, visible: mx.array) -> mx.array:
        """Return minimum joint energy over binary hidden-unit states."""
        hidden_activation = visible @ self.weights + self.hidden_bias
        hidden_energy = mx.sum(mx.maximum(hidden_activation, 0.0), axis=1)
        visible_energy = mx.sum(visible * self.visible_bias, axis=1)
        return -(visible_energy + hidden_energy)


def corrupt_images(images: mx.array, noise_std: float = 0.3) -> mx.array:
    """Create negative examples with clipped additive Gaussian pixel noise."""
    if noise_std <= 0:
        raise ValueError("noise_std must be positive")
    noise = mx.random.normal(shape=images.shape, scale=noise_std)
    return mx.clip(images + noise, 0.0, 1.0)


def train_rbm(
    model: RestrictedBoltzmannMachine,
    train_loader: MLXDataLoader,
    test_loader: MLXDataLoader,
    optimizer: Optimizer,
    epochs: int,
    margin: float = 1.0,
    corruption_std: float = 0.3,
) -> RestrictedBoltzmannMachine:
    """Train by ranking data images below their corrupted counterparts."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if margin < 0:
        raise ValueError("margin must be nonnegative")
    if corruption_std <= 0:
        raise ValueError("corruption_std must be positive")

    def loss_fn(model: RestrictedBoltzmannMachine, images: mx.array) -> mx.array:
        negatives = corrupt_images(images, noise_std=corruption_std)
        return contrastive_energy_loss(
            model.energy(images), model.energy(negatives), margin=margin
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    for epoch in range(epochs):
        model.train()
        train_loader.reset()
        for images, _labels in train_loader:
            _, grads = loss_and_grad(model, images)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

        accuracy = test_rbm(
            model,
            test_loader,
            margin=margin,
            corruption_std=corruption_std,
        )
        print(
            f"[Epoch {epoch + 1} / {epochs}]: "
            f"Test contrastive energy accuracy {accuracy:.2%}"
        )

    return model


def test_rbm(
    model: RestrictedBoltzmannMachine,
    test_loader: MLXDataLoader,
    margin: float = 1.0,
    corruption_std: float = 0.3,
) -> float:
    """Measure the fraction of test images ranked below their corruptions."""
    if margin < 0:
        raise ValueError("margin must be nonnegative")
    if corruption_std <= 0:
        raise ValueError("corruption_std must be positive")

    model.eval()
    test_loader.reset()

    correct = 0
    total = 0
    for images, _labels in test_loader:
        negatives = corrupt_images(images, noise_std=corruption_std)
        ranked_below = model.energy(images) + margin < model.energy(negatives)
        correct += cast(int, mx.sum(ranked_below).item())
        total += images.shape[0]

    if total == 0:
        raise ValueError("Test loader is empty")
    return correct / total
