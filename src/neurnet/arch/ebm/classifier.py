"""A class-conditional energy model for MNIST classification."""

from typing import cast

import mlx.core as mx
from mlx import nn
from mlx.optimizers import Optimizer

from neurnet.arch.mlp import MLPClassifier
from neurnet.datasets import MLXDataLoader
from neurnet.nn import MLXLossFunction


class EnergyClassifier(MLPClassifier):
    """Interpret negative class logits as conditional energies."""

    def __init__(self, input_dims: int, n_classes: int, hidden_dims: int):
        super().__init__(input_dims, n_classes, hidden_dims)
        self.n_classes = n_classes

    def __call__(self, images: mx.array) -> mx.array:
        """Return energies with shape (batch_size, n_classes)."""
        return -super().__call__(images)

    def energy(self, images: mx.array, labels: mx.array) -> mx.array:
        """Return the energy of each image paired with its supplied label."""
        return mx.squeeze(
            mx.take_along_axis(self(images), labels[:, None], axis=1), axis=1
        )

    def predict(self, images: mx.array) -> mx.array:
        return mx.argmin(self(images), axis=1)


def train_energy_classifier(
    model: EnergyClassifier,
    train_loader: MLXDataLoader,
    test_loader: MLXDataLoader,
    optimizer: Optimizer,
    loss_function: MLXLossFunction,
    epochs: int,
) -> EnergyClassifier:
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    def loss_fn(
        model: EnergyClassifier, images: mx.array, targets: mx.array
    ) -> mx.array:
        return loss_function(model(images), targets)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    for epoch in range(epochs):
        model.train()
        train_loader.reset()
        for images, labels in train_loader:
            _, grads = loss_and_grad(model, images, labels)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

        accuracy = test_energy_classifier(model, test_loader)
        print(f"[Epoch {epoch + 1} / {epochs}]: Test accuracy {accuracy:.2%}")

    return model


def test_energy_classifier(
    model: EnergyClassifier, test_loader: MLXDataLoader
) -> float:
    model.eval()
    test_loader.reset()

    correct = 0
    total = 0
    for images, labels in test_loader:
        correct += cast(int, mx.sum(model.predict(images) == labels).item())
        total += labels.size

    if total == 0:
        raise ValueError("Test loader is empty")
    return correct / total
