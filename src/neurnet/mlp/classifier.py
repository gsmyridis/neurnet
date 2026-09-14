from collections.abc import Callable
from typing import cast

import mlx.core as mx
from mlx import nn
from mlx.optimizers import Optimizer

from neurnet.utils.data import MLXDataLoader


class Flatten(nn.Module):
    def __init__(self, start_axis: int):
        super().__init__()
        self.start_axis = start_axis

    def __call__(self, x: mx.array) -> mx.array:
        return mx.flatten(x, start_axis=self.start_axis)


class MLPClassifier(nn.Module):
    def __init__(self, input_dims: int, n_classes: int):
        super().__init__()

        hidden_layer_1_dims = input_dims // 2
        hidden_layer_2_dims = hidden_layer_1_dims // 2
        self.feats = nn.Sequential(
            # Start from axis=1 to preserve the barch dimension.
            # e.g. (B, 28, 28, 1) -> (B, 784)
            Flatten(start_axis=1),
            nn.Linear(input_dims=input_dims, output_dims=hidden_layer_1_dims),
            nn.ReLU(),
            nn.Linear(input_dims=hidden_layer_1_dims, output_dims=hidden_layer_2_dims),
            nn.ReLU(),
            nn.Linear(input_dims=hidden_layer_2_dims, output_dims=n_classes),
        )

    def __call__(self, input: mx.array) -> mx.array:
        return self.feats(input)


def train_mlp_classifier(
    model: MLPClassifier,
    train_loader: MLXDataLoader,
    test_loader: MLXDataLoader,
    loss_fn: Callable[[nn.Module, mx.array, mx.array], float],
    optimizer: Optimizer,
    epochs: int,
) -> MLPClassifier:

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    for e in range(epochs):
        for images, labels in train_loader:
            _, grads = loss_and_grad_fn(model, images, labels)

            # Update the optimizer state and model parameters
            # in a single call
            optimizer.update(model, grads)

            # Force a graph evaluation
            mx.eval(model.parameters(), optimizer.state)

        accuracy = test_mlp_classifier(model, test_loader)
        train_loader.reset()
        print(f"[Epoch {e + 1} / {epochs}]: Test accuracy {100 * accuracy:.2f}%")

    return model


def test_mlp_classifier(model: MLPClassifier, test_loader: MLXDataLoader) -> float:
    model.eval()
    test_loader.reset()

    total_correct = 0
    total_samples = 0

    for images, labels in test_loader:
        predictions = mx.argmax(model(images), axis=1)
        batch_correct = mx.sum(predictions == labels)

        # mx.array.item() is broadly typed, but a sum of booleans is an integer.
        total_correct += cast(int, batch_correct.item())
        total_samples += labels.size

    if total_samples == 0:
        raise ValueError("Test loader is empty")

    return total_correct / total_samples
