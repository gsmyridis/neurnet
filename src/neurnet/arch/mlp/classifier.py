from typing import cast

import mlx.core as mx
from mlx import nn
from mlx.optimizers import Optimizer

from neurnet.datasets import MLXDataLoader
from neurnet.nn import Flatten, MLXLossFunction


class MLPClassifier(nn.Module):
    def __init__(self, input_dims: int, n_classes: int, hidden_dims: int):
        super().__init__()
        if input_dims <= 0 or n_classes <= 1 or hidden_dims <= 0:
            raise ValueError(
                "input_dims and hidden_dims must be positive; n_classes > 1"
            )

        self.image_encoder = nn.Sequential(
            # Start from axis=1 to preserve the barch dimension.
            # e.g. (B, 28, 28, 1) -> (B, 784)
            Flatten(start_axis=1),
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(),
        )
        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, n_classes),
        )

    def __call__(self, images: mx.array) -> mx.array:
        """Return class logits with shape (batch_size, n_classes)."""
        return self.classifier_head(self.image_encoder(images))

    def predict(self, images: mx.array) -> mx.array:
        return mx.argmax(self(images), axis=1)


def train_mlp_classifier(
    model: MLPClassifier,
    train_loader: MLXDataLoader,
    test_loader: MLXDataLoader,
    optimizer: Optimizer,
    loss_function: MLXLossFunction,
    epochs: int,
) -> MLPClassifier:
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    def loss_fn(model: MLPClassifier, images: mx.array, targets: mx.array) -> mx.array:
        return loss_function(model(images), targets)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    for epoch in range(epochs):
        model.train()
        train_loader.reset()
        for images, labels in train_loader:
            _, grads = loss_and_grad(model, images, labels)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

        accuracy = test_mlp_classifier(model, test_loader)
        print(f"[Epoch {epoch + 1} / {epochs}]: Test accuracy {accuracy:.2%}")

    return model


def test_mlp_classifier(model: MLPClassifier, test_loader: MLXDataLoader) -> float:
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
