from typing import cast

import mlx.core as mx
from mlx import nn
from mlx.nn.losses import cross_entropy
from mlx.optimizers import SGD

from neurnet.datasets import MNISTDataset
from neurnet.mlp import MLPClassifier, test_mlp_classifier, train_mlp_classifier


def loss_fn(model: nn.Module, images: mx.array, labels: mx.array) -> float:
    return cast(float, mx.mean(cross_entropy(model(images), labels)))


def main():

    # ===-------------------------------------------------------------------===
    # Constants
    # ===-------------------------------------------------------------------===

    DATA_ROOT = "./data"
    PREFETCH_BATCHES = 4
    PREFETCH_WORKER_THREADS = 2
    BATCH_SIZE = 32

    IMAGE_DIMS = 28 * 28
    N_CLASSES = 10

    LEARNING_RATE = 0.001
    N_EPOCHS = 10

    # ===-------------------------------------------------------------------===
    # Load training / testing dataloaders
    # ===-------------------------------------------------------------------===

    train_set = MNISTDataset(
        train=True,
        root_dir=DATA_ROOT,
        shuffle=True,
        batch_size=BATCH_SIZE,
    )
    train_loader = train_set.to_mlx(
        prefetch_batches=PREFETCH_BATCHES,
        prefetch_worker_threads=PREFETCH_WORKER_THREADS,
    )

    test_set = MNISTDataset(
        train=False,
        root_dir=DATA_ROOT,
        shuffle=False,
        batch_size=BATCH_SIZE,
    )
    test_loader = test_set.to_mlx(
        prefetch_batches=PREFETCH_BATCHES,
        prefetch_worker_threads=PREFETCH_WORKER_THREADS,
    )

    model = MLPClassifier(input_dims=IMAGE_DIMS, n_classes=N_CLASSES)
    optimizer = SGD(learning_rate=LEARNING_RATE)

    trained_model = train_mlp_classifier(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        epochs=N_EPOCHS,
    )

    accuracy = test_mlp_classifier(trained_model, test_loader)
    print(f"Accuracy of trained model: {100 * accuracy:.2f}%")


if __name__ == "__main__":
    main()
