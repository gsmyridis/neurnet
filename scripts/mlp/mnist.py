import argparse

import mlx.core as mx
from mlx import nn
from mlx.optimizers import Adam

from neurnet.arch.mlp import MLPClassifier, train_mlp_classifier
from neurnet.datasets import MNISTDataset


def classification_loss(predictions: mx.array, labels: mx.array) -> mx.array:
    return mx.mean(nn.losses.cross_entropy(predictions, labels))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    if args.batch_size <= 0 or args.learning_rate <= 0:
        parser.error("batch size and learning rate must be positive")
    if args.epochs <= 0:
        parser.error("epochs must be positive")

    train_loader = MNISTDataset(
        train=True,
        root_dir=args.data_root,
        shuffle=True,
        batch_size=args.batch_size,
    ).to_mlx(prefetch_batches=4, prefetch_worker_threads=2)
    test_loader = MNISTDataset(
        train=False,
        root_dir=args.data_root,
        shuffle=False,
        batch_size=args.batch_size,
    ).to_mlx(prefetch_batches=4, prefetch_worker_threads=2)

    model = MLPClassifier(
        input_dims=MNISTDataset.IMAGE_SIZE,
        n_classes=MNISTDataset.N_CLASSES,
        hidden_dims=MNISTDataset.IMAGE_SIZE // 2,
    )
    train_mlp_classifier(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=Adam(learning_rate=args.learning_rate),
        loss_function=classification_loss,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
