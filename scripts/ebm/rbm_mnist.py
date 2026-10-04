"""Train an RBM-shaped energy model on MNIST using contrastive negatives."""

import argparse

from mlx.optimizers import Adam

from neurnet.arch.ebm import RestrictedBoltzmannMachine, train_rbm
from neurnet.datasets import MNIST_IMAGE_SIZE, MNISTDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-units", type=int, default=256)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--corruption-std", type=float, default=0.3)
    args = parser.parse_args()

    if args.batch_size <= 0 or args.learning_rate <= 0:
        parser.error("batch size and learning rate must be positive")
    if args.epochs <= 0:
        parser.error("epochs must be positive")
    if args.hidden_units <= 0 or args.corruption_std <= 0:
        parser.error("hidden units and corruption standard deviation must be positive")
    if args.margin < 0:
        parser.error("margin must be nonnegative")

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

    model = RestrictedBoltzmannMachine(
        visible_units=MNIST_IMAGE_SIZE,
        hidden_units=args.hidden_units,
    )
    train_rbm(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=Adam(learning_rate=args.learning_rate),
        epochs=args.epochs,
        margin=args.margin,
        corruption_std=args.corruption_std,
    )


if __name__ == "__main__":
    main()
