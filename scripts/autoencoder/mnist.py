"""Train a convolutional autoencoder on MNIST or Fashion MNIST."""

import argparse
from typing import Literal

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np
from mlx.optimizers import AdamW

from neurnet.arch.autoencoder.conv import ConvAE, train_autoencoder
from neurnet.datasets import FashionMNISTDataset, MLXDataLoader, MNISTDataset
from neurnet.utils.image import show_image


def get_train_test_dataloaders(
    name: Literal["mnist", "fashion-mnist"],
    batch_size: int,
    root_dir: str = "./data",
) -> tuple[MLXDataLoader, MLXDataLoader]:
    dataset_class = {"mnist": MNISTDataset, "fashion-mnist": FashionMNISTDataset}[name]
    train_set = dataset_class(
        root_dir=root_dir, train=True, shuffle=True, batch_size=batch_size
    )
    train_loader = train_set.to_mlx(prefetch_batches=4, prefetch_worker_threads=2)

    test_set = dataset_class(
        root_dir=root_dir, train=False, shuffle=False, batch_size=batch_size
    )
    test_loader = test_set.to_mlx(prefetch_batches=4, prefetch_worker_threads=2)

    return train_loader, test_loader


def show_latent_space(model: ConvAE, loader: MLXDataLoader) -> None:
    """Project test codes onto their first two principal components."""
    model.eval()
    loader.reset()

    codes, labels = [], []
    for images, targets in loader:
        codes.append(np.array(model.encode(images)))
        labels.append(np.array(targets))

    if not codes:
        raise ValueError("Cannot visualize an empty dataset.")

    latent = np.concatenate(codes, axis=0)
    labels = np.concatenate(labels)
    if min(latent.shape) < 2:
        raise ValueError("PCA needs at least two images and two latent dimensions.")

    # Center the codes, then project onto directions of greatest variance.
    centered = latent - latent.mean(axis=0)
    _, singular_values, directions = np.linalg.svd(centered, full_matrices=False)
    coordinates = centered @ directions[:2].T
    variance = singular_values**2
    total_variance = variance.sum()
    if total_variance == 0:
        raise ValueError(
            "All latent codes are identical; there is no variation to plot."
        )
    explained = variance[:2] / total_variance

    fig, ax = plt.subplots(figsize=(8, 6))
    points = ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=labels,
        cmap="tab10",
        vmin=-0.5,
        vmax=9.5,
        s=6,
        alpha=0.6,
    )
    fig.colorbar(points, ax=ax, ticks=range(10), label="Class label")
    ax.set_xlabel(f"PC1 ({explained[0]:.1%} of latent variance)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1%} of latent variance)")
    ax.set_title("Test latent space — PCA")
    fig.tight_layout()
    plt.show()


def show_reconstructions(model: ConvAE, loader: MLXDataLoader, count: int) -> None:
    """Display consecutive groups of up to four original/reconstruction pairs."""
    model.eval()
    loader.reset()
    rows = []
    for images, _ in loader:
        images = model.encoder.prepare_images(images[: count - len(rows)])
        reconstructed = model(images)
        rows.extend(
            mx.concatenate([original, reconstruction], axis=1)
            for original, reconstruction in zip(images, reconstructed, strict=True)
        )
        if len(rows) == count:
            break

    if not rows:
        raise ValueError("Cannot display images from an empty dataset.")
    for start in range(0, len(rows), 2):
        group = rows[start : start + 2]
        show_image(
            mx.concatenate(group, axis=0),
            title=(
                f"Images {start + 1}–{start + len(group)}: "
                "Originals (left) | Reconstructions (right)"
            ),
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=("mnist", "fashion-mnist"), default="mnist"
    )
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--code-size", type=int, default=16)
    parser.add_argument(
        "--show-pics",
        type=int,
        default=0,
        metavar="N",
        help="Display N test images with reconstructions, in groups of four (default: 0).",
    )
    parser.add_argument(
        "--plot-latent-pca",
        action="store_true",
        help="Plot test latent codes using PCA.",
    )
    args = parser.parse_args()

    if args.batch_size <= 0 or args.learning_rate <= 0:
        parser.error("batch size and learning rate must be positive")
    if args.epochs <= 0 or args.code_size <= 0:
        parser.error("epochs and code size must be positive")
    if args.plot_latent_pca and args.code_size < 2:
        parser.error("--plot-latent-pca requires --code-size >= 2")
    if args.show_pics < 0:
        parser.error("--show-pics must be nonnegative")

    return args


def main() -> None:
    args = parse_arguments()
    train_loader, test_loader = get_train_test_dataloaders(
        args.dataset, args.batch_size, args.data_root
    )

    model = ConvAE(
        code_size=args.code_size, image_dims=MNISTDataset.IMAGE_DIMS, in_channels=1
    )
    optimizer = AdamW(
        learning_rate=args.learning_rate, weight_decay=0.0, bias_correction=True
    )

    model = train_autoencoder(args.epochs, model, train_loader, test_loader, optimizer)

    if args.show_pics:
        show_reconstructions(model, test_loader, args.show_pics)

    if args.plot_latent_pca:
        show_latent_space(model, test_loader)


if __name__ == "__main__":
    main()
