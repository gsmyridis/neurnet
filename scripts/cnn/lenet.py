import argparse
from math import isfinite
from pathlib import Path

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from neurnet.cnn import LeNet, test_lenet, train_lenet

torch.use_deterministic_algorithms(True)

# ===-----------------------------------------------------------------------===
# Constants
# ===-----------------------------------------------------------------------===

DATA_DIR = "./data"
CIFAR10_CLASSES = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# ===-----------------------------------------------------------------------===
# Argument parsing
# ===-----------------------------------------------------------------------===


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or test LeNet on CIFAR-10.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a LeNet model.")
    train_parser.add_argument("--epochs", type=positive_int, default=50)
    train_parser.add_argument("--learning-rate", type=positive_float, default=0.001)
    train_parser.add_argument("--train-batch-size", type=positive_int, default=8)
    train_parser.add_argument("--test-batch-size", type=positive_int, default=10_000)
    train_parser.add_argument("--save-to", type=Path, required=True)
    train_parser.set_defaults(handler=train)

    test_parser = subparsers.add_parser("test", help="Test a saved LeNet model.")
    test_parser.add_argument("--model-path", type=Path, required=True)
    test_parser.add_argument("--batch-size", type=positive_int, default=10_000)
    test_parser.set_defaults(handler=test)

    return parser


# ===-----------------------------------------------------------------------===
# Dataloaders
# ===-----------------------------------------------------------------------===


def get_train_data(batch_size: int) -> DataLoader:
    train_transforms = T.Compose(
        [
            T.RandomHorizontalFlip(),
            T.RandomCrop(size=32, padding=4),
            T.ToTensor(),
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    train_set = CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=train_transforms
    )
    return DataLoader(train_set, batch_size=batch_size, shuffle=True)


def get_test_data(batch_size: int) -> DataLoader:
    test_transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    test_set = CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=test_transform
    )
    return DataLoader(test_set, batch_size=batch_size, shuffle=False)


# ===-----------------------------------------------------------------------===
# Train / Test
# ===-----------------------------------------------------------------------===


def train(args: argparse.Namespace) -> None:
    train_loader = get_train_data(args.train_batch_size)
    test_loader = get_test_data(args.test_batch_size)
    lenet = LeNet()
    optim = torch.optim.Adam(lenet.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        train_lenet(lenet, train_loader, optim, epoch)
        test_lenet(lenet, test_loader)
        print()

    print("Finished training.")

    args.save_to.parent.mkdir(parents=True, exist_ok=True)
    lenet.save_to(args.save_to)


def test(args: argparse.Namespace) -> None:
    test_loader = get_test_data(args.batch_size)
    lenet = LeNet.from_path(args.model_path)
    test_lenet(lenet, test_loader)


# ===-----------------------------------------------------------------------===
# Entrypoint
# ===-----------------------------------------------------------------------===


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
