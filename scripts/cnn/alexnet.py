import argparse
import copy
import os
import time
from math import isfinite
from pathlib import Path
from typing import cast

import kagglehub
import torch
from torch import nn
from torch.optim import SGD, Optimizer
from torch.types import Device
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import AlexNet_Weights

from neurnet.cnn import AlexNet
from neurnet.utils.data import SizedTorchDataLoader
from neurnet.utils.gpu import get_torch_device

# ===-----------------------------------------------------------------------===
# Constants
# ===-----------------------------------------------------------------------===

_SECS_IN_MIN = 60

_HYMENOPTERA_DATA = "hymenoptera_data"
_DATA_DIR = os.path.join("./data", _HYMENOPTERA_DATA)

_WEIGHTS = AlexNet_Weights.IMAGENET1K_V1
_WEIGHTS_TRANSFORMS = _WEIGHTS.transforms()

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
    parser = argparse.ArgumentParser(
        description="Fine-tune or test AlexNet on the Hymenoptera dataset."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    finetune_parser = subparsers.add_parser(
        "finetune", help="Fine-tune a pretrained AlexNet model."
    )
    finetune_parser.add_argument("--epochs", type=positive_int, default=10)
    finetune_parser.add_argument("--learning-rate", type=positive_float, default=0.0001)
    finetune_parser.add_argument("--train-batch-size", type=positive_int, default=8)
    finetune_parser.add_argument("--test-batch-size", type=positive_int, default=8)
    finetune_parser.add_argument("--save-to", type=Path)
    finetune_parser.set_defaults(handler=finetune)

    test_parser = subparsers.add_parser("test", help="Test a saved AlexNet model.")
    test_parser.add_argument("--model-path", type=Path, required=True)
    test_parser.add_argument("--batch-size", type=positive_int, default=8)
    test_parser.set_defaults(handler=test)

    return parser


# ===-----------------------------------------------------------------------===
# Dataloaders
# ===-----------------------------------------------------------------------===


def _download_dataset(path: str) -> str:
    dir_path = kagglehub.dataset_download("ajayrana/hymenoptera-data", output_dir=path)
    return os.path.join(dir_path, _HYMENOPTERA_DATA)


def get_train_data(batch_size: int) -> tuple[list[str], SizedTorchDataLoader]:
    path = _download_dataset(_DATA_DIR)
    train_set = datasets.ImageFolder(
        os.path.join(path, "train"),
        _WEIGHTS_TRANSFORMS,
    )

    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    return train_set.classes, SizedTorchDataLoader(loader, len(train_set))


def get_test_data(batch_size: int) -> tuple[list[str], SizedTorchDataLoader]:
    path = _download_dataset(_DATA_DIR)
    test_data = datasets.ImageFolder(os.path.join(path, "val"), _WEIGHTS_TRANSFORMS)
    loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return test_data.classes, SizedTorchDataLoader(loader, len(test_data))


# ===-----------------------------------------------------------------------===
# Load pre-trained AlexNet model
# ===-----------------------------------------------------------------------===


def load_modified_alexnet(n_classes: int) -> AlexNet:
    """Load pretrained AlexNet weights and replace the ImageNet classifier."""
    alexnet = AlexNet()
    alexnet.load_state_dict(_WEIGHTS.get_state_dict(progress=True, check_hash=True))
    alexnet.classifier[6] = nn.Linear(4096, n_classes)
    return alexnet


def evaluate_model(
    model: AlexNet,
    data_loader: SizedTorchDataLoader,
    loss_func: nn.Module,
    device: Device,
) -> tuple[float, float]:
    model.eval()
    loss = 0.0
    successes = 0

    with torch.no_grad():
        for images, targets in data_loader.loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss_curr = loss_func(outputs, targets)
            predictions = outputs.argmax(dim=1)

            loss += loss_curr.item() * images.size(0)
            successes += (predictions == targets).sum().item()

    return loss / data_loader.len, successes / data_loader.len


# ===-----------------------------------------------------------------------===
# Finetuning
# ===-----------------------------------------------------------------------===


def finetune_model(
    model: AlexNet,
    train_loader: SizedTorchDataLoader,
    test_loader: SizedTorchDataLoader,
    loss_func: nn.Module,
    optim: Optimizer,
    epochs: int,
    device: Device,
) -> AlexNet:

    model_weights = copy.deepcopy(model.state_dict())
    accuracy = float("-inf")

    dloaders = {
        "train": train_loader,
        "test": test_loader,
    }

    for e in range(epochs):
        # For each epoch we run through the training and validation set
        for dset, dloader in dloaders.items():
            # Set model to training mode (i.e. trainable weights),
            # else set model to validation mode.
            is_training = dset == "train"
            model.train() if is_training else model.eval()

            loss = 0.0
            successes = 0

            # Iterate over the (training / validation) data.
            for images, targets in dloader.loader:
                images = images.to(device)
                targets = targets.to(device)

                with torch.set_grad_enabled(is_training):
                    ops = model(images)
                    _, preds = torch.max(ops, 1)
                    loss_curr = loss_func(ops, targets)

                    # Backward pass only if in training mode
                    if is_training:
                        optim.zero_grad()
                        loss_curr.backward()
                        optim.step()

                loss += loss_curr.item() * images.size(0)
                successes += torch.sum(preds == targets.data)

            loss_epoch = loss / dloader.len
            successes = cast(torch.Tensor, successes)
            accuracy_epoch = successes.to(torch.float32) / dloader.len

            print(
                f"[Epoch: {e + 1} / {epochs}, {dset.capitalize()}] Loss: {loss_epoch:.3f}, Accuracy: {100 * accuracy_epoch:.2f}%"
            )

            # Create a checkpoint when the accuracy is improved
            # in the current epoch.
            if dset == "test" and accuracy_epoch > accuracy:
                accuracy = accuracy_epoch
                model_weights = copy.deepcopy(model.state_dict())

        print()

    print(f"Best validation set accuracy: {100 * accuracy:.2f}%")

    # load the best model version (weights)
    model.load_state_dict(model_weights)
    return model


# ===-----------------------------------------------------------------------===
# Fine-tune / Test
# ===-----------------------------------------------------------------------===


def finetune(args: argparse.Namespace) -> None:
    train_classes, train_loader = get_train_data(args.train_batch_size)
    test_classes, test_loader = get_test_data(args.test_batch_size)
    if train_classes != test_classes:
        raise ValueError("training and test datasets must have the same classes")

    device = get_torch_device()
    alexnet = load_modified_alexnet(len(train_classes)).to(device)

    loss_func = nn.CrossEntropyLoss()
    optim = SGD(alexnet.parameters(), lr=args.learning_rate)

    # Train (fine-tune) and validate the model
    start = time.time()
    finetuned_alexnet = finetune_model(
        model=alexnet,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_func=loss_func,
        optim=optim,
        epochs=args.epochs,
        device=device,
    )

    if args.save_to is not None:
        args.save_to.parent.mkdir(parents=True, exist_ok=True)
        finetuned_alexnet.save_to(args.save_to)

    time_delta = time.time() - start
    mins = time_delta // _SECS_IN_MIN
    secs = time_delta % _SECS_IN_MIN
    print(f"Training finished in {mins} mins {secs:.2f} secs")


def test(args: argparse.Namespace) -> None:
    classes, test_loader = get_test_data(args.batch_size)
    device = get_torch_device()

    alexnet = AlexNet(num_classes=len(classes))
    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    alexnet.load_state_dict(state_dict)
    alexnet.to(device)

    loss, accuracy = evaluate_model(
        model=alexnet,
        data_loader=test_loader,
        loss_func=nn.CrossEntropyLoss(),
        device=device,
    )
    print(f"[Test] Loss: {loss:.3f}, Accuracy: {100 * accuracy:.2f}%")


# ===-----------------------------------------------------------------------===
# Entrypoint
# ===-----------------------------------------------------------------------===


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
