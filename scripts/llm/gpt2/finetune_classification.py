import argparse
import os
import random
from functools import partial
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import mlx.core as mx
from mlx import nn
from mlx.optimizers import AdamW, Optimizer
from tqdm import tqdm

from neurnet.arch.llm.models.gpt2 import (
    GPT2MLXModel,
    GPT2ModelType,
    GPT2Tokenizer,
    iter_gpt2_train_steps,
)
from neurnet.datasets import MLXDataLoader, SMSSpamCollection
from neurnet.device import Device
from neurnet.utils.fmt import print_header

HEADER_WIDTH = 120


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune GPT2 model for binary classification of spam messages."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Fraction of the dataset used for training.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help="Fraction of the dataset used for validation.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.1,
        help="Fraction of the dataset used for testing.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16", "mixed"),
        default="float32",
        help="Mixed uses bfloat16 compute with float32 master weights and AdamW state.",
    )
    parser.add_argument(
        "--fine-tune",
        choices=("head", "last-layer", "full"),
        default="last-layer",
        help="Train the head only, the head plus final transformer layer and layer norm, or the full model.",
    )
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--async-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlap training-step submission and GPU execution (default: enabled).",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show per-epoch progress, batch rate, and ETA (default: enabled).",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show per-epoch elapsed time and example throughput.",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        metavar="PATH",
        help="Save trained model weights to a .safetensors or .npz file.",
    )
    return parser


def print_preamble(args: argparse.Namespace) -> None:
    print_header("GPT-2 fine-tuning configuration", "=", HEADER_WIDTH)
    print(f"  Fine-tune: {args.fine_tune}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Train fraction: {args.train_fraction}")
    print(f"  Validation fraction: {args.validation_fraction}")
    print(f"  Test fraction: {args.test_fraction}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Seed: {args.seed}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Device: {args.device}")
    print(f"  Dtype: {args.dtype}")
    print(f"  Compile training step: {args.compile}")
    print(f"  Async evaluation: {args.async_eval}")
    print(f"  Epoch progress: {args.progress}")
    print(f"  Verbose: {args.verbose}")
    print(
        f"  Save path: {args.save_path if args.save_path is not None else 'disabled'}"
    )


def calculate_classification_metrics(
    model: nn.Module, data_loader: MLXDataLoader
) -> tuple[float, float]:
    model.eval()
    data_loader.reset()
    total_loss = mx.array(0.0)
    correct_preds = mx.array(0)
    n_examples = 0

    for inputs, labels in data_loader:
        logits = model(inputs)[:, -1, :].astype(mx.float32)
        batch_loss = nn.losses.cross_entropy(logits, labels, reduction="sum")
        batch_correct = (mx.argmax(logits, axis=1) == labels).sum()
        batch_examples = labels.size
        total_loss += batch_loss
        correct_preds += batch_correct
        n_examples += batch_examples
        mx.eval(total_loss, correct_preds)

    if n_examples == 0:
        raise ValueError("cannot calculate metrics on an empty dataloader")

    loss = (total_loss / n_examples).item()
    accuracy = (correct_preds / n_examples).item()
    assert isinstance(loss, float), type(loss)
    assert isinstance(accuracy, float), type(accuracy)
    return loss, accuracy


def plot_classification_history(
    training_losses: list[float],
    validation_losses: list[float],
    validation_accuracies: list[float],
    training_accuracies: list[float],
    test_loss: float,
    test_accuracy: float,
) -> None:
    epochs = range(1, len(training_losses) + 1)
    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(12, 5))

    loss_axis.plot(epochs, training_losses, label="Training loss")
    loss_axis.plot(epochs, validation_losses, label="Validation loss")
    loss_axis.axhline(test_loss, color="red", linestyle="--", label="Test loss")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Cross-entropy loss")
    loss_axis.set_title("Loss")
    loss_axis.legend()
    loss_axis.grid(True)

    accuracy_axis.plot(epochs, validation_accuracies, label="Validation accuracy")
    accuracy_axis.plot(epochs, training_accuracies, label="Training accuracy")
    accuracy_axis.axhline(
        test_accuracy, color="red", linestyle="--", label="Test accuracy"
    )
    accuracy_axis.set_xlabel("Epoch")
    accuracy_axis.set_ylabel("Accuracy")
    accuracy_axis.set_ylim(0.0, 1.0)
    accuracy_axis.set_title("Accuracy")
    accuracy_axis.legend()
    accuracy_axis.grid(True)

    figure.suptitle("GPT-2 classification training")
    figure.tight_layout()
    plt.show()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.save_path is not None:
        args.save_path = args.save_path.expanduser()
        if args.save_path.suffix not in (".safetensors", ".npz"):
            parser.error("--save-path must end with '.safetensors' or '.npz'")

    print_preamble(args)
    print_header("", "=", HEADER_WIDTH)

    num_classes = 2

    # Set the device
    device = Device.gpu() if args.device == "gpu" else Device.cpu()
    mx.set_default_device(device.to_mlx())
    mx.random.seed(args.seed)
    random.seed(args.seed)

    # Load the model
    model_dir = os.path.join("models", "gpt2")
    model_type = GPT2ModelType.SMALL
    tokenizer = GPT2Tokenizer.from_pretrained(model_type, model_dir)
    model = GPT2MLXModel.from_pretrained(model_type, model_dir)
    model_config = model.config()

    # Set dtype
    if args.dtype == "bfloat16":
        model.set_dtype(mx.bfloat16)
        # TODO: add mixed with master weights

    # We freeze the model parameters so that they are no longer trainable.
    # Then we replace the language modelling head with a new linear layer
    # that maps embeddings to classes. This new linear layer is trainable
    # by default.
    model.freeze()
    model.lm_head = nn.Linear(input_dims=model_config.emb_dim, output_dims=num_classes)
    model.lm_head.set_dtype(model.transformer.wte.weight.dtype)
    # Technically only fine-tuning the last layer is sufficient, however
    # empirically, the results are better if we also fine-tune some additional
    # layers, specifically the latter ones, which learn higher-level linguistic
    # patterns.
    if args.fine_tune == "last-layer":
        model.transformer.ln_f.unfreeze()
        model.transformer.h[-1].unfreeze()
    # We additionally, we can try to fine-tune the whole model and assess
    # the effectiveness.
    if args.fine_tune == "full":
        model.unfreeze()

    # Load the data loaders
    data_root_dir = "data"
    dataset = SMSSpamCollection(
        root_dir=data_root_dir,
        tokenizer=tokenizer,
        max_sequence_len=model_config.context_length,
        batch_size=args.batch_size,
        random_state=args.seed,
        drop_last=True,
    )
    train_loader, validation_loader, test_loader = dataset.train_test_split(
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )

    # Choose optimizer
    optimizer = AdamW(learning_rate=args.learning_rate)

    # Perform the finetuning
    print("=" * HEADER_WIDTH)
    (
        training_losses,
        validation_losses,
        validation_accuracies,
        training_accuracies,
    ) = fine_tune_gpt2_for_classification(
        epochs=args.epochs,
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        optimizer=optimizer,
        compiled=args.compile,
        async_eval=args.async_eval,
        progress=args.progress,
        verbose=args.verbose,
    )

    test_loss, test_accuracy = calculate_classification_metrics(model, test_loader)
    print(f"Final test loss: {test_loss:.4f}, accuracy: {test_accuracy:.2%}")

    # Save the model
    if args.save_path is not None:
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_weights(str(args.save_path))
        print(f"Saved fine-tuned model weights to {args.save_path}")

    # Plot the results
    plot_classification_history(
        training_losses,
        validation_losses,
        validation_accuracies,
        training_accuracies,
        test_loss,
        test_accuracy,
    )


def fine_tune_gpt2_for_classification(
    epochs: int,
    model: GPT2MLXModel,
    train_loader: MLXDataLoader,
    validation_loader: MLXDataLoader,
    optimizer: Optimizer,
    *,
    compiled: bool = True,
    async_eval: bool = True,
    progress: bool = True,
    verbose: bool = True,
) -> tuple[list[float], list[float], list[float], list[float]]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    def loss_fn(model: GPT2MLXModel, inputs: mx.array, labels: mx.array) -> mx.array:
        logits = model(inputs)[:, -1, :].astype(mx.float32)
        return nn.losses.cross_entropy(logits, labels, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    def step(inputs: mx.array, labels: mx.array) -> mx.array:
        loss, grads = loss_and_grad(model, inputs, labels)
        optimizer.update(model, grads)
        return loss

    state = [model.state, optimizer.state, mx.random.state]
    train_step = (
        partial(mx.compile, inputs=state, outputs=state)(step) if compiled else step
    )
    training_losses: list[float] = []
    validation_losses: list[float] = []
    validation_accuracies: list[float] = []
    training_accuracies: list[float] = []

    for epoch in range(epochs):
        train_loader.reset()
        model.train()
        epoch_start = perf_counter()
        epoch_examples = 0

        with tqdm(
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{epochs}",
            unit="batch",
            mininterval=0.5,
            dynamic_ncols=True,
            disable=not progress,
            leave=False,
        ) as bar:
            for _, examples in iter_gpt2_train_steps(
                train_step,
                train_loader,
                state,
                async_eval=async_eval,
            ):
                epoch_examples += examples
                bar.update()

        if epoch_examples == 0:
            raise ValueError("cannot fine-tune on an empty dataloader")

        training_loss, training_accuracy = calculate_classification_metrics(
            model, train_loader
        )
        validation_loss, validation_accuracy = calculate_classification_metrics(
            model, validation_loader
        )
        training_losses.append(training_loss)
        validation_losses.append(validation_loss)
        validation_accuracies.append(validation_accuracy)
        training_accuracies.append(training_accuracy)

        log = (
            f"[Epoch {epoch + 1} / {epochs}]: "
            f"Train loss {training_loss:.4f}, "
            f"Validation loss {validation_loss:.4f}, "
            f"Train accuracy {training_accuracy:.2%}, "
            f"Validation accuracy {validation_accuracy:.2%}"
        )
        if verbose:
            elapsed = perf_counter() - epoch_start
            log += (
                f", Elapsed time: {elapsed:.1f}s, "
                f"{epoch_examples / elapsed:.0f} examples/sec"
            )
        print(log)

    train_loader.reset()

    return (
        training_losses,
        validation_losses,
        validation_accuracies,
        training_accuracies,
    )


if __name__ == "__main__":
    main()
