import argparse
import os
from collections.abc import Callable
from functools import partial
from itertools import islice
from pathlib import Path
from time import perf_counter_ns

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn

from neurnet.arch.llm.models.gpt2 import (
    GPT2_CONFIG_124M,
    GPT2MLXModel,
    GPT2ModelType,
    GPT2Tokenizer,
    iter_gpt2_train_steps,
    make_gpt2_train_step,
    train_gpt2,
)
from neurnet.arch.llm.types import Tokenizer
from neurnet.datasets import MLXDataLoader, TinyShakespeareDataset
from neurnet.device import Device
from neurnet.nn import MLXLossFunction
from neurnet.utils.fmt import print_header

NANOS_IN_SEC = 1e9
HEADER_WIDTH = 120


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train GPT-2 on Tiny Shakespeare.")
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size. Omit to tune for the best size.",
    )
    parser.add_argument("--batch-size-tuning-training-steps", type=int, default=10)
    parser.add_argument("--batch-size-tuning-max-exponent", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--prefetch-batches", type=int, default=64)
    parser.add_argument("--prefetch-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16", "mixed"),
        default="float32",
        help="Mixed uses bfloat16 compute with float32 master weights and AdamW state.",
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
        help="Show per-epoch loss, total time and toke throughput.",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        metavar="PATH",
        help="Save trained model weights to a .safetensors or .npz file.",
    )
    return parser


def print_preamble(args: argparse.Namespace) -> None:

    print_header("GPT-2 training configuration", "=", HEADER_WIDTH)
    print(f"  Sequence length: {args.sequence_length}")
    print(f"  Batch size: {args.batch_size if args.batch_size is not None else 'auto'}")
    if args.batch_size is None:
        print(
            f"  Batch-size tuning training steps: {args.batch_size_tuning_training_steps}"
        )
        print(
            f"  Batch-size tuning max exponent: {args.batch_size_tuning_max_exponent}"
        )
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Prefetch batches: {args.prefetch_batches}")
    print(f"  Prefetch workers: {args.prefetch_workers}")
    print(f"  Seed: {args.seed}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Dtype: {args.dtype}")
    print(f"  Compile training step: {args.compile}")
    print(f"  Async evaluation: {args.async_eval}")
    print(f"  Epoch progress: {args.progress}")
    print(f"  Verbose: {args.verbose}")
    print(
        f"  Save path: {args.save_path if args.save_path is not None else 'disabled'}"
    )


def make_model(dtype: mx.Dtype = mx.float32) -> GPT2MLXModel:
    model = GPT2MLXModel(
        GPT2_CONFIG_124M,
        embd_pdrop=0.0,
        resid_pdrop=0.0,
        attn_pdrop=0.0,
    )
    if dtype != mx.float32:
        model.set_dtype(dtype)
        model.transformer.wte.weight = model.lm_head.weight
    return model


def make_dataloader(
    tokenizer: Tokenizer,
    sequence_len: int,
    batch_size: int,
    prefetch_batches: int,
    prefetch_workers: int,
    *,
    drop_last: bool = False,
) -> MLXDataLoader:

    dataset = TinyShakespeareDataset(
        root_dir=os.path.join("data", "shakespeare"),
        sequence_length=sequence_len,
        tokenizer=tokenizer,
        shuffle=True,
        batch_size=batch_size,
        drop_last=drop_last,
    )
    dataloader = dataset.to_mlx(
        prefetch_batches=prefetch_batches, prefetch_worker_threads=prefetch_workers
    )
    return dataloader


def make_optimizer(learning_rate: float) -> optim.AdamW:
    return optim.AdamW(learning_rate=learning_rate)


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.save_path is not None:
        args.save_path = args.save_path.expanduser()
        if args.save_path.suffix not in (".safetensors", ".npz"):
            parser.error("--save-path must end with '.safetensors' or '.npz'")
    print_preamble(args)
    mx.random.seed(args.seed)

    # Create tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(
        GPT2ModelType.SMALL, cache_dir=os.path.join("models", "gpt2")
    )

    # Set device
    mlx_device = Device.gpu().to_mlx()
    mx.set_default_device(mlx_device)

    optimizer_fn = partial(make_optimizer, args.learning_rate)
    loss_fn = nn.losses.cross_entropy
    dtype = mx.float32 if args.dtype == "float32" else mx.bfloat16
    master_weights = args.dtype == "mixed"
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else find_optimum_batch_size(
            tokenizer=tokenizer,
            loss_fn=loss_fn,
            optimizer_fn=optimizer_fn,
            sequence_len=args.sequence_length,
            prefetch_batches=args.prefetch_batches,
            prefetch_workers=args.prefetch_workers,
            n_steps=args.batch_size_tuning_training_steps,
            max_batch_exponent=args.batch_size_tuning_max_exponent,
            dtype=dtype,
            compiled=args.compile,
            master_weights=master_weights,
            async_eval=args.async_eval,
        )
    )
    model = make_model(dtype)
    optimizer = optimizer_fn()
    dataloader = make_dataloader(
        tokenizer,
        args.sequence_length,
        batch_size,
        args.prefetch_batches,
        args.prefetch_workers,
    )

    print_header("Training GPT2", "=", 120)
    train_gpt2(
        model,
        dataloader,
        optimizer,
        loss_fn=loss_fn,
        epochs=args.epochs,
        verbose=args.verbose,
        compiled=args.compile,
        master_weights=master_weights,
        async_eval=args.async_eval,
        progress=args.progress,
    )
    if args.save_path is not None:
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_weights(str(args.save_path))
        print(f"Saved trained model weights to {args.save_path}")


def find_optimum_batch_size(
    tokenizer: Tokenizer,
    sequence_len: int,
    prefetch_batches: int,
    prefetch_workers: int,
    loss_fn: MLXLossFunction,
    optimizer_fn: Callable[[], optim.Optimizer],
    max_batch_exponent: int,
    n_steps: int,
    dtype: mx.Dtype,
    compiled: bool = True,
    master_weights: bool = False,
    async_eval: bool = True,
) -> int:
    if n_steps <= 0:
        raise ValueError("batch-size tuning steps must be positive")
    if max_batch_exponent < 0:
        raise ValueError("batch-size tuning max exponent cannot be negative")
    print_header("Tuning batch-size", "=", HEADER_WIDTH)

    results: list[tuple[int, float]] = []

    for exponent in range(max_batch_exponent + 1):
        batch_size = 2**exponent
        print("Batch size : ", batch_size)
        model = make_model(dtype)
        dataloader = make_dataloader(
            tokenizer,
            sequence_len,
            batch_size,
            prefetch_batches,
            prefetch_workers,
            drop_last=True,
        )

        optimizer = optimizer_fn()
        step = make_gpt2_train_step(
            model,
            optimizer,
            loss_fn,
            compiled=compiled,
            master_weights=master_weights,
        )
        model.train()
        state = [model.state, optimizer.state, mx.random.state]

        # Pay the first-use compilation and allocation costs before measuring.
        for _ in iter_gpt2_train_steps(
            step, islice(dataloader, 2), state, async_eval=async_eval
        ):
            pass
        dataloader.reset()
        mx.reset_peak_memory()

        time_start_train = perf_counter_ns()
        tokens_train = 0
        actual_steps = 0
        for _, tokens in iter_gpt2_train_steps(
            step, islice(dataloader, n_steps), state, async_eval=async_eval
        ):
            tokens_train += tokens
            actual_steps += 1

        time_end_train = perf_counter_ns()
        if actual_steps == 0:
            raise ValueError(f"no full batches for batch size {batch_size}")
        time_train = time_end_train - time_start_train
        throughput_train = NANOS_IN_SEC * tokens_train / time_train
        results.append((batch_size, throughput_train))
        peak_gib = mx.get_peak_memory() / 2**30

        print(
            f"Training: {tokens_train} tokens, {actual_steps} steps, "
            f"{throughput_train:.0f} tokens/sec, {peak_gib:.2f} GiB peak"
        )
        print("-" * HEADER_WIDTH)

    batch_size_opt = max(results, key=lambda result: result[1])[0]
    print("Using optimum batch size:", batch_size_opt)

    return batch_size_opt


if __name__ == "__main__":
    main()
