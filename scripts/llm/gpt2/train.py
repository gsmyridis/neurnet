import argparse
import os
from itertools import islice
from time import perf_counter_ns
from typing import cast

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn

from neurnet.arch.llm.models.gpt2 import (
    GPT2_CONFIG_124M,
    GPT2MLXModel,
    GPT2ModelType,
    GPT2Tokenizer,
    evaluate_gpt2,
    train_gpt2,
)
from neurnet.arch.llm.types import Tokenizer
from neurnet.datasets import TinyShakespeareDataset
from neurnet.device import Device
from neurnet.nn import MLXLossFunction
from neurnet.utils import print_header
from neurnet.utils.data import MLXDataLoader

NANOS_IN_SEC = 1e9
HEADER_WIDTH = 120


def build_parser() -> argparse.ArgumentParser:
    DEFAULT_SEQUENCE_LENGTH = 256
    DEFAULT_BATCH_SIZE: int | None = None
    DEFAULT_BATCH_SIZE_TUNING_TRAINING_STEPS = 100
    DEFAULT_BATCH_SIZE_TUNING_MAX_EXPONENT = 5
    DEFAULT_LEARNING_RATE = 3e-4
    DEFAULT_PREFETCH_BATCHES = 64
    DEFAULT_PREFETCH_WORKERS = 4
    DEFAULT_SEED = 42

    parser = argparse.ArgumentParser(description="Train GPT-2 on Tiny Shakespeare.")
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Training batch size. Omit to tune for the best size.",
    )
    parser.add_argument(
        "--batch-size-tuning-training-steps",
        type=int,
        default=DEFAULT_BATCH_SIZE_TUNING_TRAINING_STEPS,
    )
    parser.add_argument(
        "--batch-size-tuning-max-exponent",
        type=int,
        default=DEFAULT_BATCH_SIZE_TUNING_MAX_EXPONENT,
    )
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument(
        "--prefetch-batches", type=int, default=DEFAULT_PREFETCH_BATCHES
    )
    parser.add_argument(
        "--prefetch-workers", type=int, default=DEFAULT_PREFETCH_WORKERS
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--verbose", action="store_true")
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
    print(f"  Verbose: {args.verbose}")


def make_model() -> GPT2MLXModel:
    return GPT2MLXModel(
        GPT2_CONFIG_124M,
        embd_pdrop=0.0,
        resid_pdrop=0.0,
        attn_pdrop=0.0,
    )


def make_dataloader(
    tokenizer: Tokenizer,
    sequence_len: int,
    batch_size: int,
    prefetch_batches: int,
    prefetch_workers: int,
) -> MLXDataLoader:

    dataset = TinyShakespeareDataset(
        root_dir=os.path.join("data", "shakespeare"),
        sequence_length=sequence_len,
        tokenizer=tokenizer,
        shuffle=True,
        batch_size=batch_size,
    )
    dataloader = dataset.to_mlx(
        prefetch_batches=prefetch_batches, prefetch_worker_threads=prefetch_workers
    )
    return dataloader


def count_tokens_in_dataloader(dataloader: MLXDataLoader) -> int:
    dataloader.reset()
    total_tokens = 0
    for inputs, _ in dataloader:
        total_tokens += inputs.size

    dataloader.reset()
    return total_tokens


def main():
    args = build_parser().parse_args()
    print_preamble(args)
    mx.random.seed(args.seed)

    # Create tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(
        GPT2ModelType.SMALL, cache_dir=os.path.join("models", "gpt2")
    )

    # Create model
    model = make_model()

    # Set device
    mlx_device = Device.gpu().to_mlx()
    mx.set_default_device(mlx_device)

    optimizer = optim.AdamW(learning_rate=args.learning_rate)
    loss_fn = nn.losses.cross_entropy
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else find_optimum_batch_size(
            tokenizer=tokenizer,
            loss_fn=loss_fn,
            optimizer=optimizer,
            sequence_len=args.sequence_length,
            prefetch_batches=args.prefetch_batches,
            prefetch_workers=args.prefetch_workers,
            n_steps=args.batch_size_tuning_training_steps,
            max_batch_exponent=args.batch_size_tuning_max_exponent,
        )
    )
    dataloader = make_dataloader(
        tokenizer,
        args.sequence_length,
        batch_size,
        args.prefetch_batches,
        args.prefetch_workers,
    )

    print_header("Training GPT2", "=", 120)
    train_gpt2(model, dataloader, optimizer, loss_fn=loss_fn, verbose=args.verbose)


def find_optimum_batch_size(
    tokenizer: Tokenizer,
    sequence_len: int,
    prefetch_batches: int,
    prefetch_workers: int,
    loss_fn: MLXLossFunction,
    optimizer: optim.Optimizer,
    max_batch_exponent: int,
    n_steps: int,
) -> int:
    print_header("Tuning batch-size", "=", HEADER_WIDTH)

    batch_sizes: list[int] = []
    throughputs_train: list[float] = []
    throughputs_eval: list[float] = []

    for exponent in range(max_batch_exponent + 1):
        batch_size = 2**exponent
        print("Batch size : ", batch_size)
        batch_sizes.append(batch_size)

        model = make_model()
        dataloader = make_dataloader(
            tokenizer,
            sequence_len,
            batch_size,
            prefetch_batches,
            prefetch_workers,
        )

        def loss_fn_inner(m: nn.Module, ins: mx.array, targs: mx.array) -> mx.array:
            logits = m(ins)  # (batch-size, token-sequence-length, vocab-size)
            loss = mx.mean(nn.losses.cross_entropy(logits, targs))
            return loss

        loss_and_grad_fn = nn.value_and_grad(model, loss_fn_inner)

        time_start_train = perf_counter_ns()
        tokens_train = 0
        for inputs, targets in islice(dataloader, n_steps):
            _, grads = loss_and_grad_fn(model, inputs, targets)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            tokens_train += inputs.size

        time_end_train = perf_counter_ns()

        time_start_eval = perf_counter_ns()
        _ = evaluate_gpt2(model, dataloader, loss_fn)
        time_end_eval = perf_counter_ns()
        tokens_eval = count_tokens_in_dataloader(dataloader)

        # Compute train, evaluation and total throughputs
        time_train = time_end_train - time_start_train
        throughput_train = int(NANOS_IN_SEC * tokens_train / time_train)
        throughputs_train.append(throughput_train)

        time_eval = time_end_eval - time_start_eval
        throughput_eval = int(NANOS_IN_SEC * tokens_eval / time_eval)
        throughputs_eval.append(throughput_eval)

        print(
            f"Training   : [ Tokens: {tokens_train}, Time: {time_train}ns, Throughput: {throughput_train} tokens/sec, Iterations: {n_steps} ]"
        )
        print(
            f"Evaluation : [ Tokens: {tokens_eval}, Time: {time_eval}ns, Throughput: {throughput_eval} tokens/sec ]"
        )
        print("-" * HEADER_WIDTH)

    batch_size_opt_idx = mx.argmax(mx.array(throughputs_train))
    batch_size_opt = mx.array(batch_sizes)[batch_size_opt_idx].item()
    print("Using optimum batch size:", batch_size_opt)
    print("=" * HEADER_WIDTH)

    return cast(int, batch_size_opt)


if __name__ == "__main__":
    main()
