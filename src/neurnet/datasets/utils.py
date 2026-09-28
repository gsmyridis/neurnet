from collections.abc import Sequence
from pathlib import Path

import mlx.core as mx
import mlx.data as dx
from mlx.data import Buffer

from neurnet.arch.llm.types import Tokenizer


def text_file_to_buffer_mlx(
    path: str,
    tokenizer: Tokenizer,
    sequence_length: int,
    batch_size: int,
    shuffle: bool,
) -> Buffer:
    text = Path(path).read_text(encoding="utf-8")
    return text_to_buffer_mlx(
        text=text,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def text_to_buffer_mlx(
    text: str,
    tokenizer: Tokenizer,
    sequence_length: int,
    batch_size: int,
    shuffle: bool,
) -> Buffer:
    """Create next-token training batches from text with a tokenizer."""
    tokens = tokenizer.encode(text)
    return tokens_to_buffer_mlx(tokens, sequence_length, batch_size, shuffle)


def tokens_to_buffer_mlx(
    tokens: Sequence[int], sequence_length: int, batch_size: int, shuffle: bool
) -> Buffer:
    """Create next-token training batches from a token sequence.

    Each buffer element contains ``inputs`` and one-token-shifted ``targets``.
    The final sample retains any token pairs that do not fill a sequence.
    """
    tokens_array = mx.array(list(tokens))
    # We take the divmod of len(tokens_array) - 1 because we need the prediction
    # to be in the sequence.
    num_full_sequences, remainder = divmod(len(tokens_array) - 1, sequence_length)

    def make_batch(offset: int) -> dict[str, mx.array]:
        size = min(batch_size, num_full_sequences - offset)
        start = offset * sequence_length
        end = (offset + size) * sequence_length
        return {
            "inputs": tokens_array[start:end].reshape(size, sequence_length),
            "targets": tokens_array[start + 1 : end + 1].reshape(size, sequence_length),
        }

    batches = [
        make_batch(offset) for offset in range(0, num_full_sequences, batch_size)
    ]
    if remainder:
        start = num_full_sequences * sequence_length
        batches.append(
            {
                "inputs": tokens_array[start : start + remainder].reshape(1, remainder),
                "targets": tokens_array[start + 1 : start + remainder + 1].reshape(
                    1, remainder
                ),
            }
        )

    buffer = dx.buffer_from_vector(batches)

    if shuffle:
        buffer = buffer.shuffle()

    return buffer
