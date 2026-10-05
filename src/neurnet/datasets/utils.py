import shutil
import tempfile
import urllib.request
import zipfile
from collections.abc import Sequence
from pathlib import Path

import mlx.core as mx
import mlx.data as dx
from mlx.data import Buffer

from neurnet.arch.llm.types import Tokenizer

# ===--------------------------------------------------------------------------===
# Text to buffer of tokens
# ===--------------------------------------------------------------------------===


def text_file_to_buffer_mlx(
    path: str,
    tokenizer: Tokenizer,
    sequence_length: int,
    batch_size: int,
    shuffle: bool,
    *,
    drop_last: bool = False,
) -> Buffer:
    text = Path(path).read_text(encoding="utf-8")
    return text_to_buffer_mlx(
        text=text,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )


def text_to_buffer_mlx(
    text: str,
    tokenizer: Tokenizer,
    sequence_length: int,
    batch_size: int,
    shuffle: bool,
    *,
    drop_last: bool = False,
) -> Buffer:
    """Create next-token training batches from text with a tokenizer."""
    tokens = tokenizer.encode(text)
    return tokens_to_buffer_mlx(
        tokens, sequence_length, batch_size, shuffle, drop_last=drop_last
    )


def tokens_to_buffer_mlx(
    tokens: Sequence[int],
    sequence_length: int,
    batch_size: int,
    shuffle: bool,
    *,
    drop_last: bool = False,
) -> Buffer:
    """Create next-token training batches from a token sequence.

    Each buffer element contains ``inputs`` and one-token-shifted ``targets``.
    The final sample retains any token pairs that do not fill a sequence.
    With ``drop_last=True``, omit incomplete batches and short sequences so
    every input and target has shape ``(batch_size, sequence_length)``.
    """
    tokens_array = mx.array(list(tokens))
    # We take the divmod of len(tokens_array) - 1 because we need the prediction
    # to be in the sequence.
    num_full_sequences, remainder = divmod(len(tokens_array) - 1, sequence_length)
    if drop_last:
        num_full_sequences -= num_full_sequences % batch_size

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
    if remainder and not drop_last:
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


# ===--------------------------------------------------------------------------===
# Download and unzip file from URL
# ===--------------------------------------------------------------------------===


def download_unzip(
    url: str,
    root_dir: str | Path,
    extracted_path: str | Path,
) -> Path:
    """Download a zip and extract it under ``root_dir / extracted_path``.

    The archive's internal directory structure is preserved. For example, a
    flat archive extracted with ``root_dir="data"`` and
    ``extracted_path="smm_spam_collection"`` places its files in
    ``data/smm_spam_collection``.

    If the destination directory already exists, it is returned without
    downloading the archive again.
    """
    destination = Path(root_dir) / extracted_path
    if destination.exists():
        if destination.is_dir():
            return destination
        raise FileExistsError(
            f"Extraction destination is not a directory: {destination}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)

    # Download and extract in a temporary directory, then move the completed
    # contents into place so a failed download does not leave a partial target.
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary_dir:
        temporary_path = Path(temporary_dir)
        zip_path = temporary_path / "download.zip"
        staging_path = temporary_path / "contents"
        staging_path.mkdir()

        with urllib.request.urlopen(url) as response, zip_path.open("wb") as out_file:
            shutil.copyfileobj(response, out_file)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(staging_path)

        staging_path.rename(destination)

    return destination
