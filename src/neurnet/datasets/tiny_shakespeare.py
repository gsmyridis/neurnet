from pathlib import Path
from urllib.request import urlretrieve

import mlx.core as mx
from mlx.data import Stream

from neurnet.arch.llm import Tokenizer
from neurnet.utils.data import Dataset, MLXDataLoader

from .utils import text_file_to_buffer_mlx


class TinyShakespeareDataset(Dataset):
    SOURCE_LINK = (
        "https://raw.githubusercontent.com/karpathy/char-rnn/"
        "refs/heads/master/data/tinyshakespeare/input.txt"
    )
    _FILENAME = "tiny_shakespeare.txt"

    def __init__(
        self,
        root_dir: str,
        tokenizer: Tokenizer,
        sequence_length: int,
        shuffle: bool,
        batch_size: int,
        *,
        drop_last: bool = False,
    ):
        self.root_dir = root_dir
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.drop_last = drop_last

    def download(self) -> Path:
        """Download Tiny Shakespeare into root dir unless it already exists."""
        path = Path(self.root_dir) / self._FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)

        if not path.exists():
            urlretrieve(self.SOURCE_LINK, path)

        return path

    def to_mlx(
        self,
        prefetch_batches: int,
        prefetch_worker_threads: int,
    ) -> MLXDataLoader:
        """Returns an MLX dataloader."""
        path = self.download()
        return TinyShakespeareMLXDataLoader(
            path=str(path),
            sequence_length=self.sequence_length,
            tokenizer=self.tokenizer,
            shuffle=self.shuffle,
            batch_size=self.batch_size,
            prefetch_batches=prefetch_batches,
            prefetch_worker_threads=prefetch_worker_threads,
            drop_last=self.drop_last,
        )


class TinyShakespeareMLXDataLoader(MLXDataLoader):
    def __init__(
        self,
        path: str,
        tokenizer: Tokenizer,
        sequence_length: int,
        shuffle: bool,
        batch_size: int,
        prefetch_batches: int,
        prefetch_worker_threads: int,
        *,
        drop_last: bool = False,
    ):

        buffer = text_file_to_buffer_mlx(
            path=path,
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        self._num_batches = len(buffer)
        self._stream = buffer.to_stream().prefetch(
            prefetch_batches, prefetch_worker_threads
        )

    def __len__(self) -> int:
        return self._num_batches

    def __next__(self) -> tuple[mx.array, mx.array]:
        next_sample = next(self._stream)
        return mx.array(next_sample["inputs"]), mx.array(next_sample["targets"])

    def stream(self) -> Stream:
        return self._stream
