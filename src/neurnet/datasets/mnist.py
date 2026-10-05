from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from mlx.data.datasets import load_mnist

from .types import Dataset, MLXDataLoader

MNIST_N_CLASSES = 10
MNIST_IMAGE_DIMS = (28, 28)
MNIST_IMAGE_SIZE = MNIST_IMAGE_DIMS[0] * MNIST_IMAGE_DIMS[1]


@dataclass(frozen=True)
class MNISTDataset(Dataset):
    train: bool
    root_dir: str
    shuffle: bool
    batch_size: int

    @staticmethod
    def image_dims() -> tuple[int, int]:
        return MNIST_IMAGE_DIMS

    @staticmethod
    def n_classes() -> int:
        return MNIST_N_CLASSES

    def download_path(self) -> Path:
        return Path(self.root_dir) / "msist" / str(self.train)

    def download(self) -> Path:
        path = self.download_path()
        os.makedirs(path, exist_ok=True)
        _ = load_mnist(train=self.train, root=path)
        return path

    def to_mlx(
        self,
        prefetch_batches: int,
        prefetch_worker_threads: int,
    ) -> MNISTMLXDataLoader:
        return MNISTMLXDataLoader(
            train=self.train,
            root_dir=self.root_dir,
            shuffle=self.shuffle,
            batch_size=self.batch_size,
            prefetch_batches=prefetch_batches,
            prefetch_worker_threads=prefetch_worker_threads,
        )


class MNISTMLXDataLoader(MLXDataLoader):
    def __init__(
        self,
        root_dir: str,
        train: bool,
        shuffle: bool = False,
        batch_size: int = 1,
        prefetch_batches: int = 4,
        prefetch_worker_threads: int = 2,
    ):
        buffer = load_mnist(train=train, root=root_dir)
        if shuffle:
            buffer = buffer.shuffle()

        self._stream = (
            buffer.to_stream()
            .key_transform("image", lambda x: x.astype("float32").reshape(-1) / 255.0)
            .batch(batch_size)
            .prefetch(prefetch_batches, prefetch_worker_threads)
        )

    def __next__(self) -> tuple[mx.array, mx.array]:
        next_sample = next(self._stream)
        return mx.array(next_sample["image"]), mx.array(next_sample["label"])

    def reset(self) -> None:
        """Resets the iterator."""
        self._stream.reset()
