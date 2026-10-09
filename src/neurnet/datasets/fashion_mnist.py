from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import mlx.core as mx
from mlx.data.datasets import load_fashion_mnist

from .types import Dataset, MLXDataLoader


@dataclass(frozen=True)
class FashionMNISTDataset(Dataset):
    N_CLASSES: ClassVar[int] = 10
    IMAGE_DIMS: ClassVar[tuple[int, int]] = (28, 28)
    IMAGE_SIZE: ClassVar[int] = IMAGE_DIMS[0] * IMAGE_DIMS[1]

    train: bool
    root_dir: str
    shuffle: bool
    batch_size: int

    @staticmethod
    def image_dims() -> tuple[int, int]:
        return FashionMNISTDataset.IMAGE_DIMS

    @staticmethod
    def n_classes() -> int:
        return FashionMNISTDataset.N_CLASSES

    def download_path(self) -> Path:
        return Path(self.root_dir) / "fashion_mnist"

    def download(self) -> Path:
        path = self.download_path()
        os.makedirs(path, exist_ok=True)
        _ = load_fashion_mnist(train=self.train, root=path)
        return path

    def to_mlx(
        self,
        prefetch_batches: int,
        prefetch_worker_threads: int,
    ) -> FashionMNISTMLXDataLoader:
        return FashionMNISTMLXDataLoader(
            train=self.train,
            root_dir=str(self.download_path()),
            shuffle=self.shuffle,
            batch_size=self.batch_size,
            prefetch_batches=prefetch_batches,
            prefetch_worker_threads=prefetch_worker_threads,
        )


class FashionMNISTMLXDataLoader(MLXDataLoader):
    def __init__(
        self,
        root_dir: str,
        train: bool,
        shuffle: bool = False,
        batch_size: int = 1,
        prefetch_batches: int = 4,
        prefetch_worker_threads: int = 2,
    ):
        buffer = load_fashion_mnist(train=train, root=root_dir)
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
