from __future__ import annotations

from abc import ABC
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Self

import mlx.core as mx
import torch

# ===-----------------------------------------------------------------------===
# Dataset
# ===-----------------------------------------------------------------------===


class Dataset(ABC):
    def to_mlx(
        self,
        prefetch_batches: int,
        prefetch_worker_threads: int,
    ) -> MLXDataLoader:
        """Returns an MLX dataloader."""
        raise NotImplementedError("'to_mlx' is not implemented.")

    def to_torch(self) -> torch.utils.data.DataLoader:
        """Returns a Torch dataloader."""
        raise NotImplementedError("'to_torch' is not implemented.")


# ===-----------------------------------------------------------------------===
# MLX Dataloader
# ===-----------------------------------------------------------------------===


class MLXDataLoader(Iterator, Iterable):
    def __iter__(self) -> Self:
        return self

    def __next__(self) -> tuple[mx.array, mx.array]:
        raise NotImplementedError("'__next__' has not been implemented.")

    def reset(self) -> None:
        raise NotImplementedError("'reset' has not been implemented.")


# ===-----------------------------------------------------------------------===
# Torch Dataloader
# ===-----------------------------------------------------------------------===


@dataclass(frozen=True)
class SizedTorchDataLoader:
    loader: torch.utils.data.DataLoader
    len: int

    def __iter__(self):
        return iter(self.loader)
