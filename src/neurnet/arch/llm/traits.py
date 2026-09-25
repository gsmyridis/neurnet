from abc import ABC, abstractmethod
from collections.abc import Sequence

import mlx.core as mx
import mlx.nn
import torch

from neurnet.arch.llm.config import LanguageModelConfig
from neurnet.arch.llm.kv_cache import KVCache

type LanguageModel = LanguageModelTorch | LanguageModelMLX
type TensorType = torch.Tensor | mx.array


class Tokenizer(ABC):
    @abstractmethod
    def encode(self, text: str) -> Sequence[int]:
        raise NotImplementedError("'encode' is not implemented")

    @abstractmethod
    def decode(self, ids: Sequence[int]) -> str:
        raise NotImplementedError("'decode' is not implemented")


class LanguageModelTorch(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self, idx: torch.Tensor, cache: KVCache[torch.Tensor] | None = None
    ) -> torch.Tensor:
        raise NotImplementedError("'forward' is not implemented")

    @abstractmethod
    def config(self) -> LanguageModelConfig:
        raise NotImplementedError("'config' is not implemented")

    @staticmethod
    def check_tensor_type(tensor: TensorType) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("tensor type must be 'torch.Tensor'")

    def reset_kv_cache(self) -> None:
        pass


class LanguageModelMLX(ABC, mlx.nn.Module):
    @abstractmethod
    def __call__(
        self, idx: mx.array, cache: KVCache[mx.array] | None = None
    ) -> mx.array:
        return self(idx)

    @abstractmethod
    def config(self) -> LanguageModelConfig:
        raise NotImplementedError("'config' is not implemented")

    @staticmethod
    def check_tensor_type(tensor: TensorType) -> None:
        if not isinstance(tensor, mx.array):
            raise TypeError("tensor type must be 'mx.array'")

    def reset_kv_cache(self) -> None:
        pass
