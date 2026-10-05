from abc import ABC, abstractmethod
from collections.abc import Sequence

import mlx.core as mx
import mlx.nn
import torch

from neurnet.arch.llm.config import LanguageModelConfig
from neurnet.arch.llm.kv_cache import KVCacheMLX, KVCacheTorch

type LanguageModel = LanguageModelTorch | LanguageModelMLX
type TensorType = torch.Tensor | mx.array


class Tokenizer(ABC):
    @abstractmethod
    def end_of_sequence_token_id(self) -> int:
        raise NotImplementedError("'end_of_sequence_token_id' is not implemented")

    @abstractmethod
    def encode(self, text: str) -> Sequence[int]:
        raise NotImplementedError("'encode' is not implemented")

    @abstractmethod
    def decode(self, ids: Sequence[int]) -> str:
        raise NotImplementedError("'decode' is not implemented")


class LanguageModelTorch(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self, idx: torch.Tensor, cache: KVCacheTorch | None = None
    ) -> torch.Tensor:
        raise NotImplementedError("'forward' is not implemented")

    @abstractmethod
    def config(self) -> LanguageModelConfig:
        raise NotImplementedError("'config' is not implemented")

    @staticmethod
    def check_tensor_type(tensor: TensorType) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("tensor type must be 'torch.Tensor'")

    def create_kv_cache(
        self, batch_size: int, max_length: int, device: torch.device
    ) -> KVCacheTorch:
        raise NotImplementedError("this model has no fixed Torch KV cache")


class LanguageModelMLX(ABC, mlx.nn.Module):
    @abstractmethod
    def __call__(self, idx: mx.array, cache: KVCacheMLX | None = None) -> mx.array:
        return self(idx)

    @abstractmethod
    def config(self) -> LanguageModelConfig:
        raise NotImplementedError("'config' is not implemented")

    @staticmethod
    def check_tensor_type(tensor: TensorType) -> None:
        if not isinstance(tensor, mx.array):
            raise TypeError("tensor type must be 'mx.array'")

    def create_kv_cache(self, batch_size: int, max_length: int) -> KVCacheMLX:
        raise NotImplementedError("this model has no fixed MLX KV cache")
