import mlx.core as mx
import torch

type LayerCache[T] = tuple[T, T]
type LayerCacheTorch = LayerCache[torch.Tensor]
type LayerCacheMLX = LayerCache[mx.array]

type KVCacheTorch = KVCache[torch.Tensor]
type KVCacheMLX = KVCache[mx.array]


class KVCache[T]:
    def __init__(self, n_layers: int) -> None:
        self.cache: list[LayerCache[T] | None] = [None] * n_layers

    def get(self, layer_idx: int) -> LayerCache[T] | None:
        return self.cache[layer_idx]

    def update(self, layer_idx: int, value: LayerCache[T]) -> None:
        self.cache[layer_idx] = value

    def get_all(self) -> list[LayerCache[T] | None]:
        return self.cache

    def reset(self) -> None:
        for i in range(len(self.cache)):
            self.cache[i] = None
