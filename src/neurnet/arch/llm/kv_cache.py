import mlx.core as mx
import torch

type LayerCacheMLX = tuple[mx.array, mx.array]
type LayerCacheTorch = tuple[torch.Tensor, torch.Tensor]

_CACHE_BUCKET_SIZE = 256


def fixed_cache_capacity(max_length: int, context_length: int) -> int:
    """Round a positive token budget up to a 256-token cache bucket.

    The result is capped at ``context_length``. Raise ``ValueError`` if the
    requested budget is non-positive or exceeds the model's context length.
    """
    if max_length < 1:
        raise ValueError("cache length must be positive")
    if max_length > context_length:
        raise ValueError("model context length exceeded")
    # The final partial bucket is still fixed-shape. Capping it avoids unused
    # positions and masked attention work; 256 alignment is not always faster.
    return min(
        context_length,
        ((max_length + _CACHE_BUCKET_SIZE - 1) // _CACHE_BUCKET_SIZE)
        * _CACHE_BUCKET_SIZE,
    )


class KVCacheMLX:
    """Fixed-shape MLX storage with an array position for compiled decoding."""

    def __init__(
        self,
        *,
        n_layers: int,
        batch_size: int,
        n_heads: int,
        head_dim: int,
        capacity: int,
        dtype: mx.Dtype,
    ) -> None:
        shape = (batch_size, n_heads, capacity, head_dim)
        self._layers = [
            (mx.zeros(shape, dtype), mx.zeros(shape, dtype)) for _ in range(n_layers)
        ]
        self.capacity = capacity
        self.position = mx.array(0, dtype=mx.int32)
        self.offset = 0

    @property
    def layers(self) -> tuple[LayerCacheMLX, ...]:
        return tuple(self._layers)

    @layers.setter
    def layers(self, layers: tuple[LayerCacheMLX, ...]) -> None:
        if len(layers) != len(self._layers):
            raise ValueError("cache layer count cannot change")
        self._layers = list(layers)

    def write_prefix(self, layer_idx: int, key: mx.array, value: mx.array) -> None:
        destination = self._layers[layer_idx]
        self._layers[layer_idx] = (
            mx.slice_update(destination[0], key, mx.array(0), (2,)),
            mx.slice_update(destination[1], value, mx.array(0), (2,)),
        )

    def finish_prefill(self, length: int) -> None:
        if length > self.capacity:
            raise ValueError("cache capacity exceeded")
        self.position = mx.array(length, dtype=mx.int32)
        self.offset = length

    def reset(self) -> None:
        # Prefill overwrites the visible prefix; decoding masks the tail.
        self.position = mx.array(0, dtype=mx.int32)
        self.offset = 0


class KVCacheTorch:
    """Fixed-shape Torch storage with a tensor position for compiled decoding."""

    def __init__(
        self,
        *,
        n_layers: int,
        batch_size: int,
        n_heads: int,
        head_dim: int,
        capacity: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        shape = (batch_size, n_heads, capacity, head_dim)
        self._layers = [
            (
                torch.zeros(shape, dtype=dtype, device=device),
                torch.zeros(shape, dtype=dtype, device=device),
            )
            for _ in range(n_layers)
        ]
        self.capacity = capacity
        self.position = torch.zeros((), dtype=torch.long, device=device)
        self.offset = 0

    @property
    def layers(self) -> tuple[LayerCacheTorch, ...]:
        return tuple(self._layers)

    def write_prefix(
        self, layer_idx: int, key: torch.Tensor, value: torch.Tensor
    ) -> None:
        destination = self._layers[layer_idx]
        destination[0][:, :, : key.shape[2], :].copy_(key)
        destination[1][:, :, : value.shape[2], :].copy_(value)

    def finish_prefill(self, length: int) -> None:
        if length > self.capacity:
            raise ValueError("cache capacity exceeded")
        self.position.fill_(length)
        self.offset = length

    def reset(self) -> None:
        self.position.zero_()
        self.offset = 0
