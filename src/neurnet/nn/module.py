import mlx.core as mx
from mlx import nn

# ===-----------------------------------------------------------------------===
# MLX
# ===-----------------------------------------------------------------------===


class ModuleList(nn.Module):
    def __init__(self, modules: list[nn.Module]):
        self.inner = modules

    def __call__(self, x: mx.array) -> mx.array:
        for module in self.inner:
            x = module(x)
        return x


class Flatten(nn.Module):
    def __init__(self, start_axis: int):
        super().__init__()
        self.start_axis = start_axis

    def __call__(self, x: mx.array) -> mx.array:
        return mx.flatten(x, start_axis=self.start_axis)
