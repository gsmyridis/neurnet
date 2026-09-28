from enum import Enum

import mlx.nn
import torch.nn


class Backend(Enum):
    TORCH = "Torch"
    MLX = "MLX"

    def __str__(self) -> str:
        return self.value


def get_backend(module: mlx.nn.Module | torch.nn.Module) -> Backend:
    if isinstance(module, mlx.nn.Module):
        return Backend.MLX
    if isinstance(module, torch.nn.Module):
        return Backend.TORCH
