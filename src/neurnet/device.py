from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Self

import mlx.core as mx
import torch


class DeviceType(Enum):
    CPU = "cpu"
    GPU = "gpu"

    @classmethod
    def from_string(cls, string: str) -> Self:
        return cls(string.lower())


@dataclass(frozen=True)
class Device:
    dtype: DeviceType

    @classmethod
    def cpu(cls):
        return cls(DeviceType.CPU)

    @classmethod
    def gpu(cls):
        return cls(DeviceType.GPU)

    @classmethod
    def available(cls):
        if is_torch_gpu_available():
            return cls.gpu()
        return cls.cpu()

    def is_cpu(self) -> bool:
        return self.dtype == DeviceType.CPU

    def is_gpu(self) -> bool:
        return self.dtype == DeviceType.GPU

    def to_torch(self) -> torch.device:
        if self.is_cpu():
            return torch.device("cpu")

        return get_torch_gpu_device()

    def to_mlx(self) -> mx.DeviceType:
        if self.is_cpu():
            return mx.DeviceType.cpu

        if mx.device_count(mx.DeviceType.gpu) == 0:
            raise ValueError("MLX GPU is not available")

        return mx.DeviceType.gpu


# ===---------------------------------------------------------------------------===
# Torch device
# ===---------------------------------------------------------------------------===


def is_torch_gpu_available() -> bool:
    device, _ = _try_get_torch_gpu_with_name(False)
    return device is not None


def get_torch_device(
    enable_tensor_cores: bool = True, *, verbose: bool = False
) -> torch.device:
    device, backend_name = _try_get_torch_gpu_with_name(enable_tensor_cores)

    if device is None:
        if verbose:
            print("Using CPU")
        return torch.device("cpu")

    if verbose:
        print(f"Using {backend_name} GPU")
    return device


def get_torch_gpu_device(enable_tensor_cores: bool = True) -> torch.device:
    device, _ = _try_get_torch_gpu_with_name(enable_tensor_cores)
    if device is None:
        raise ValueError("GPU is not available")

    return device


def _try_get_torch_gpu_with_name(
    enable_tensor_cores: bool,
) -> tuple[torch.device | None, str | None]:
    if torch.cuda.is_available():
        device = torch.device("cuda")

        if enable_tensor_cores:
            major, minor = map(int, torch.__version__.split(".")[:2])
            # PyTorch 2.9 and 2.10 still read the legacy TF32 setting in torch.compile.
            # See https://github.com/pytorch/pytorch/issues/166387
            # and https://github.com/rasbt/reasoning-from-scratch/issues/256
            if (major, minor) >= (2, 11):
                torch.backends.cuda.matmul.fp32_precision = "tf32"
                torch.backends.cudnn.conv.fp32_precision = "tf32"
            else:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        return device, "CUDA"

    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        return device, "MPS"

    elif torch.xpu.is_available():
        device = torch.device("xpu")
        return device, "XPU"

    return None, None
