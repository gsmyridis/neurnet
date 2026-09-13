from typing import Self

import torch
from torch import nn


class DeserializableTorchModel(nn.Module):
    @classmethod
    def from_path(cls, path: str) -> Self:
        cached = cls()
        cached.load_state_dict(torch.load(path))
        return cached


class SerializableTorchModel(nn.Module):
    def save_to(self, path: str) -> None:
        torch.save(self.state_dict(), path)
