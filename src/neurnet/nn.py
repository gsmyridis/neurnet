from typing import Self

import torch

# ===-----------------------------------------------------------------------===
# Torch
# ===-----------------------------------------------------------------------===


class DeserializableTorchModel(torch.nn.Module):
    @classmethod
    def from_path(cls, path: str) -> Self:
        cached = cls()
        cached.load_state_dict(torch.load(path))
        return cached


class SerializableTorchModel(torch.nn.Module):
    def save_to(self, path: str) -> None:
        torch.save(self.state_dict(), path)
