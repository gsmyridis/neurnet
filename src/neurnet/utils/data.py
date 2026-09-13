from dataclasses import dataclass

from torch.utils.data import DataLoader


@dataclass(frozen=True)
class SizedDataLoader:
    loader: DataLoader
    len: int

    def __iter__(self):
        return iter(self.loader)
