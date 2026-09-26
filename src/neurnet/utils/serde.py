from os import PathLike

import torch
from torch import nn


def serialize_torch_model(model: nn.Module, path: str | PathLike[str]) -> None:
    _validate_torch_model_path(path)
    torch.save(model.state_dict(), path)


def deserialise_torch_model[ModelT: nn.Module](
    model_type: type[ModelT], path: str | PathLike[str]
) -> ModelT:
    _validate_torch_model_path(path)
    model = model_type()
    model.load_state_dict(torch.load(path, weights_only=True))
    return model


def _validate_torch_model_path(path: str | PathLike[str]) -> None:
    if not str(path).endswith(".pth"):
        raise ValueError("Torch model paths must end with '.pth'")
