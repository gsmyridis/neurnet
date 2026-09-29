import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from neurnet.utils.serde import deserialise_torch_model, serialize_torch_model


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)


class TorchModelSerializationTests(unittest.TestCase):
    def test_serializes_and_deserialises_model_state(self) -> None:
        torch.manual_seed(0)
        model = _TinyModel()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pth"
            serialize_torch_model(model, path)
            restored = deserialise_torch_model(_TinyModel, path)

        for expected, actual in zip(
            model.state_dict().values(), restored.state_dict().values(), strict=True
        ):
            torch.testing.assert_close(actual, expected)

    def test_requires_pth_model_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "must end with '.pth'"):
            serialize_torch_model(_TinyModel(), "model.pt")


if __name__ == "__main__":
    unittest.main()
