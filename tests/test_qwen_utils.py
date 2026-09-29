import unittest
from pathlib import Path
from unittest import mock

import torch

from neurnet.arch.llm.config import LanguageModelConfig
from neurnet.arch.llm.models.qwen import utils as qwen_utils
from neurnet.arch.llm.models.qwen.torch import Qwen3TorchModel


def _tiny_qwen_config() -> LanguageModelConfig:
    return LanguageModelConfig(
        vocab_size=16,
        context_length=8,
        emb_dim=8,
        n_heads=2,
        n_layers=1,
        hidden_dim=16,
        head_dim=None,
        qk_norm=False,
        n_kv_groups=1,
        rope_base=10_000.0,
    )


class QwenLoaderTests(unittest.TestCase):
    def test_compile_preserves_qwen_model_type(self) -> None:
        with (
            mock.patch.object(qwen_utils, "QWEN_CONFIG_06_B", _tiny_qwen_config()),
            mock.patch.object(qwen_utils, "download_qwen3_small"),
            mock.patch.object(qwen_utils.torch, "load", return_value={}),
            mock.patch.object(Qwen3TorchModel, "load_state_dict"),
            mock.patch.object(qwen_utils, "Qwen3Tokenizer"),
            mock.patch.object(qwen_utils.torch, "compile", return_value=mock.Mock()),
        ):
            model, _ = qwen_utils.load_model_and_tokenizer(
                "base",
                torch.device("cpu"),
                use_compile=True,
                local_dir=Path("models/qwen3"),
            )

        self.assertIsInstance(model, Qwen3TorchModel)


if __name__ == "__main__":
    unittest.main()
