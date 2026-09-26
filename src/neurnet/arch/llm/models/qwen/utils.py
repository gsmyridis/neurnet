import os
from pathlib import Path
from typing import Any, Literal, cast

import torch

from .download import download_qwen3_small
from .tokenizer import Qwen3Tokenizer
from .torch import QWEN_CONFIG_06_B, Qwen3TorchModel


def load_model_and_tokenizer(
    model_type: Literal["base", "reasoning"],
    device: Any,
    use_compile: bool,
    local_dir: str | Path,
) -> tuple[Qwen3TorchModel, Qwen3Tokenizer]:
    if model_type == "base":
        download_qwen3_small(kind="base", out_dir=local_dir)

        tokenizer_path = os.path.join(local_dir, "tokenizer-base.json")
        model_path = os.path.join(local_dir, "qwen3-0.6B-base.pth")
        tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)

    elif model_type == "reasoning":
        download_qwen3_small(kind="reasoning", out_dir=local_dir)

        tokenizer_path = os.path.join(local_dir, "tokenizer-reasoning.json")
        model_path = os.path.join(local_dir, "qwen3-0.6B-reasoning.pth")
        tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)
    else:
        raise ValueError(f"Invalid choice: which_model={model_type}")

    model = Qwen3TorchModel(QWEN_CONFIG_06_B)
    model.load_state_dict(torch.load(model_path))

    model.to(device)

    if use_compile:
        model = cast(Qwen3TorchModel, torch.compile(model))

    return model, tokenizer


def load_tokenizer(
    model_type: Literal["base", "reasoning"], local_dir: str | Path
) -> Qwen3Tokenizer:
    if model_type == "base":
        download_qwen3_small(kind="base", tokenizer_only=True, out_dir=local_dir)

        tokenizer_path = Path(local_dir) / "tokenizer-base.json"
        tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)

    elif model_type == "reasoning":
        download_qwen3_small(kind="reasoning", tokenizer_only=True, out_dir=local_dir)

        tokenizer_path = Path(local_dir) / "tokenizer-reasoning.json"
        tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)

    else:
        raise ValueError(f"Invalid choice: which_model={model_type}")

    return tokenizer
