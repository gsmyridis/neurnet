from collections.abc import Sequence
from typing import cast

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from neurnet.arch.llm.traits import Tokenizer

from .config import GPT2ModelType


class GPT2Tokenizer(Tokenizer):
    """Tokenizer adapter for Hugging Face GPT-2 checkpoints."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self._tokenizer = tokenizer

    @classmethod
    def from_pretrained(
        cls, model_type: GPT2ModelType, cache_dir: str
    ) -> "GPT2Tokenizer":
        tokenizer = AutoTokenizer.from_pretrained(
            model_type.value,
            cache_dir=cache_dir,
        )
        tokenizer = cast(PreTrainedTokenizerBase, tokenizer)
        return cls(tokenizer)

    @property
    def eos_token_id(self) -> int | None:
        return self._tokenizer.eos_token_id

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text)

    def decode(self, ids: Sequence[int]) -> str:
        decoded = self._tokenizer.decode(list(ids))
        assert isinstance(decoded, str)
        return decoded
