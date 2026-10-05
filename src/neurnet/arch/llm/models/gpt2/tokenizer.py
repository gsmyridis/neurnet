from collections.abc import Sequence

import tiktoken

from neurnet.arch.llm.types import Tokenizer

from .config import GPT2ModelType


class GPT2Tokenizer(Tokenizer):
    """Tokenizer adapter for GPT-2's tiktoken encoding."""

    _ENCODING_NAME = "gpt2"
    _END_OF_TEXT = "<|endoftext|>"

    def __init__(self, encoding: tiktoken.Encoding) -> None:
        self._encoding = encoding

    @classmethod
    def from_pretrained(
        cls, model_type: GPT2ModelType, cache_dir: str
    ) -> "GPT2Tokenizer":
        """Create the shared tokenizer used by every GPT-2 checkpoint.

        ``model_type`` and ``cache_dir`` are retained for compatibility with
        the model-loading API. GPT-2 checkpoints share one vocabulary, and
        tiktoken manages its encoding cache independently.
        """
        del model_type, cache_dir
        return cls(tiktoken.get_encoding(cls._ENCODING_NAME))

    def end_of_sequence_token_id(self) -> int:
        return self._encoding.eot_token

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode(text, allowed_special={self._END_OF_TEXT})

    def decode(self, ids: Sequence[int]) -> str:
        decoded = self._encoding.decode(list(ids))
        assert isinstance(decoded, str)
        return decoded
