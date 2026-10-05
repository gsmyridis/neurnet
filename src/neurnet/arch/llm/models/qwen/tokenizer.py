import re
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from neurnet.arch.llm.types import Tokenizer


class Qwen3Tokenizer(Tokenizer):
    _SPECIALS: tuple[str, ...] = (
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|object_ref_start|>",
        "<|object_ref_end|>",
        "<|box_start|>",
        "<|box_end|>",
        "<|quad_start|>",
        "<|quad_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|vision_pad|>",
        "<|image_pad|>",
        "<|video_pad|>",
    )
    _SPLIT_RE: re.Pattern[str] = re.compile(r"(<\|[^>]+?\|>)")

    def __init__(
        self,
        tokenizer_file_path: str | Path = "tokenizer-base.json",
    ) -> None:
        from tokenizers import Tokenizer

        tok_path = Path(tokenizer_file_path)
        if not tok_path.is_file():
            raise FileNotFoundError(
                f"Tokenizer file '{tok_path}' not found. Please ensure it's available."
            )

        self._tok: Tokenizer = Tokenizer.from_file(str(tok_path))
        self._special_to_id: dict[str, int | None] = {
            t: self._tok.token_to_id(t) for t in self._SPECIALS
        }

        self.pad_token = "<|endoftext|>"
        self.pad_token_id = self._special_to_id.get(self.pad_token)

        # Match HF behavior: chat model → <|im_end|>, base model → <|endoftext|>
        fname = tok_path.name.lower()
        if "base" in fname and "reasoning" not in fname:
            self.eos_token = "<|endoftext|>"
        else:
            self.eos_token = "<|im_end|>"
        self.eos_token_id = self._special_to_id.get(self.eos_token)

    def end_of_sequence_token_id(self) -> int:
        assert self.eos_token_id is not None
        return self.eos_token_id

    def encode(self, text: str) -> Sequence[int]:
        stripped = text.strip()
        if stripped in self._special_to_id and "\n" not in stripped:
            return [cast(int, self._special_to_id[stripped])]

        ids: list[int] = []
        for part in filter(None, self._SPLIT_RE.split(text)):
            if part in self._special_to_id:
                ids.append(cast(int, self._special_to_id[part]))
            else:
                ids.extend(self._tok.encode(part).ids)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=False)
