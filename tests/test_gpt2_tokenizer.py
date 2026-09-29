import unittest
from unittest import mock

from neurnet.arch.llm.models.gpt2.config import GPT2ModelType
from neurnet.arch.llm.models.gpt2.tokenizer import GPT2Tokenizer
from neurnet.arch.llm.types import Tokenizer


class GPT2TokenizerTests(unittest.TestCase):
    def test_wraps_tiktoken_encoding(self) -> None:
        encoding = mock.Mock(eot_token=50_256)
        encoding.encode.return_value = [1, 2]
        encoding.decode.return_value = "Hello"
        tokenizer = GPT2Tokenizer(encoding)

        self.assertIsInstance(tokenizer, Tokenizer)
        self.assertEqual(tokenizer.eos_token_id, 50_256)
        self.assertEqual(tokenizer.encode("Hello"), [1, 2])
        self.assertEqual(tokenizer.decode((1, 2)), "Hello")
        encoding.encode.assert_called_once_with(
            "Hello", allowed_special={"<|endoftext|>"}
        )
        encoding.decode.assert_called_once_with([1, 2])

    def test_loads_the_shared_gpt2_encoding(self) -> None:
        encoding = mock.Mock()

        with mock.patch(
            "neurnet.arch.llm.models.gpt2.tokenizer.tiktoken.get_encoding",
            return_value=encoding,
        ) as get_encoding:
            tokenizer = GPT2Tokenizer.from_pretrained(
                GPT2ModelType.MEDIUM,
                cache_dir="models/gpt2",
            )

        self.assertIsInstance(tokenizer, GPT2Tokenizer)
        get_encoding.assert_called_once_with("gpt2")


if __name__ == "__main__":
    unittest.main()
