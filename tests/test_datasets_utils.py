import unittest
from collections.abc import Sequence

from neurnet.arch.llm.types import Tokenizer
from neurnet.datasets.utils import text_to_buffer_mlx, tokens_to_buffer_mlx


class _SequentialTokenizer(Tokenizer):
    def end_of_sequence_token_id(self) -> int:
        return 0

    def encode(self, text: str) -> Sequence[int]:
        return list(range(518))

    def decode(self, ids: Sequence[int]) -> str:
        return ""


class DatasetUtilsTests(unittest.TestCase):
    def test_text_to_buffer_encodes_text(self) -> None:
        buffer = text_to_buffer_mlx(
            "test corpus",
            _SequentialTokenizer(),
            sequence_length=8,
            batch_size=64,
            shuffle=False,
        )

        self.assertEqual(len(buffer), 2)
        self.assertEqual(buffer[0]["inputs"].shape, (64, 8))
        self.assertEqual(buffer[1]["inputs"].shape, (1, 5))

    def test_tokens_to_buffer_retains_the_final_partial_sequence(self) -> None:
        buffer = tokens_to_buffer_mlx(
            list(range(538)), sequence_length=8, batch_size=64, shuffle=True
        )

        self.assertEqual(len(buffer), 3)
        first_batch, last_batch, final_sequence = buffer[0], buffer[1], buffer[2]
        self.assertEqual(first_batch["inputs"].shape, (64, 8))
        self.assertEqual(last_batch["inputs"].shape, (3, 8))
        self.assertEqual(last_batch["targets"].shape, (3, 8))
        self.assertEqual(last_batch["inputs"][0, 0].item(), 512)
        self.assertEqual(last_batch["inputs"][-1, -1].item(), 535)
        self.assertEqual(last_batch["targets"][0, 0].item(), 513)
        self.assertEqual(last_batch["targets"][-1, -1].item(), 536)
        self.assertEqual(final_sequence["inputs"].shape, (1, 1))
        self.assertEqual(final_sequence["targets"].shape, (1, 1))
        self.assertEqual(final_sequence["inputs"][0, 0].item(), 536)
        self.assertEqual(final_sequence["targets"][0, 0].item(), 537)


if __name__ == "__main__":
    unittest.main()
