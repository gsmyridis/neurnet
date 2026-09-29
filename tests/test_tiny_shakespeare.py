import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path

from neurnet.arch.llm.types import Tokenizer
from neurnet.datasets.tiny_shakespeare import TinyShakespeareMLXDataLoader


class _SequentialTokenizer(Tokenizer):
    def __init__(self, num_tokens: int) -> None:
        self._num_tokens = num_tokens

    def encode(self, text: str) -> Sequence[int]:
        return list(range(self._num_tokens))

    def decode(self, ids: Sequence[int]) -> str:
        return ""


class TinyShakespeareMLXDataLoaderTests(unittest.TestCase):
    def test_yields_full_input_target_batches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny_shakespeare.txt"
            path.write_text("test corpus", encoding="utf-8")
            loader = TinyShakespeareMLXDataLoader(
                path=str(path),
                tokenizer=_SequentialTokenizer(num_tokens=538),
                sequence_length=8,
                batch_size=64,
                shuffle=False,
                prefetch_batches=1,
                prefetch_worker_threads=1,
            )

            inputs, targets = next(loader)
            last_inputs, last_targets = next(loader)
            final_inputs, final_targets = next(loader)

        self.assertEqual(inputs.shape, (64, 8))
        self.assertEqual(targets.shape, (64, 8))
        self.assertEqual(inputs[0, 0].item(), 0)
        self.assertEqual(inputs[-1, -1].item(), 511)
        self.assertEqual(targets[0, 0].item(), 1)
        self.assertEqual(targets[-1, -1].item(), 512)
        self.assertEqual(last_inputs.shape, (3, 8))
        self.assertEqual(last_targets.shape, (3, 8))
        self.assertEqual(last_inputs[0, 0].item(), 512)
        self.assertEqual(last_targets[-1, -1].item(), 536)
        self.assertEqual(final_inputs.shape, (1, 1))
        self.assertEqual(final_targets.shape, (1, 1))
        self.assertEqual(final_inputs[0, 0].item(), 536)
        self.assertEqual(final_targets[0, 0].item(), 537)


if __name__ == "__main__":
    unittest.main()
