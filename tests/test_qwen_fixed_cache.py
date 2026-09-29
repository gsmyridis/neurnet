import unittest

import torch

from neurnet.arch.llm.config import LanguageModelConfig
from neurnet.arch.llm.kv_cache import KVCacheTorch
from neurnet.arch.llm.models.qwen.torch import Qwen3TorchModel


def tiny_config() -> LanguageModelConfig:
    return LanguageModelConfig(
        vocab_size=32,
        context_length=16,
        emb_dim=16,
        n_heads=4,
        n_layers=2,
        hidden_dim=32,
        head_dim=4,
        qk_norm=True,
        n_kv_groups=2,
        rope_base=10_000.0,
    )


class QwenFixedCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(10)
        self.model = Qwen3TorchModel(tiny_config(), dtype=torch.float32).eval()
        self.prompt = torch.tensor([[1, 2, 3]])
        self.continuation = [4, 5, 6]

    @torch.inference_mode()
    def test_fixed_cache_matches_uncached_logits_across_decode_steps(self) -> None:
        expected = [self.model(self.prompt)]
        prefix = self.prompt
        for token in self.continuation:
            prefix = torch.cat((prefix, torch.tensor([[token]])), dim=1)
            expected.append(self.model(prefix)[:, -1:])

        fixed = self.model.make_kv_cache(1, 8, torch.device("cpu"))
        self.assertIsInstance(fixed, KVCacheTorch)
        self.assertEqual(fixed.capacity, self.model.cfg.context_length)
        actual = [self.model(self.prompt, fixed)]
        for token in self.continuation:
            actual.append(self.model(torch.tensor([[token]]), fixed))

        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-5)
        self.assertEqual(fixed.offset, 6)
        self.assertEqual(fixed.layers[0][0].shape, (1, 2, 16, 4))

    @torch.inference_mode()
    def test_reset_reuses_allocation_without_stale_tokens(self) -> None:
        fixed = self.model.make_kv_cache(1, 8, torch.device("cpu"))
        first = self.model(self.prompt, fixed)
        self.model(torch.tensor([[4]]), fixed)
        storage = fixed.layers[0][0]

        fixed.reset()
        second = self.model(self.prompt, fixed)
        self.assertIs(storage, fixed.layers[0][0])
        torch.testing.assert_close(first, second)
        self.assertEqual(fixed.offset, 3)

    @torch.inference_mode()
    def test_context_limit_is_checked(self) -> None:
        with self.assertRaisesRegex(ValueError, "context length"):
            self.model.make_kv_cache(1, 17, torch.device("cpu"))

        fixed = self.model.make_kv_cache(1, 16, torch.device("cpu"))
        self.model(torch.zeros((1, 16), dtype=torch.long), fixed)
        with self.assertRaisesRegex(ValueError, "context length"):
            self.model(torch.tensor([[1]]), fixed)

    @torch.inference_mode()
    def test_compiled_decoder_reuses_one_graph(self) -> None:
        compile_count = 0

        def counting_backend(graph: torch.fx.GraphModule, _inputs: list[torch.Tensor]):
            nonlocal compile_count
            compile_count += 1
            return graph.forward

        self.assertIsNone(self.model.compile(backend=counting_backend, fullgraph=True))
        fixed = self.model.make_kv_cache(1, 8, torch.device("cpu"))
        self.model(self.prompt, fixed)
        for token in self.continuation:
            self.model(torch.tensor([[token]]), fixed)

        self.assertEqual(compile_count, 1)

    @torch.inference_mode()
    def test_cache_dtype_tracks_model_conversion(self) -> None:
        model = Qwen3TorchModel(tiny_config()).float().eval()
        fixed = model.make_kv_cache(1, 8, torch.device("cpu"))
        self.assertEqual(fixed.layers[0][0].dtype, torch.float32)
        model(self.prompt, fixed)
        logits = model(torch.tensor([[4]]), fixed)
        self.assertEqual(logits.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
