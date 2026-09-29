import unittest

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn

from neurnet.arch.llm.config import LanguageModelConfig
from neurnet.arch.llm.kv_cache import KVCacheMLX
from neurnet.arch.llm.models.gpt2.mlx import GPT2MLXModel, make_gpt2_train_step


def small_config() -> LanguageModelConfig:
    return LanguageModelConfig(
        vocab_size=32,
        context_length=8,
        emb_dim=12,
        n_heads=3,
        n_layers=2,
        hidden_dim=48,
        head_dim=None,
        qk_norm=False,
        n_kv_groups=1,
        rope_base=10_000.0,
    )


class GPT2TrainingAndCacheTests(unittest.TestCase):
    def test_compiled_fixed_cache_matches_eager_and_keeps_shape(self) -> None:
        eager = GPT2MLXModel(small_config()).eval()
        compiled = GPT2MLXModel(small_config()).eval()
        compiled.update(eager.parameters())
        compiled.compile()

        fixed = compiled.make_kv_cache(batch_size=1, max_length=8)
        self.assertIsInstance(fixed, KVCacheMLX)
        prefix = mx.array([[]], dtype=mx.int32)
        for token_ids in ([[1, 2]], [[3]], [[4]], [[5]], [[6]]):
            tokens = mx.array(token_ids)
            prefix = mx.concatenate([prefix, tokens], axis=1)
            actual = compiled(tokens, cache=fixed)
            expected = eager(prefix)[:, -tokens.shape[1] :]
            mx.eval(actual, expected)
            self.assertTrue(bool(mx.allclose(actual, expected, atol=1e-5)))
            self.assertTrue(all(layer[0].shape[2] == 8 for layer in fixed.layers))

        fixed.reset()
        tokens = mx.array([[7, 1]])
        actual = compiled(tokens, cache=fixed)
        expected = eager(tokens)
        mx.eval(actual, expected)
        self.assertTrue(bool(mx.allclose(actual, expected, atol=1e-5)))

        fresh_cache = compiled.make_kv_cache(batch_size=1, max_length=8)
        fresh = compiled(tokens, cache=fresh_cache)
        mx.eval(fresh)
        self.assertTrue(bool(mx.allclose(fresh, expected, atol=1e-5)))

    def test_compiled_train_step_updates_all_precisions(self) -> None:
        tokens = mx.array([[1, 2, 3], [4, 5, 6]])
        targets = mx.array([[2, 3, 4], [5, 6, 7]])
        for dtype, master_weights in (
            (mx.float32, False),
            (mx.bfloat16, False),
            (mx.bfloat16, True),
        ):
            with self.subTest(dtype=dtype, master_weights=master_weights):
                model = GPT2MLXModel(
                    small_config(), embd_pdrop=0.0, resid_pdrop=0.0, attn_pdrop=0.0
                )
                if dtype == mx.bfloat16:
                    model.set_dtype(dtype)
                    model.transformer.wte.weight = model.lm_head.weight
                optimizer = optim.AdamW(learning_rate=1e-3)
                step = make_gpt2_train_step(
                    model,
                    optimizer,
                    nn.losses.cross_entropy,
                    compiled=True,
                    master_weights=master_weights,
                )
                before = model.lm_head.weight
                loss = step(tokens, targets)
                mx.eval(loss, model.state, optimizer.state)

                self.assertTrue(bool(mx.isfinite(loss)))
                self.assertFalse(bool(mx.allclose(before, model.lm_head.weight)))
                self.assertTrue(
                    bool(
                        mx.array_equal(
                            model.transformer.wte.weight, model.lm_head.weight
                        )
                    )
                )
                after_first = model.lm_head.weight
                second_loss = step(tokens, targets)
                mx.eval(second_loss, model.state, optimizer.state)
                self.assertFalse(
                    bool(mx.array_equal(after_first, model.lm_head.weight))
                )
                self.assertTrue(
                    bool(
                        mx.array_equal(
                            model.transformer.wte.weight, model.lm_head.weight
                        )
                    )
                )
                if master_weights:
                    self.assertEqual(
                        optimizer.state["lm_head"]["weight"]["m"].dtype,
                        mx.float32,
                    )

    def test_fixed_cache_enforces_context_limit(self) -> None:
        model = GPT2MLXModel(small_config()).eval().compile()
        cache = model.make_kv_cache(batch_size=1, max_length=8)
        model(mx.array([[1] * 8]), cache=cache)
        with self.assertRaisesRegex(ValueError, "context length exceeded"):
            model(mx.array([[2]]), cache=cache)


if __name__ == "__main__":
    unittest.main()
