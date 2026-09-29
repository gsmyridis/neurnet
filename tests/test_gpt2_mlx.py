import unittest

import mlx.core as mx
import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from neurnet.arch.llm.config import LanguageModelConfig
from neurnet.arch.llm.inference import generate_token_stream
from neurnet.arch.llm.models.gpt2.mlx import CausalSelfAttention, GPT2MLXModel


def _test_config() -> LanguageModelConfig:
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


class CausalSelfAttentionTests(unittest.TestCase):
    def test_preserves_shape_and_cannot_attend_to_future_tokens(self) -> None:
        attention = CausalSelfAttention(
            dim_embed=12,
            n_head=3,
            attn_pdrop=0.5,
            resid_pdrop=0.5,
        )
        attention.eval()

        x = mx.random.normal((2, 5, 12))
        x_with_changed_future = mx.concatenate(
            [x[:, :3], mx.random.normal((2, 2, 12))],
            axis=1,
        )

        output, _ = attention(x)
        changed_output, _ = attention(x_with_changed_future)
        mx.eval(output, changed_output)

        self.assertEqual(output.shape, x.shape)
        self.assertTrue(bool(mx.allclose(output[:, :3], changed_output[:, :3])))


class GPT2Tests(unittest.TestCase):
    def test_loads_hugging_face_state_dict(self) -> None:
        torch.manual_seed(0)
        hf_config = GPT2Config(
            vocab_size=32,
            n_positions=8,
            n_embd=12,
            n_layer=2,
            n_head=3,
            embd_pdrop=0.0,
            resid_pdrop=0.0,
            attn_pdrop=0.0,
            bos_token_id=None,
            eos_token_id=None,
        )
        model_hf = GPT2LMHeadModel(hf_config)
        model_hf.eval()

        model = GPT2MLXModel(
            _test_config(),
            embd_pdrop=0.0,
            resid_pdrop=0.0,
            attn_pdrop=0.0,
        )
        model.eval()
        model._load_hugging_face_weights(model_hf.state_dict())

        indices = [[1, 2, 3, 4]]
        logits = model(mx.array(indices))
        mx.eval(logits)
        with torch.no_grad():
            logits_hf = model_hf(torch.tensor(indices)).logits

        np.testing.assert_allclose(
            np.array(logits),
            logits_hf.detach().numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_returns_logits_for_every_input_token(self) -> None:
        model = GPT2MLXModel(_test_config())

        logits = model(mx.array([[1, 2, 3], [4, 5, 6]]))
        mx.eval(logits)

        self.assertEqual(logits.shape, (2, 3, 32))

    def test_cached_forward_matches_uncached_forward(self) -> None:
        model = GPT2MLXModel(_test_config())
        model.eval()
        cache = model.make_kv_cache(batch_size=1, max_length=4)

        model(mx.array([[1, 2, 3]]), cache=cache)
        cached_logits = model(mx.array([[4]]), cache=cache)
        uncached_logits = model(mx.array([[1, 2, 3, 4]]))
        mx.eval(cached_logits, uncached_logits)

        np.testing.assert_allclose(
            np.array(cached_logits[:, -1]),
            np.array(uncached_logits[:, -1]),
            rtol=1e-5,
            atol=1e-5,
        )
        self.assertEqual(cache.layers[0][0].shape[2], cache.capacity)
        self.assertEqual(cache.offset, 4)

    def test_compiled_model_supports_cached_generation(self) -> None:
        eager_model = GPT2MLXModel(_test_config())
        eager_model.eval()
        mx.eval(eager_model.parameters())

        compiled_model = GPT2MLXModel(_test_config())
        compiled_model.update(eager_model.parameters())
        compiled_model.eval()
        compiled_model.compile()

        token_ids = mx.array([[1, 2, 3]])
        eager_ids = list(
            generate_token_stream(
                eager_model,
                token_ids,
                max_new_tokens=3,
                use_kv_cache=True,
            )
        )
        compiled_ids = list(
            generate_token_stream(
                compiled_model,
                token_ids,
                max_new_tokens=3,
                use_kv_cache=True,
            )
        )

        self.assertEqual(compiled_ids, eager_ids)

    def test_training_mode_controls_all_model_dropouts(self) -> None:
        model = GPT2MLXModel(
            _test_config(),
            embd_pdrop=0.5,
            resid_pdrop=0.5,
            attn_pdrop=0.5,
        )
        indices = mx.array([[1, 2, 3], [4, 5, 6]])

        model.eval()
        eval_logits_1 = model(indices)
        mx.eval(eval_logits_1)
        eval_logits_2 = model(indices)
        mx.eval(eval_logits_2)
        self.assertTrue(bool(mx.allclose(eval_logits_1, eval_logits_2)))

        model.train()
        train_logits_1 = model(indices)
        mx.eval(train_logits_1)
        train_logits_2 = model(indices)
        mx.eval(train_logits_2)
        self.assertFalse(bool(mx.allclose(train_logits_1, train_logits_2)))


if __name__ == "__main__":
    unittest.main()
