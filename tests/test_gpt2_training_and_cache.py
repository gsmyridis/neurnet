import io
import unittest
from contextlib import redirect_stderr, redirect_stdout

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn
from mlx.utils import tree_flatten

from neurnet.arch.llm.config import LanguageModelConfig
from neurnet.arch.llm.kv_cache import KVCacheMLX
from neurnet.arch.llm.models.gpt2.mlx import (
    GPT2MLXModel,
    iter_gpt2_train_steps,
    make_gpt2_train_step,
    train_gpt2,
)
from neurnet.datasets import MLXDataLoader


class _Batches(MLXDataLoader):
    def __init__(self, batches):
        self.batches = batches
        self.reset()

    def __len__(self):
        return len(self.batches)

    def __next__(self):
        return next(self.iterator)

    def reset(self):
        self.iterator = iter(self.batches)


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
    def test_async_epochs_match_sync_with_progress_and_partial_batches(self) -> None:
        batches = [
            (mx.array([[1, 2, 3], [4, 5, 6]]), mx.array([[2, 3, 4], [5, 6, 7]])),
            (mx.array([[7, 8, 9], [1, 2, 3]]), mx.array([[8, 9, 1], [2, 3, 4]])),
            (mx.array([[2]]), mx.array([[3]])),
        ]
        for compiled in (False, True):
            for dtype, master_weights in (
                (mx.float32, False),
                (mx.bfloat16, False),
                (mx.bfloat16, True),
            ):
                with self.subTest(
                    compiled=compiled, dtype=dtype, master=master_weights
                ):
                    models = [
                        GPT2MLXModel(
                            small_config(),
                            embd_pdrop=0.0,
                            resid_pdrop=0.0,
                            attn_pdrop=0.0,
                        )
                        for _ in range(2)
                    ]
                    for model in models:
                        model.set_dtype(dtype)
                        model.transformer.wte.weight = model.lm_head.weight
                    models[1].update(models[0].parameters())
                    optimizers = [optim.AdamW(learning_rate=1e-3) for _ in models]
                    reports = []
                    for async_eval, model, optimizer in zip(
                        (False, True), models, optimizers, strict=True
                    ):
                        report, progress = io.StringIO(), io.StringIO()
                        loader = _Batches(batches)
                        with redirect_stdout(report), redirect_stderr(progress):
                            train_gpt2(
                                model,
                                loader,
                                optimizer,
                                nn.losses.cross_entropy,
                                epochs=2,
                                verbose=True,
                                compiled=compiled,
                                master_weights=master_weights,
                                async_eval=async_eval,
                                progress=True,
                            )
                        self.assertIn("Epoch 2/2", progress.getvalue())
                        self.assertEqual(len(list(loader)), 3)
                        reports.append(
                            [
                                line.split(",")[0]
                                for line in report.getvalue().splitlines()
                            ]
                        )
                    self.assertEqual(reports[0], reports[1])
                    for left, right in (
                        (models[0].parameters(), models[1].parameters()),
                        (optimizers[0].state, optimizers[1].state),
                    ):
                        for (name, expected), (actual_name, actual) in zip(
                            tree_flatten(left), tree_flatten(right), strict=True
                        ):
                            self.assertEqual(name, actual_name)
                            self.assertTrue(
                                bool(mx.allclose(expected, actual, atol=1e-6)), name
                            )

    def test_async_iterator_bounds_lookahead_and_drains_final_step(self) -> None:
        for batch_count in (1, 3):
            with self.subTest(batch_count=batch_count):
                state = [mx.array(0)]
                submitted = []
                batch = (mx.array([[1, 2]]), mx.array([[2, 3]]))

                def step(inputs, targets, state=state, submitted=submitted):
                    submitted.append(1)
                    state[0] = state[0] + 1
                    return state[0]

                completed = iter_gpt2_train_steps(step, [batch] * batch_count, state)
                for index in range(batch_count):
                    loss, tokens = next(completed)
                    self.assertEqual(loss.item(), index + 1)
                    self.assertEqual(tokens, 2)
                    self.assertEqual(len(submitted), min(index + 2, batch_count))
                with self.assertRaises(StopIteration):
                    next(completed)
                self.assertEqual(state[0].item(), batch_count)

    def test_training_rejects_empty_loader_without_progress_output(self) -> None:
        for async_eval in (False, True):
            output = io.StringIO()
            with (
                self.subTest(async_eval=async_eval),
                redirect_stderr(output),
                self.assertRaisesRegex(ValueError, "empty dataloader"),
            ):
                train_gpt2(
                    GPT2MLXModel(small_config()),
                    _Batches([]),
                    optim.AdamW(learning_rate=1e-3),
                    nn.losses.cross_entropy,
                    epochs=1,
                    async_eval=async_eval,
                    progress=False,
                )
            self.assertEqual(output.getvalue(), "")

    def test_compiled_fixed_cache_matches_eager_and_keeps_shape(self) -> None:
        eager = GPT2MLXModel(small_config()).eval()
        compiled = GPT2MLXModel(small_config()).eval()
        compiled.update(eager.parameters())
        compiled.compile()

        fixed = compiled.create_kv_cache(batch_size=1, max_length=8)
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

        fresh_cache = compiled.create_kv_cache(batch_size=1, max_length=8)
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
        cache = model.create_kv_cache(batch_size=1, max_length=8)
        model(mx.array([[1] * 8]), cache=cache)
        with self.assertRaisesRegex(ValueError, "context length exceeded"):
            model(mx.array([[2]]), cache=cache)


if __name__ == "__main__":
    unittest.main()
