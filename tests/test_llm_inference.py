import sys
import unittest
from collections.abc import Callable, Sequence
from unittest import mock

import mlx.core as mx
import torch

import neurnet.arch.llm.inference.runtime as runtime_module
from neurnet.arch.llm.config import LanguageModelConfig
from neurnet.arch.llm.inference import (
    ChatMessage,
    ChatRole,
    ChatSession,
    GenerationConfig,
    GenerationPolicy,
    GenerationResult,
    InferenceRuntime,
    generate_token_stream,
)
from neurnet.arch.llm.models.gpt2.mlx import GPT2MLXModel
from neurnet.arch.llm.models.qwen.torch import Qwen3TorchModel
from neurnet.arch.llm.types import LanguageModelTorch, Tokenizer
from neurnet.device import Device
from scripts.llm import chat as chat_script


def _tiny_qwen_config() -> LanguageModelConfig:
    return LanguageModelConfig(
        vocab_size=16,
        context_length=8,
        emb_dim=8,
        n_heads=2,
        n_layers=1,
        hidden_dim=16,
        head_dim=None,
        qk_norm=False,
        n_kv_groups=1,
        rope_base=10_000.0,
    )


def _tiny_gpt2_config() -> LanguageModelConfig:
    return LanguageModelConfig(
        vocab_size=16,
        context_length=8,
        emb_dim=8,
        n_heads=2,
        n_layers=1,
        hidden_dim=16,
        head_dim=None,
        qk_norm=False,
        n_kv_groups=1,
        rope_base=10_000.0,
    )


class GenerateTokenStreamTests(unittest.TestCase):
    def test_zero_temperature_uses_greedy_decoding_without_sampling(self) -> None:
        model = _FixedLogitsTorchModel(
            _tiny_qwen_config(),
            torch.tensor([0.0, 0.1, 0.0, 0.0]),
        )

        with mock.patch.object(torch, "multinomial") as multinomial:
            generated_ids = list(
                generate_token_stream(
                    model,
                    torch.tensor([[0]]),
                    max_new_tokens=1,
                    policy=GenerationPolicy(
                        temperature=0,
                        repetition_penalty=1,
                        max_consecutive_repeats=0,
                        repeat_ngram_size=0,
                    ),
                )
            )

        self.assertEqual(generated_ids, [1])
        multinomial.assert_not_called()

    def test_top_k_one_samples_only_the_highest_logit_token(self) -> None:
        model = _FixedLogitsTorchModel(
            _tiny_qwen_config(),
            torch.tensor([0.0, 0.1, 0.0, 0.0]),
        )

        generated_ids = list(
            generate_token_stream(
                model,
                torch.tensor([[0]]),
                max_new_tokens=1,
                policy=GenerationPolicy(
                    top_k=1,
                    repetition_penalty=1,
                    max_consecutive_repeats=0,
                    repeat_ngram_size=0,
                ),
            )
        )

        self.assertEqual(generated_ids, [1])

    def test_stops_before_a_fourth_consecutive_token(self) -> None:
        model = _CyclingTorchModel(_tiny_qwen_config(), {0: 1, 1: 1})

        generated_ids = list(
            generate_token_stream(
                model,
                torch.tensor([[0]]),
                max_new_tokens=5,
                policy=GenerationPolicy(
                    max_consecutive_repeats=3,
                    repeat_ngram_size=0,
                ),
            )
        )

        self.assertEqual(generated_ids, [1, 1, 1])

    def test_stops_before_repeating_an_ngram(self) -> None:
        model = _CyclingTorchModel(_tiny_qwen_config(), {0: 1, 1: 2, 2: 1})

        generated_ids = list(
            generate_token_stream(
                model,
                torch.tensor([[0]]),
                max_new_tokens=6,
                policy=GenerationPolicy(
                    max_consecutive_repeats=0,
                    repeat_ngram_size=2,
                ),
            )
        )

        self.assertEqual(generated_ids, [1, 2, 1])

    def test_torch_cached_generation_matches_non_cached_generation(self) -> None:
        torch.manual_seed(0)
        model = Qwen3TorchModel(_tiny_qwen_config())
        token_ids = torch.tensor([[1, 2, 3]])

        generated_ids_not_cached = list(
            generate_token_stream(
                model, token_ids, max_new_tokens=3, use_kv_cache=False
            )
        )
        generated_ids_cached = list(
            generate_token_stream(model, token_ids, max_new_tokens=3, use_kv_cache=True)
        )

        self.assertEqual(generated_ids_not_cached, generated_ids_cached)

    def test_cached_generation_requires_fixed_cache_support(self) -> None:
        model = _FixedLogitsTorchModel(
            _tiny_qwen_config(), torch.tensor([0.0, 1.0, 0.0, 0.0])
        )

        with self.assertRaisesRegex(NotImplementedError, "fixed Torch KV cache"):
            list(
                generate_token_stream(
                    model,
                    torch.tensor([[1]]),
                    max_new_tokens=1,
                    use_kv_cache=True,
                )
            )

    def test_mlx_cached_generation_matches_non_cached_generation(self) -> None:
        mx.random.seed(0)
        model = GPT2MLXModel(_tiny_gpt2_config())
        token_ids = mx.array([[1, 2, 3]])

        generated_ids_not_cached = list(
            generate_token_stream(
                model, token_ids, max_new_tokens=3, use_kv_cache=False
            )
        )
        generated_ids_cached = list(
            generate_token_stream(model, token_ids, max_new_tokens=3, use_kv_cache=True)
        )

        self.assertEqual(generated_ids_not_cached, generated_ids_cached)


class ChatTests(unittest.TestCase):
    def test_base_qwen_defaults_to_plain_prompt(self) -> None:
        self.assertIs(
            chat_script._select_prompt_builder("qwen3", None, reasoning=False),
            chat_script._build_plain_prompt,
        )

    def test_reasoning_qwen_defaults_to_chat_prompt(self) -> None:
        self.assertIs(
            chat_script._select_prompt_builder("qwen3", None, reasoning=True),
            chat_script._build_qwen_prompt,
        )

    def test_parse_arguments_defaults_to_cpu(self) -> None:
        with mock.patch.object(sys, "argv", ["chat.py"]):
            args = chat_script.parse_arguments()

        self.assertEqual(args.device, "cpu")

    def test_parse_arguments_accepts_prompt_format_choice(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            ["chat.py", "--prompt-format", "plain"],
        ):
            args = chat_script.parse_arguments()

        self.assertEqual(args.prompt_format, "plain")

    def test_parse_arguments_accepts_qwen3(self) -> None:
        with mock.patch.object(sys, "argv", ["chat.py", "--model", "qwen3"]):
            args = chat_script.parse_arguments()

        self.assertEqual(args.model, "qwen3")
        self.assertIs(
            chat_script._select_prompt_builder(args.model, None),
            chat_script._build_plain_prompt,
        )

    def test_parse_arguments_accepts_generation_policy(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            [
                "chat.py",
                "--temperature",
                "0.8",
                "--top-p",
                "0.95",
                "--top-k",
                "40",
                "--repetition-penalty",
                "1.2",
                "--max-consecutive-repeats",
                "2",
                "--repeat-ngram-size",
                "4",
            ],
        ):
            args = chat_script.parse_arguments()

        self.assertEqual(args.temperature, 0.8)
        self.assertEqual(args.top_p, 0.95)
        self.assertEqual(args.top_k, 40)
        self.assertEqual(args.repetition_penalty, 1.2)
        self.assertEqual(args.max_consecutive_repeats, 2)
        self.assertEqual(args.repeat_ngram_size, 4)

    def test_build_prompt_from_history_uses_chat_role_values(self) -> None:
        prompt = chat_script._build_qwen_prompt(
            [
                ChatMessage(ChatRole.SYSTEM, "You are helpful."),
                ChatMessage(ChatRole.USER, "Hello"),
            ]
        )

        self.assertIn("<|im_start|>system\nYou are helpful.<|im_end|>\n", prompt)
        self.assertIn("<|im_start|>user\nHello<|im_end|>\n", prompt)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))
        self.assertNotIn("ChatRole.", prompt)

    def test_build_prompt_from_history_supports_plain_prompt_format(self) -> None:
        prompt = chat_script._build_plain_prompt(
            [
                ChatMessage(ChatRole.SYSTEM, "You are helpful."),
                ChatMessage(ChatRole.USER, "Hello"),
            ]
        )

        self.assertEqual(
            prompt,
            "You are helpful.\nHello\n",
        )

    def test_session_stores_successful_turn_and_can_clear_it(self) -> None:
        runtime = _FakeRuntime()
        session = ChatSession(runtime, chat_script._build_plain_prompt)

        result = session.reply("Hello")

        self.assertEqual(result.text, "Answer")
        self.assertEqual(
            runtime.prompts,
            ["You are a helpful assistant.\nHello\n"],
        )
        self.assertEqual(session.turn_count, 1)
        self.assertEqual(session.history[-1], ChatMessage(ChatRole.ASSISTANT, "Answer"))

        session.clear()

        self.assertEqual(session.turn_count, 0)
        self.assertEqual(
            session.history,
            (ChatMessage(ChatRole.SYSTEM, "You are a helpful assistant."),),
        )

    def test_session_rolls_back_user_message_when_generation_fails(self) -> None:
        session = ChatSession(
            _FakeRuntime(error=RuntimeError("generation failed")),
            chat_script._build_plain_prompt,
        )

        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            session.reply("Hello")

        self.assertEqual(session.turn_count, 0)
        self.assertEqual(len(session.history), 1)

    def test_runtime_passes_cache_flag_and_stops_on_any_eos_token(self) -> None:
        model = Qwen3TorchModel(_tiny_qwen_config())
        tokenizer = _TestTokenizer(
            encoded_prompt=[1, 2],
            decoded_tokens={3: "ok", 99: "<eos>"},
        )
        runtime = InferenceRuntime(
            model,
            tokenizer,
            device=Device.cpu(),
            eos_token_ids=(99,),
        )

        with mock.patch.object(
            runtime_module, "generate_token_stream", return_value=iter([3, 99])
        ) as generate_token_stream:
            result = runtime.generate(
                "Hello",
                GenerationConfig(max_new_tokens=2, use_kv_cache=True),
            )

        self.assertEqual(result.text, "ok")
        self.assertEqual(result.token_ids, (3,))
        self.assertTrue(generate_token_stream.call_args.kwargs["use_kv_cache"])
        self.assertEqual(
            generate_token_stream.call_args.kwargs["policy"],
            GenerationPolicy(),
        )

    def test_runtime_buffers_incomplete_byte_level_text_until_it_is_valid(self) -> None:
        model = Qwen3TorchModel(_tiny_qwen_config())
        tokenizer = _ByteFragmentTokenizer()
        runtime = InferenceRuntime(
            model,
            tokenizer,
            device=Device.cpu(),
        )
        chunks: list[str] = []

        with mock.patch.object(
            runtime_module, "generate_token_stream", return_value=iter([1, 2])
        ):
            result = runtime.generate(
                "Hello",
                GenerationConfig(max_new_tokens=2),
                on_text=chunks.append,
            )

        self.assertEqual(result.text, "é")
        self.assertEqual(chunks, ["é"])

    def test_runtime_seeds_torch_generation(self) -> None:
        runtime = InferenceRuntime(
            Qwen3TorchModel(_tiny_qwen_config()),
            _TestTokenizer(encoded_prompt=[1], decoded_tokens={}),
            device=Device.cpu(),
        )

        with mock.patch.object(torch, "manual_seed") as manual_seed:
            runtime.set_seed(123)

        manual_seed.assert_called_once_with(123)

    def test_runtime_rejects_negative_generation_seed(self) -> None:
        runtime = InferenceRuntime(
            Qwen3TorchModel(_tiny_qwen_config()),
            _TestTokenizer(encoded_prompt=[1], decoded_tokens={}),
            device=Device.cpu(),
        )

        with self.assertRaisesRegex(ValueError, "seed cannot be negative"):
            runtime.set_seed(-1)


class _FakeRuntime:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.prompts: list[str] = []

    def generate(
        self,
        prompt: str,
        config: GenerationConfig,
        *,
        on_text: Callable[[str], None] | None = None,
    ) -> GenerationResult:
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        return GenerationResult("Answer", (1,), 0.1)


class _CyclingTorchModel(LanguageModelTorch):
    def __init__(
        self, config: LanguageModelConfig, next_token_ids: dict[int, int]
    ) -> None:
        super().__init__()
        self._config = config
        self._next_token_ids = next_token_ids

    def forward(self, idx: torch.Tensor, cache: object = None) -> torch.Tensor:
        next_token_id = self._next_token_ids[int(idx[0, -1])]
        logits = torch.full(
            (idx.shape[0], idx.shape[1], self._config.vocab_size),
            -100.0,
        )
        logits[:, :, next_token_id] = 100.0
        return logits

    def config(self) -> LanguageModelConfig:
        return self._config


class _FixedLogitsTorchModel(LanguageModelTorch):
    def __init__(self, config: LanguageModelConfig, logits: torch.Tensor) -> None:
        super().__init__()
        self._config = config
        self._logits = logits

    def forward(self, idx: torch.Tensor, cache: object = None) -> torch.Tensor:
        return self._logits.repeat(idx.shape[0], idx.shape[1], 1)

    def config(self) -> LanguageModelConfig:
        return self._config


class _TestTokenizer(Tokenizer):
    def __init__(
        self, encoded_prompt: list[int], decoded_tokens: dict[int, str]
    ) -> None:
        self._encoded_prompt = encoded_prompt
        self._decoded_tokens = decoded_tokens

    def encode(self, text: str) -> list[int]:
        return self._encoded_prompt

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self._decoded_tokens[token_id] for token_id in ids)


class _ByteFragmentTokenizer(Tokenizer):
    def encode(self, text: str) -> list[int]:
        return [1]

    def decode(self, ids: Sequence[int]) -> str:
        if ids == [1]:
            return "\ufffd"
        return "é"


if __name__ == "__main__":
    unittest.main()
