from __future__ import annotations

import argparse
import os
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import mlx.core as mx
import torch

from neurnet.arch.llm import (
    ChatMessage,
    ChatRole,
    LanguageModel,
    LanguageModelMLX,
    LanguageModelTorch,
    TensorType,
    Tokenizer,
    generate_stats,
    generate_token_stream,
)
from neurnet.arch.llm.models.gpt2 import GPT2MLXModel, GPT2ModelType, GPT2Tokenizer
from neurnet.arch.llm.models.qwen import Qwen3TorchModel, load_model_and_tokenizer
from neurnet.utils.gpu import get_torch_device

PromptFormat = Literal["qwen", "plain"]

GPT2_MODEL_CHOICES = [model_type.value for model_type in GPT2ModelType]
CHOICES = ["qwen3", *GPT2_MODEL_CHOICES]

DEVICE_CHOICES = ["cpu", "gpu"]

PROMPT_FORMAT_CHOICES: list[PromptFormat] = ["qwen", "plain"]


@dataclass(frozen=True)
class ChatRuntime:
    model: LanguageModel
    tokenizer: Tokenizer
    device: torch.device | None
    eos_token_ids: tuple[int, ...]
    prompt_format: PromptFormat
    use_kv_cache: bool
    backend_name: str


# ===-----------------------------------------------------------------------===
# Parse arguments
# ===-----------------------------------------------------------------------===


def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run LLM text generation (interactive REPL)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=CHOICES,
        default="qwen3",
        help="Model to chat with.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=DEVICE_CHOICES,
        default="cpu",
        help="Device to run Qwen on. 'gpu' auto-detects the available GPU backend. "
        "Ignored for GPT-2, which currently uses MLX.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate in each turn.",
    )
    parser.add_argument(
        "--prompt-format",
        choices=PROMPT_FORMAT_CHOICES,
        default=None,
        help="Prompt format. Defaults to Qwen for Qwen and plain for GPT-2.",
    )
    parser.add_argument("--compile", action="store_true", help="Compile Torch model.")
    parser.add_argument(
        "--reasoning", action="store_true", help="Use Qwen reasoning model variant."
    )
    return parser.parse_args()


def main() -> None:

    # ===-------------------------------------------------------------------===
    # Parse arguments
    # ===-------------------------------------------------------------------===
    args = parse_arguments()

    # ===-------------------------------------------------------------------===
    # Load tokenizer and model
    # ===-------------------------------------------------------------------===
    runtime = load_chat_runtime(args)
    runtime.model.eval()
    max_new_tokens = args.max_new_tokens

    # ===-------------------------------------------------------------------===
    # Print preamble
    # ===-------------------------------------------------------------------===
    print()
    print("=" * 60)
    print(f"model     : {args.model}")
    print(f"backend   : {runtime.backend_name}")
    print(f"torch     : {torch.__version__}")
    if runtime.device is not None:
        print(f"device    : {runtime.device}")
    print(f"cache     : {runtime.use_kv_cache}")
    print(f"compile   : {args.compile and args.model == 'qwen3'}")
    if args.model == "qwen3":
        print(f"reasoning : {args.reasoning}")
    print("memory    : True")
    print(f"max_new_tokens (per turn): {max_new_tokens}")
    print(f"context_length: {runtime.model.config().context_length}")
    print("=" * 60)
    print()
    print("Interactive REPL with memory. Type '\\exit' or '\\quit' to quit.")
    print("Commands: \\clear (forget memory), \\history (show turn count)\n")

    history: list[ChatMessage] = [
        ChatMessage(ChatRole.SYSTEM, "You are a helpful assistant.")
    ]

    # Interactive REPL (Read, Evaluate, Print, Loop)
    try:
        while True:
            try:
                user_in = input(">> ").strip()
            except EOFError:
                print()
                break

            low = user_in.lower()

            # Exit
            if low in {r"\exit", r"\quit"}:
                break

            # Clear history
            if low == r"\clear":
                # Reset history but keep the system prompt
                system_entries = [m for m in history if m.role == ChatRole.SYSTEM]
                history.clear()
                if system_entries:
                    history.extend(system_entries)
                else:
                    history.append(
                        ChatMessage(ChatRole.SYSTEM, "You are a helpful assistant.")
                    )
                print("[Memory cleared!]\n")
                continue

            if low == r"\history":
                # Count assistant turns as the number of model replies so far
                assistant_turns = sum(
                    1 for m in history if m.role == ChatRole.ASSISTANT
                )
                print(f"[Stored turns: {assistant_turns}]\n")
                continue
            if not user_in:
                continue

            print("\n" + "-" * 60)
            print("[User]")
            print(user_in + "\n")
            run_generate(
                runtime.model,
                runtime.tokenizer,
                runtime.device,
                history,
                user_in,
                max_new_tokens,
                runtime.eos_token_ids,
                prompt_format=runtime.prompt_format,
                use_kv_cache=runtime.use_kv_cache,
            )

    except KeyboardInterrupt:
        print("\nInterrupted by user.")


def load_chat_runtime(args: argparse.Namespace) -> ChatRuntime:
    match args.model:
        case "qwen3":
            device = torch.device("cpu") if args.device == "cpu" else get_torch_device()
            qwen_dir = os.path.join("models", "qwen3")
            qwen_model_type = "reasoning" if args.reasoning else "base"
            model, tokenizer = load_model_and_tokenizer(
                qwen_model_type,
                device,
                False,
                qwen_dir,
            )

            if args.compile:
                model = cast(Qwen3TorchModel, torch.compile(model))

            return ChatRuntime(
                model=model,
                tokenizer=tokenizer,
                device=device,
                eos_token_ids=_eos_token_ids(
                    tokenizer,
                    extra_tokens=("<|im_end|>", "<|endoftext|>"),
                ),
                prompt_format=args.prompt_format or "qwen",
                use_kv_cache=True,
                backend_name="torch",
            )

        case model_name if model_name in GPT2_MODEL_CHOICES:
            if args.device == "gpu":
                warnings.warn(
                    f"--device is ignored for --model={model_name}; GPT-2 uses MLX."
                )
            if args.compile:
                warnings.warn(
                    f"--compile is ignored for --model={model_name}; GPT-2 uses MLX."
                )

            model_type = GPT2ModelType(model_name)
            gpt2_dir = os.path.join("models", "gpt2")
            model = GPT2MLXModel.from_pretrained(model_type, cache_dir=gpt2_dir)
            tokenizer = GPT2Tokenizer.from_pretrained(model_type, cache_dir=gpt2_dir)

            return ChatRuntime(
                model=model,
                tokenizer=tokenizer,
                device=None,
                eos_token_ids=_eos_token_ids(tokenizer),
                prompt_format=args.prompt_format or "plain",
                use_kv_cache=True,
                backend_name="mlx",
            )

    raise ValueError(f"Unsupported model: {args.model}")


def run_generate(
    model: LanguageModel,
    tokenizer: Tokenizer,
    device: torch.device | None,
    history: list[ChatMessage],
    user_text: str,
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...],
    *,
    prompt_format: PromptFormat = "qwen",
    use_kv_cache: bool = True,
) -> str:
    # Add user prompt to history
    history.append(ChatMessage(ChatRole.USER, user_text))

    # Encode full history
    prompt = build_prompt_from_history(
        history,
        add_assistant_header=True,
        prompt_format=prompt_format,
    )
    input_ids = tokenizer.encode(prompt)
    input_token_ids_tensor = _make_input_token_ids(model, input_ids, device)

    # Left-truncate (to make space for generation)
    input_token_ids_tensor = trim_input_tensor(
        input_ids_tensor=input_token_ids_tensor,
        context_len=model.config().context_length,
        max_new_tokens=max_new_tokens,
    )

    start_time = time.time()
    all_token_ids: list[int] = []

    print("[Model]\n", end="", flush=True)
    for token_id in generate_token_stream(
        model=model,
        token_ids=input_token_ids_tensor,
        max_new_tokens=max_new_tokens,
        use_kv_cache=use_kv_cache,
    ):
        if token_id in eos_token_ids:
            break
        piece = tokenizer.decode([token_id])
        print(piece, end="", flush=True)
        all_token_ids.append(token_id)

    end_time = time.time()
    print("\n")

    print("[Stats]")
    generate_stats(torch.tensor(all_token_ids), tokenizer, start_time, end_time)
    print("-" * 60)

    # Add model reply to history
    assistant_text = tokenizer.decode(all_token_ids)
    history.append(ChatMessage(ChatRole.ASSISTANT, assistant_text))
    return assistant_text


def build_prompt_from_history(
    history: list[ChatMessage],
    add_assistant_header: bool = True,
    *,
    prompt_format: PromptFormat = "qwen",
) -> str:
    match prompt_format:
        case "qwen":
            return _build_qwen_prompt_from_history(history, add_assistant_header)
        case "plain":
            return _build_plain_prompt_from_history(history, add_assistant_header)


def trim_input_tensor(
    input_ids_tensor: TensorType, context_len: int, max_new_tokens: int
) -> TensorType:
    assert max_new_tokens < context_len
    keep_len = max(1, context_len - max_new_tokens)

    # If the prompt is too long, left-truncate to keep_len
    if input_ids_tensor.shape[1] > keep_len:
        input_ids_tensor = input_ids_tensor[:, -keep_len:]

    return input_ids_tensor


def _build_qwen_prompt_from_history(
    history: list[ChatMessage], add_assistant_header: bool
) -> str:
    parts = []
    for message in history:
        role = message.role.value
        content = message.content
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

    if add_assistant_header:
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def _build_plain_prompt_from_history(
    history: list[ChatMessage], add_assistant_header: bool
) -> str:
    role_labels = {
        ChatRole.SYSTEM: "System",
        ChatRole.USER: "User",
        ChatRole.ASSISTANT: "Assistant",
    }

    parts = []
    for message in history:
        parts.append(f"{role_labels[message.role]}: {message.content}\n\n")

    if add_assistant_header:
        parts.append("Assistant:")
    return "".join(parts)


def _make_input_token_ids(
    model: LanguageModel,
    input_ids: Sequence[int],
    device: torch.device | None,
) -> TensorType:
    if isinstance(model, LanguageModelMLX):
        return mx.array([input_ids])
    if isinstance(model, LanguageModelTorch):
        if device is None:
            raise ValueError("Torch models require a device")
        return torch.tensor(input_ids, device=device).unsqueeze(0)

    raise TypeError(f"Unsupported LanguageModel type: {type(model)}")


def _eos_token_ids(
    tokenizer: Any, *, extra_tokens: Sequence[str] = ()
) -> tuple[int, ...]:
    eos_token_ids = []

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        eos_token_ids.append(int(eos_token_id))

    for token in extra_tokens:
        ids = tokenizer.encode(token)
        if ids:
            eos_token_ids.append(int(ids[0]))

    return tuple(dict.fromkeys(eos_token_ids))


if __name__ == "__main__":
    main()
