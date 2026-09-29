from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Literal

from neurnet.arch.llm.inference import (
    ChatMessage,
    ChatSession,
    GenerationConfig,
    GenerationPolicy,
    GenerationResult,
    InferenceRuntime,
    PromptBuilder,
)
from neurnet.device import Device, DeviceType
from scripts.llm.models import MODEL_CHOICES, QWEN_MODEL_CHOICES, load_model_runtime

PromptFormat = Literal["qwen", "plain"]

DEVICE_CHOICES = [str(device_type) for device_type in DeviceType]
PROMPT_FORMAT_CHOICES: list[PromptFormat] = ["qwen", "plain"]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run LLM text generation (interactive REPL)",
    )
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default="qwen3",
        help="Model to chat with.",
    )
    parser.add_argument(
        "--device",
        choices=DEVICE_CHOICES,
        default=str(DeviceType.CPU),
        help="Requested device. 'gpu' selects the backend's available GPU.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate in each turn.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature; 0 uses greedy decoding.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus-sampling probability mass.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Keep only this many highest-logit tokens before sampling.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Penalty applied to tokens already present in the completion context.",
    )
    parser.add_argument(
        "--max-consecutive-repeats",
        type=int,
        default=3,
        help="Stop before exceeding this many consecutive identical tokens; 0 disables it.",
    )
    parser.add_argument(
        "--repeat-ngram-size",
        type=int,
        default=3,
        help="Stop before repeating an n-gram of this size; 0 disables it.",
    )
    parser.add_argument(
        "--prompt-format",
        choices=PROMPT_FORMAT_CHOICES,
        default=None,
        help="Prompt format. Defaults to Qwen for Qwen and plain for GPT-2.",
    )
    parser.add_argument("--compile", action="store_true", help="Compile the model.")
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="Use the chat-trained Qwen reasoning checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    device = Device(DeviceType.from_string(args.device))
    generation_config = _generation_config_from_args(args)
    runtime = load_model_runtime(
        args.model,
        device,
        compile=args.compile,
        reasoning=args.reasoning,
    )
    session = ChatSession(
        runtime,
        prompt_builder=_select_prompt_builder(
            args.model, args.prompt_format, reasoning=args.reasoning
        ),
        generation_config=generation_config,
    )

    _print_preamble(args, runtime, session, device)
    try:
        _run_repl(session)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")


def _generation_config_from_args(args: argparse.Namespace) -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        policy=GenerationPolicy(
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            max_consecutive_repeats=args.max_consecutive_repeats,
            repeat_ngram_size=args.repeat_ngram_size,
        ),
    )


def _select_prompt_builder(
    model_name: str, prompt_format: PromptFormat | None, *, reasoning: bool = False
) -> PromptBuilder:
    selected_format = prompt_format or (
        "qwen" if model_name in QWEN_MODEL_CHOICES and reasoning else "plain"
    )
    match selected_format:
        case "qwen":
            return _build_qwen_prompt
        case "plain":
            return _build_plain_prompt


def _build_qwen_prompt(history: Sequence[ChatMessage]) -> str:
    parts = [
        f"<|im_start|>{message.role}\n{message.content}<|im_end|>\n"
        for message in history
    ]
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def _build_plain_prompt(history: Sequence[ChatMessage]) -> str:
    parts = [f"{message.content}\n" for message in history]
    return "".join(parts)


def _print_preamble(
    args: argparse.Namespace,
    runtime: InferenceRuntime,
    session: ChatSession,
    device: Device,
) -> None:
    print()
    print("=" * 60)
    print(f"model     : {args.model}")
    print(f"backend   : {runtime.backend()}")
    print(f"device    : {device.dtype}")
    print(f"cache     : {session.generation_config.use_kv_cache}")
    print(f"compile   : {args.compile}")
    if args.model in QWEN_MODEL_CHOICES:
        print(f"reasoning : {args.reasoning}")
        if not args.reasoning:
            print("checkpoint: base text model; use --reasoning for chat")
    print("memory    : True")
    print(f"max_new_tokens (per turn): {session.generation_config.max_new_tokens}")
    policy = session.generation_config.policy
    print(
        "sampling  : "
        f"temperature={policy.temperature}, top_k={policy.top_k}, "
        f"top_p={policy.top_p}, "
        f"repetition_penalty={policy.repetition_penalty}"
    )
    print(
        "guards    : "
        f"consecutive={policy.max_consecutive_repeats}, "
        f"ngram={policy.repeat_ngram_size}"
    )
    print(f"context_length: {runtime.context_length}")
    print("=" * 60)
    print()
    print("Interactive REPL with memory. Type '\\exit' or '\\quit' to quit.")
    print("Commands: \\clear (forget memory), \\history (show turn count)\n")


def _run_repl(session: ChatSession) -> None:
    while True:
        try:
            user_text = input(">> ").strip()
        except EOFError:
            print()
            return

        command = user_text.lower()
        if command in {r"\exit", r"\quit"}:
            return
        if command == r"\clear":
            session.clear()
            print("[Memory cleared!]\n")
            continue
        if command == r"\history":
            print(f"[Stored turns: {session.turn_count}]\n")
            continue
        if not user_text:
            continue

        print("\n" + "-" * 60)
        print("[User]")
        print(user_text + "\n")
        print("[Model]\n", end="", flush=True)
        result = session.reply(user_text, on_text=_print_text_chunk)
        print("\n")
        _print_stats(result)
        print("-" * 60)


def _print_text_chunk(text: str) -> None:
    print(text, end="", flush=True)


def _print_stats(result: GenerationResult) -> None:
    print("[Stats]")
    print(f"\nTime: {result.elapsed_seconds:.2f} sec")
    if result.elapsed_seconds > 0:
        print(f"{int(len(result.token_ids) / result.elapsed_seconds)} tokens/sec")
    else:
        print("0 tokens/sec")


if __name__ == "__main__":
    main()
