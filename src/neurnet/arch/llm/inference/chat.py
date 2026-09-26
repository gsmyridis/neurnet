from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum

from .runtime import GenerationConfig, GenerationResult, InferenceRuntime


class ChatRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class ChatMessage:
    role: ChatRole
    content: str


type PromptBuilder = Callable[[Sequence[ChatMessage]], str]


class ChatSession:
    """A stateful chat session over an inference runtime."""

    def __init__(
        self,
        runtime: InferenceRuntime,
        prompt_builder: PromptBuilder,
        *,
        system_prompt: str = "You are a helpful assistant.",
        generation_config: GenerationConfig | None = None,
    ) -> None:
        self._runtime = runtime
        self._prompt_builder = prompt_builder
        self._system_message = ChatMessage(ChatRole.SYSTEM, system_prompt)
        self._history = [self._system_message]
        self._generation_config = generation_config or GenerationConfig()

    @property
    def history(self) -> tuple[ChatMessage, ...]:
        return tuple(self._history)

    @property
    def generation_config(self) -> GenerationConfig:
        return self._generation_config

    @property
    def turn_count(self) -> int:
        return sum(message.role == ChatRole.ASSISTANT for message in self._history)

    def clear(self) -> None:
        self._history = [self._system_message]

    def reply(
        self,
        user_text: str,
        *,
        on_text: Callable[[str], None] | None = None,
    ) -> GenerationResult:
        self._history.append(ChatMessage(ChatRole.USER, user_text))
        prompt = self._prompt_builder(self._history)
        try:
            result = self._runtime.generate(
                prompt,
                self._generation_config,
                on_text=on_text,
            )
        except Exception:
            self._history.pop()
            raise

        self._history.append(ChatMessage(ChatRole.ASSISTANT, result.text))
        return result
