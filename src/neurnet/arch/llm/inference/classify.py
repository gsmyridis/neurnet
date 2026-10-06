import mlx.core as mx
from mlx import nn

from neurnet.arch.llm import Tokenizer


def classify_id(
    model: nn.Module, tokenizer: Tokenizer, text: str, max_sequence_len: int
) -> int:
    tokens_list = list(tokenizer.encode(text))
    padding = max_sequence_len - len(tokens_list)
    tokens_list += [tokenizer.end_of_sequence_token_id()] * padding
    tokens = mx.array([tokens_list])

    model.eval()
    logits = model(tokens)[:, -1, :]
    class_idx = mx.argmax(logits, axis=1).item()

    assert isinstance(class_idx, int)
    return class_idx


def classify_label(
    model: nn.Module,
    tokenizer: Tokenizer,
    text: str,
    max_sequence_len: int,
    labels: tuple[str, ...],
) -> str:
    idx = classify_id(model, tokenizer, text, max_sequence_len)
    return labels[idx]
