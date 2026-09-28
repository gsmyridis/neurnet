from enum import Enum

from neurnet.arch.llm.config import LanguageModelConfig


class GPT2ModelType(Enum):
    SMALL = "gpt2"
    MEDIUM = "gpt2-medium"
    LARGE = "gpt2-large"
    XL = "gpt2-xl"

    def get_config(self) -> LanguageModelConfig:
        match self:
            case GPT2ModelType.SMALL:
                return GPT2_CONFIG_124M
            case GPT2ModelType.MEDIUM:
                return GPT2_CONFIG_350M
            case GPT2ModelType.LARGE:
                return GPT2_CONFIG_774M
            case GPT2ModelType.XL:
                return GPT2_CONFIG_1558M

    def __str__(self) -> str:
        return self.value


GPT2_CONFIG_124M = LanguageModelConfig(
    vocab_size=50_257,
    context_length=1024,
    emb_dim=768,
    n_heads=12,
    n_layers=12,
    hidden_dim=3072,
    head_dim=None,
    # TODO: Remove below
    qk_norm=True,
    n_kv_groups=1,
    rope_base=1_000_000.0,
)
"""GPT2 124 million parameter model config."""

GPT2_CONFIG_350M = LanguageModelConfig(
    vocab_size=50_257,
    context_length=1024,
    emb_dim=1024,
    n_heads=16,
    n_layers=24,
    hidden_dim=4096,
    head_dim=None,
    # TODO: Remove below
    qk_norm=True,
    n_kv_groups=1,
    rope_base=1_000_000.0,
)
"""GPT2 350 million parameter model config."""

GPT2_CONFIG_774M = LanguageModelConfig(
    vocab_size=50_257,
    context_length=1024,
    emb_dim=1280,
    n_heads=20,
    n_layers=36,
    hidden_dim=5120,
    head_dim=None,
    # TODO: Remove below
    qk_norm=True,
    n_kv_groups=1,
    rope_base=1_000_000.0,
)
"""GPT2 774 million parameter model config."""

GPT2_CONFIG_1558M = LanguageModelConfig(
    vocab_size=50_257,
    context_length=1024,
    emb_dim=1600,
    n_heads=25,
    n_layers=48,
    hidden_dim=6400,
    head_dim=None,
    # TODO: Remove below
    qk_norm=True,
    n_kv_groups=1,
    rope_base=1_000_000.0,
)
"""GPT2 1558 million parameter model config."""
