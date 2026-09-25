from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class LanguageModelConfig:
    vocab_size: int
    """Vocabulary size."""
    context_length: int
    """Length originally used during training."""
    emb_dim: int
    """Embedding dimension."""
    n_heads: int
    """Number of attention heads."""
    n_layers: int
    """Number of layers."""
    hidden_dim: int
    """Size of intermediate dim in FeedForward."""
    head_dim: int | None
    """Size of the heads in GQA."""
    qk_norm: bool
    """Whether to normalize queries & keys in GQA."""
    n_kv_groups: int
    """Key-Value groups for GQA."""
    rope_base: float
    """The base in RoPE's "theta"."""

    @property
    def effective_head_dim(self) -> int:
        if self.head_dim is not None:
            return self.head_dim
        return self.emb_dim // self.n_heads

    def __post_init__(self) -> None:
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.n_kv_groups <= 0:
            raise ValueError("n_kv_groups must be positive")
        if self.n_heads % self.n_kv_groups != 0:
            raise ValueError("n_heads must be divisible by n_kv_groups")
        if self.head_dim is None and self.emb_dim % self.n_heads != 0:
            raise ValueError(
                "emb_dim must be divisible by n_heads when head_dim is omitted"
            )
        if self.effective_head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
