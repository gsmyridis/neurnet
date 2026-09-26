from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import nn

from neurnet.arch.llm.config import LanguageModelConfig
from neurnet.arch.llm.kv_cache import KVCacheTorch, LayerCacheTorch
from neurnet.arch.llm.types import LanguageModelTorch

# ===-----------------------------------------------------------------------===
# Config
# ===-----------------------------------------------------------------------===

QWEN_CONFIG_06_B: LanguageModelConfig = LanguageModelConfig(
    vocab_size=151_936,
    context_length=40_960,
    emb_dim=1024,
    n_heads=16,
    n_layers=28,
    hidden_dim=3072,
    head_dim=128,
    qk_norm=True,
    n_kv_groups=8,
    rope_base=1_000_000.0,
)
"""Qwen 600 million parameter model config."""

# ===-----------------------------------------------------------------------===
# Model
# ===-----------------------------------------------------------------------===


class Qwen3TorchModel(LanguageModelTorch):
    cfg: LanguageModelConfig
    cos: torch.Tensor
    sin: torch.Tensor

    def __init__(
        self, cfg: LanguageModelConfig, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        super().__init__()

        # Config
        self.cfg: LanguageModelConfig = cfg
        self._dtype = dtype
        self.current_cache_pos = 0  # Track current position in KV cache

        # Reusable utilities
        cos, sin = compute_rope_params(
            head_dim=cfg.effective_head_dim,
            theta_base=cfg.rope_base,
            context_length=cfg.context_length,
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # Main model parameters
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim, dtype=dtype)
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg, dtype) for _ in range(cfg.n_layers)]
        )
        self.final_norm = RMSNorm(cfg.emb_dim)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False, dtype=dtype)

    def forward(
        self, idx: torch.Tensor, cache: KVCacheTorch | None = None
    ) -> torch.Tensor:
        # Forward pass
        tok_embeds = self.tok_emb(idx)
        x = tok_embeds

        num_tokens = x.shape[1]
        if cache is not None:
            pos_start = self.current_cache_pos
            pos_end = pos_start + num_tokens
            self.current_cache_pos = pos_end
            mask = torch.triu(
                torch.ones(pos_end, pos_end, device=x.device, dtype=torch.bool),
                diagonal=1,
            )[pos_start:pos_end, :pos_end]
        else:
            pos_start = 0  # Not strictly necessary but helps torch.compile TODO: Why?
            mask = torch.triu(
                torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool),
                diagonal=1,
            )

        # Prefill (no cache): mask starts as (num_tokens, num_tokens)
        # Cached decoding: mask starts as (num_tokens, prev_k_number_tokens + num_tokens)
        #
        # We add two leading dimensions so the mask becomes
        # (1, 1, num_tokens, num_tokens) during prefill and
        # (1, 1, num_tokens, total_key_tokens) during cached decoding.
        # These extra dimensions let PyTorch broadcast the same mask
        # across all batches and attention heads when applying it to
        # attn_scores of shape (batch, num_heads, num_tokens, total_key_tokens).
        mask = mask[None, None, :, :]  # broadcast mask

        for i, block in enumerate(self.trf_blocks):
            blk_cache = cache.get(i) if cache else None
            x, new_blk_cache = block(
                x, mask, self.cos, self.sin, start_pos=pos_start, cache=blk_cache
            )
            if cache is not None:
                cache.update(i, new_blk_cache)

        x = self.final_norm(x)
        logits = self.out_head(x.to(self._dtype))
        return logits

    def reset_kv_cache(self) -> None:
        self.current_cache_pos = 0

    def config(self) -> LanguageModelConfig:
        return self.cfg


# ===-----------------------------------------------------------------------===
# Transformer
# ===-----------------------------------------------------------------------===


class TransformerBlock(nn.Module):
    def __init__(self, cfg: LanguageModelConfig, dtype: torch.dtype) -> None:
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg.emb_dim,
            num_heads=cfg.n_heads,
            head_dim=cfg.effective_head_dim,
            num_kv_groups=cfg.n_kv_groups,
            qk_norm=cfg.qk_norm,
            dtype=dtype,
        )
        self.ff = FeedForward(cfg, dtype)
        self.norm1 = RMSNorm(cfg.emb_dim, eps=1e-6)
        self.norm2 = RMSNorm(cfg.emb_dim, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos: int = 0,
        cache: LayerCacheTorch | None = None,
    ) -> tuple[torch.Tensor, LayerCacheTorch]:
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x, next_cache = self.att(
            x, mask, cos, sin, start_pos=start_pos, cache=cache
        )  # Shape [batch_size, num_tokens, emb_size]
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut  # Add the original input back

        return x, next_cache


class FeedForward(nn.Module):
    def __init__(self, cfg: LanguageModelConfig, dtype: torch.dtype) -> None:
        super().__init__()
        self.fc1 = nn.Linear(cfg.emb_dim, cfg.hidden_dim, dtype=dtype, bias=False)
        self.fc2 = nn.Linear(cfg.emb_dim, cfg.hidden_dim, dtype=dtype, bias=False)
        self.fc3 = nn.Linear(cfg.hidden_dim, cfg.emb_dim, dtype=dtype, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        num_heads: int,
        num_kv_groups: int,
        head_dim: int | None = None,
        qk_norm: bool = False,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        assert num_heads % num_kv_groups == 0, (
            "num_heads must be divisible by num_kv_groups"
        )

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            assert d_in % num_heads == 0, (
                "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            )
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(
            d_in, num_kv_groups * head_dim, bias=False, dtype=dtype
        )

        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos: int = 0,
        cache: LayerCacheTorch | None = None,
    ) -> tuple[torch.Tensor, LayerCacheTorch]:
        b, num_tokens, _ = x.shape

        # Apply projections
        queries = self.W_query(x)  # (b, num_tokens, num_heads * head_dim)
        keys = self.W_key(x)  # (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)  # (b, num_tokens, num_kv_groups * head_dim)

        # Reshape to heads / kv-groups
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        keys_new = keys.view(
            b, num_tokens, self.num_kv_groups, self.head_dim
        ).transpose(1, 2)
        values_new = values.view(
            b, num_tokens, self.num_kv_groups, self.head_dim
        ).transpose(1, 2)

        # Optional normalization
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys_new = self.k_norm(keys_new)

        # Apply RoPE
        queries = apply_rope(queries, cos, sin, offset=start_pos)
        keys_new = apply_rope(keys_new, cos, sin, offset=start_pos)

        if cache is not None:
            prev_k, prev_v = cache
            keys = torch.cat([prev_k, keys_new], dim=2)
            values = torch.cat([prev_v, values_new], dim=2)
        else:
            start_pos = 0  # reset RoPE
            keys, values = keys_new, values_new
        next_cache = (keys, values)

        # Expand K and V to match number of heads
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        # Attention
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)

        context = (
            (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        )
        return self.out_proj(context), next_cache


# ===-----------------------------------------------------------------------===
# RoPE
# ===-----------------------------------------------------------------------===

# ==============================================================================
# RoPE implementation summary
#
#
# There are two common styles to implement RoPE, which are
# mathematically equivalent;
# they mainly differ in how the rotation matrix pairs dimensions.
#
# 1) Split-halves style (this repo, Hugging Face Transformers):
#
#   For hidden dim d = 4 (example):
#
#       [ x0   x1 | x2   x3 ]
#         │    │    │    │
#         ▼    ▼    ▼    ▼
#        cos  cos  sin  sin
#
#   Rotation matrix:
#
#       [ cosθ0   0    -sinθ0   0   ]
#       [  0    cosθ1    0    -sinθ1]
#       [ sinθ0   0     cosθ0   0   ]
#       [  0    sinθ1    0     cosθ1]
#
#   Here, the embedding dims are split into two halves and then
#   each one is rotated in blocks.
#
#
# 2) Interleaved (even/odd) style (original paper, Llama repo):
#
#   For hidden dim d = 4 (example):
#
#       [ x0   x1   x2   x3 ]
#         │    │    │    │
#         ▼    ▼    ▼    ▼
#        cos  sin  cos  sin
#
#   Rotation matrix:
#
#       [ cosθ0  -sinθ0   0       0    ]
#       [ sinθ0   cosθ0   0       0    ]
#       [  0        0    cosθ1  -sinθ1 ]
#       [  0        0    sinθ1   cosθ1 ]
#
#
#   Here, embedding dims are interleaved as even/odd cosine/sine pairs.
#
# Both layouts encode the same relative positions; the only difference is how
# dimensions are paired.
# ==============================================================================


def compute_rope_params(
    head_dim: int,
    theta_base: float = 10_000,
    context_length: int = 4096,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (
        theta_base
        ** (
            torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float()
            / head_dim
        )
    )

    # Generate position indices
    positions = torch.arange(context_length, dtype=dtype)

    # Compute the angles
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(
        0
    )  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0
) -> torch.Tensor:
    # x: (batch_size, num_heads, seq_len, head_dim)
    _batch_size, _num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2 :]  # Second half

    # Adjust sin and cos shapes
    cos = (
        cos[offset : offset + seq_len, :].unsqueeze(0).unsqueeze(0)
    )  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[offset : offset + seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    # It's ok to use lower-precision after applying cos and sin rotation
    return x_rotated.to(dtype=x.dtype)


# ===-----------------------------------------------------------------------===
# RMSNorm
# ===-----------------------------------------------------------------------===


class RMSNorm(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        eps: float = 1e-6,
        bias: bool = False,
        qwen3_compatible: bool = True,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype

        if self.qwen3_compatible:
            x = x.to(torch.float32)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale

        if self.shift is not None:
            norm_x = norm_x + self.shift

        return norm_x.to(input_dtype)


def load_hf_weights_into_qwen(
    model: Qwen3TorchModel,
    param_config: LanguageModelConfig,
    params: Mapping[str, torch.Tensor],
) -> None:
    """
    Only used in Appendix D for loading the other Qwen3 variants.
    """

    def assign[TensorT: torch.Tensor](
        left: TensorT,
        right: torch.Tensor,
        tensor_name: str = "unknown",
    ) -> TensorT:
        if left.shape != right.shape:
            raise ValueError(
                f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}"
            )

        with torch.no_grad():
            if isinstance(right, torch.Tensor):
                left.copy_(right)
            else:
                left.copy_(torch.as_tensor(right, dtype=left.dtype, device=left.device))

        return left

    model.tok_emb.weight = assign(
        model.tok_emb.weight,
        params["model.embed_tokens.weight"],
        "model.embed_tokens.weight",
    )

    for l in range(param_config.n_layers):
        block = cast(TransformerBlock, model.trf_blocks[l])
        att = block.att

        # Q, K, V projections
        att.W_query.weight = assign(
            att.W_query.weight,
            params[f"model.layers.{l}.self_attn.q_proj.weight"],
            f"model.layers.{l}.self_attn.q_proj.weight",
        )
        att.W_key.weight = assign(
            att.W_key.weight,
            params[f"model.layers.{l}.self_attn.k_proj.weight"],
            f"model.layers.{l}.self_attn.k_proj.weight",
        )
        att.W_value.weight = assign(
            att.W_value.weight,
            params[f"model.layers.{l}.self_attn.v_proj.weight"],
            f"model.layers.{l}.self_attn.v_proj.weight",
        )

        # Output projection
        att.out_proj.weight = assign(
            att.out_proj.weight,
            params[f"model.layers.{l}.self_attn.o_proj.weight"],
            f"model.layers.{l}.self_attn.o_proj.weight",
        )

        # QK norms
        if hasattr(att, "q_norm") and att.q_norm is not None:
            att.q_norm.scale = assign(
                att.q_norm.scale,
                params[f"model.layers.{l}.self_attn.q_norm.weight"],
                f"model.layers.{l}.self_attn.q_norm.weight",
            )
        if hasattr(att, "k_norm") and att.k_norm is not None:
            att.k_norm.scale = assign(
                att.k_norm.scale,
                params[f"model.layers.{l}.self_attn.k_norm.weight"],
                f"model.layers.{l}.self_attn.k_norm.weight",
            )

        # Attention layernorm
        block.norm1.scale = assign(
            block.norm1.scale,
            params[f"model.layers.{l}.input_layernorm.weight"],
            f"model.layers.{l}.input_layernorm.weight",
        )

        # Feedforward weights
        if num_experts := cast(int | None, getattr(param_config, "num_experts", None)):
            ff = cast(Any, block.ff)
            # Load router (gating) weights
            ff.gate.weight = assign(
                ff.gate.weight,
                params[f"model.layers.{l}.mlp.gate.weight"],
                f"model.layers.{l}.mlp.gate.weight",
            )
            # Load expert weights
            for e in range(num_experts):
                prefix = f"model.layers.{l}.mlp.experts.{e}"
                ff.fc1[e].weight = assign(
                    ff.fc1[e].weight,
                    params[f"{prefix}.gate_proj.weight"],
                    f"{prefix}.gate_proj.weight",
                )
                ff.fc2[e].weight = assign(
                    ff.fc2[e].weight,
                    params[f"{prefix}.up_proj.weight"],
                    f"{prefix}.up_proj.weight",
                )
                ff.fc3[e].weight = assign(
                    ff.fc3[e].weight,
                    params[f"{prefix}.down_proj.weight"],
                    f"{prefix}.down_proj.weight",
                )
                # After assigning weights, move the expert layers from meta to CPU
                ff.fc1[e] = ff.fc1[e].to("cpu")
                ff.fc2[e] = ff.fc2[e].to("cpu")
                ff.fc3[e] = ff.fc3[e].to("cpu")

        else:
            block.ff.fc1.weight = assign(
                block.ff.fc1.weight,
                params[f"model.layers.{l}.mlp.gate_proj.weight"],
                f"model.layers.{l}.mlp.gate_proj.weight",
            )
            block.ff.fc2.weight = assign(
                block.ff.fc2.weight,
                params[f"model.layers.{l}.mlp.up_proj.weight"],
                f"model.layers.{l}.mlp.up_proj.weight",
            )
            block.ff.fc3.weight = assign(
                block.ff.fc3.weight,
                params[f"model.layers.{l}.mlp.down_proj.weight"],
                f"model.layers.{l}.mlp.down_proj.weight",
            )

        block.norm2.scale = assign(
            block.norm2.scale,
            params[f"model.layers.{l}.post_attention_layernorm.weight"],
            f"model.layers.{l}.post_attention_layernorm.weight",
        )

    # Final normalization and output head
    model.final_norm.scale = assign(
        model.final_norm.scale, params["model.norm.weight"], "model.norm.weight"
    )

    if "lm_head.weight" in params:
        model.out_head.weight = assign(
            model.out_head.weight, params["lm_head.weight"], "lm_head.weight"
        )
    else:
        model.out_head.weight = model.tok_emb.weight
        print("Model uses weight tying.")
