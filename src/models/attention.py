from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.model_config import DhruvaConfig
from src.models.rope import RotaryEmbedding, apply_rotary_pos_emb


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention (GQA) with Rotary Position Embeddings.

    Projects inputs into query, key, and value spaces, applies RoPE to queries and keys,
    expands key/value heads across query groups via repeat_interleave, and computes causal
    attention using PyTorch's native scaled_dot_product_attention.
    """

    def __init__(self, config: DhruvaConfig, rope: Optional[RotaryEmbedding] = None):
        super().__init__()
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.n_heads // self.n_kv_heads

        # Bias-free linear projections
        self.q_proj = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)

        self.rope = (
            rope
            if rope is not None
            else RotaryEmbedding(
                dim=self.head_dim,
                max_seq_len=config.max_seq_len,
                theta=config.rope_theta,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[ tuple ] = None,
        use_cache: bool = False,
    ):
        batch_size, seq_len, _ = x.shape

        # Reshape queries to (B, n_heads, S, head_dim) and keys/values to (B, n_kv_heads, S, head_dim)
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        offset = past_kv[0].shape[2] if past_kv is not None else 0

        # Apply RoPE to q and k with positional offset
        cos, sin = self.rope(seq_len, device=x.device, dtype=x.dtype, offset=offset)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        present_kv = (k, v) if use_cache else None

        # Expand k and v across query groups using repeat_interleave
        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # Causal scaled dot-product attention (single-step decode with past_kv attends to all cached keys)
        is_causal = (past_kv is None) and (seq_len > 1)
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        # Transpose and project back to hidden dimension
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.o_proj(attn_output)
        if use_cache:
            return out, present_kv
        return out
