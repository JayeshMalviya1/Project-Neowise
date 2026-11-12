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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Reshape queries to (B, n_heads, S, head_dim) and keys/values to (B, n_kv_heads, S, head_dim)
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to q and k
        cos, sin = self.rope(seq_len, device=x.device, dtype=x.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Expand k and v across query groups using repeat_interleave
        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # Causal scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Transpose and project back to hidden dimension
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)
