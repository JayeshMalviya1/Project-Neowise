from typing import Tuple
import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding (RoPE) to query and key tensors.

    Args:
        q: Query tensor of shape (batch, n_heads, seq_len, head_dim)
        k: Key tensor of shape (batch, n_kv_heads, seq_len, head_dim)
        cos: Cosine cache sliced to seq_len of shape (1, 1, seq_len, head_dim)
        sin: Sine cache sliced to seq_len of shape (1, 1, seq_len, head_dim)
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE).

    Precomputes and caches inverse frequencies and cos/sin position projections,
    supporting dynamic sequence extension beyond the pre-allocated max_seq_len.
    """

    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 500000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Precompute inverse frequencies: 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(self.max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        self.max_seq_len = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim // 2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, dim)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            self.cos_cached is None
            or seq_len > self.cos_cached.shape[2]
            or self.cos_cached.device != device
        ):
            target_len = max(seq_len, self.max_seq_len)
            self._build_cache(target_len)

        return (
            self.cos_cached[:, :, :seq_len, :].to(dtype=dtype, device=device),
            self.sin_cached[:, :, :seq_len, :].to(dtype=dtype, device=device),
        )
