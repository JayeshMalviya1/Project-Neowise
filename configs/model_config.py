from dataclasses import dataclass


@dataclass
class DhruvaConfig:
    """Configuration for Northh Dhruva Foundation Model (Project BHARAT-LLM).

    Default parameters specify the 100M base model architecture with Grouped-Query
    Attention (GQA 5:1 ratio) and SwiGLU feed-forward networks.
    """

    vocab_size: int = 64000
    dim: int = 640
    n_layers: int = 12
    n_heads: int = 10
    n_kv_heads: int = 2  # GQA 5:1 ratio (10 query heads / 2 kv heads)
    intermediate_size: int = 1792  # SwiGLU multiple of 128
    max_seq_len: int = 4096
    rope_theta: float = 500000.0
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        assert (
            self.dim % self.n_heads == 0
        ), f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})"
        assert (
            self.n_heads % self.n_kv_heads == 0
        ), f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"

    @property
    def head_dim(self) -> int:
        """Returns head dimension asserting strict divisibility constraints."""
        assert (
            self.dim % self.n_heads == 0
        ), f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})"
        assert (
            self.n_heads % self.n_kv_heads == 0
        ), f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        return self.dim // self.n_heads

    @classmethod
    def tiny_debug(cls) -> "DhruvaConfig":
        """Fast debug configuration for lightweight testing and quick gradient verification."""
        return cls(
            vocab_size=512,
            dim=128,
            n_layers=4,
            n_heads=4,
            n_kv_heads=2,
            intermediate_size=384,
            max_seq_len=256,
            rope_theta=500000.0,
            rms_norm_eps=1e-5,
            tie_word_embeddings=True,
            initializer_range=0.02,
        )
