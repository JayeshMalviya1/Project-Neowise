import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.model_config import DhruvaConfig


class SwiGLUMLP(nn.Module):
    """SwiGLU Multi-Layer Perceptron (feed-forward network).

    Computes: down_proj(silu(gate_proj(x)) * up_proj(x)) using bias-free projections.
    """

    def __init__(self, config: DhruvaConfig):
        super().__init__()
        self.dim = config.dim
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.dim, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.dim, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
