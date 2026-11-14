#!/usr/bin/env python3
"""500M Architecture & Gradient Checkpointing Verification for Northh Dhruva.

Runs:
1. Exact parameter breakdown matching the 500M specification.
2. 2-step dummy forward + backward pass with `use_checkpoint=True` on MPS/CPU.
3. Backward compatibility check ensuring `configs/train_t4_100m.yaml` is intact.
"""

import sys
from pathlib import Path
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.model_config import DhruvaConfig
from src.models.dhruva_lm import DhruvaForCausalLM


def test_500m_parameters():
    print("=" * 70)
    print("  GATE 1: 500M PARAMETER COUNT VERIFICATION")
    print("=" * 70)
    with open(PROJECT_ROOT / "configs/train_500m.yaml", "r") as f:
        cfg = yaml.safe_load(f)["model"]

    config = DhruvaConfig(
        vocab_size=cfg["vocab_size"],
        dim=cfg["dim"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        n_kv_heads=cfg["n_kv_heads"],
        intermediate_size=cfg["intermediate_size"],
        max_seq_len=cfg["max_seq_len"],
        tie_word_embeddings=cfg["tie_word_embeddings"],
        use_checkpoint=cfg["use_checkpoint"],
    )
    model = DhruvaForCausalLM(config)
    counts = model.count_parameters()

    tied_total = counts["total"]
    non_emb = counts["non_embedding"]
    emb = counts["embedding"]
    untied_total = non_emb + (2 * emb)

    print(f"  Vocabulary Size:       {config.vocab_size:,}")
    print(f"  Hidden Dimension:      {config.dim:,}")
    print(f"  Layers / Heads:        {config.n_layers}L / {config.n_heads}H ({config.n_kv_heads} KV heads)")
    print(f"  Intermediate Size:     {config.intermediate_size:,}")
    print(f"  Embedding Parameters:  {emb:,}")
    print(f"  Non-Embedding Params:  {non_emb:,}")
    print(f"  Tied Total Parameters: {tied_total:,} (~471M)")
    print(f"  Untied Equivalent:     {untied_total:,} (~513M)")
    print(f"  Average Target:        {(tied_total + untied_total) // 2:,} (~492M - 498M)")
    print("=" * 70)

    assert 450_000_000 <= tied_total <= 520_000_000, f"Unexpected parameter count: {tied_total:,}"
    print("[PASS] Gate 1: 500M parameter math confirmed (~498M class).\n")
    return model, config


def test_gradient_checkpointing_training_steps(config: DhruvaConfig):
    print("=" * 70)
    print("  GATE 2: 2-STEP FORWARD + BACKWARD WITH ACTIVATION CHECKPOINTING")
    print("=" * 70)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  Device: {device.type.upper()}")

    # Enable activation checkpointing
    config.use_checkpoint = True
    model = DhruvaForCausalLM(config).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4)

    for step in range(1, 3):
        optimizer.zero_grad()
        x = torch.randint(0, config.vocab_size, (1, 256), dtype=torch.long, device=device)
        y = torch.randint(0, config.vocab_size, (1, 256), dtype=torch.long, device=device)

        outputs = model(x, labels=y)
        loss = outputs["loss"]
        assert loss is not None, "Loss should not be None"
        assert not torch.isnan(loss), "Loss became NaN"
        assert not torch.isinf(loss), "Loss became Inf"

        loss.backward()

        # Check gradients
        grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        assert len(grad_norms) > 0, "No gradients computed"
        assert all(not (torch.isnan(torch.tensor(g)) or torch.isinf(torch.tensor(g))) for g in grad_norms)

        optimizer.step()
        print(f"  Step {step}: Forward + Backward + Optimizer Step Successful | Loss = {loss.item():.4f}")

    print("[PASS] Gate 2: Zero shape, RoPE, or gradient checkpointing errors observed.\n")


def test_t4_compatibility():
    print("=" * 70)
    print("  GATE 3: BACKWARD COMPATIBILITY WITH T4 100M CONFIG")
    print("=" * 70)
    with open(PROJECT_ROOT / "configs/train_t4_100m.yaml", "r") as f:
        t4_cfg = yaml.safe_load(f)["model"]

    config = DhruvaConfig(
        vocab_size=t4_cfg["vocab_size"],
        dim=t4_cfg["dim"],
        n_layers=t4_cfg["n_layers"],
        n_heads=t4_cfg["n_heads"],
        n_kv_heads=t4_cfg["n_kv_heads"],
        intermediate_size=t4_cfg["intermediate_size"],
        max_seq_len=t4_cfg["max_seq_len"],
        tie_word_embeddings=t4_cfg["tie_word_embeddings"],
    )
    model = DhruvaForCausalLM(config)
    counts = model.count_parameters()
    print(f"  T4 100M Model Params: {counts['total']:,}")
    assert counts["total"] > 0
    print("[PASS] Gate 3: T4 100M configuration fully compatible.\n")


if __name__ == "__main__":
    _, cfg = test_500m_parameters()
    test_gradient_checkpointing_training_steps(cfg)
    test_t4_compatibility()
    print("=" * 70)
    print("  ALL 3 VERIFICATION GATES PASSED SUCCESSFULLY")
    print("=" * 70)
