"""Standalone Neural Network Verification Suite for Northh Dhruva (Project BHARAT-LLM).

Runs 5 strict engineering gates to verify architecture math, causal attention masking,
loss formulation, autograd health, and autoregressive memorization & generation.
Run via:
    python -m tests.test_neural_net
"""

import math
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

# Ensure current package root is in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.model_config import DhruvaConfig
from src.models.dhruva_lm import DhruvaForCausalLM


def test_gate_1_parameter_math() -> None:
    """Gate 1: Exact 100M Parameter Math Verification."""
    print("Running Gate 1: Exact 100M Parameter Math...")
    config = DhruvaConfig()
    model = DhruvaForCausalLM(config)
    report = model.parameter_report()

    total_parameters = report["total_parameters"]
    embedding_parameters = report["embedding_parameters"]
    body_parameters = report["transformer_body_parameters"]

    assert (
        embedding_parameters == 40_960_000
    ), f"Expected 40,960,000 embedding params, got {embedding_parameters:,}"
    assert (
        body_parameters == 53_160_960
    ), f"Expected 53,160,960 body params, got {body_parameters:,}"
    assert (
        total_parameters == 94_120_960
    ), f"Expected 94,120,960 total params, got {total_parameters:,}"

    print(
        f"[PASS] Gate 1 (Exact 100M Parameter Math): Total={total_parameters:,} "
        f"({embedding_parameters:,} embedding + {body_parameters:,} body)"
    )


def test_gate_2_step0_theoretical_loss() -> None:
    """Gate 2: Step-0 Theoretical Loss on tiny_debug()."""
    print("Running Gate 2: Step-0 Theoretical Loss...")
    torch.manual_seed(42)
    config = DhruvaConfig.tiny_debug()
    model = DhruvaForCausalLM(config)

    batch_size, seq_len = 4, 32
    random_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), dtype=torch.long)

    outputs = model(random_ids, labels=random_ids)
    loss = outputs["loss"]
    assert loss is not None, "Loss returned None in step-0 forward pass"

    expected_loss = math.log(config.vocab_size)  # ln(512) ≈ 6.2383
    loss_delta = abs(loss.item() - expected_loss)

    assert (
        loss_delta < 0.25
    ), f"Step-0 loss {loss.item():.4f} diverged from theoretical ln({config.vocab_size})={expected_loss:.4f} by {loss_delta:.4f} (>= 0.25)"

    print(
        f"[PASS] Gate 2 (Step-0 Theoretical Loss): loss={loss.item():.4f}, "
        f"theoretical={expected_loss:.4f}, delta={loss_delta:.4f} (< 0.25)"
    )


def test_gate_3_strict_causal_masking() -> None:
    """Gate 3: Strict Causal Masking Verification."""
    print("Running Gate 3: Strict Causal Masking...")
    torch.manual_seed(1337)
    config = DhruvaConfig.tiny_debug()
    model = DhruvaForCausalLM(config)
    model.eval()

    batch_size, seq_len = 2, 16
    seq1 = torch.randint(0, config.vocab_size, (batch_size, seq_len), dtype=torch.long)
    seq2 = seq1.clone()

    # Mutate the last token (index seq_len - 1)
    seq2[:, -1] = (seq2[:, -1] + 17) % config.vocab_size

    with torch.no_grad():
        out1 = model(seq1)
        out2 = model(seq2)

    logits1 = out1["logits"]
    logits2 = out2["logits"]

    # In a strictly causal model, positions 0 .. seq_len - 2 MUST NOT be affected by the token at seq_len - 1
    earlier_logits1 = logits1[:, :-1, :]
    earlier_logits2 = logits2[:, :-1, :]

    assert torch.allclose(
        earlier_logits1, earlier_logits2, atol=1e-5
    ), "Causal leakage detected: earlier position logits altered after modifying final token"

    # And confirm the final position logits DID change
    assert not torch.allclose(
        logits1[:, -1, :], logits2[:, -1, :], atol=1e-3
    ), "Final position logits unexpectedly identical despite changing final input token"

    print("[PASS] Gate 3 (Strict Causal Masking): Verified zero future token leakage at atol=1e-5")


def test_gate_4_full_autograd_and_gradient_health() -> None:
    """Gate 4: Full Autograd & Gradient Health Verification."""
    print("Running Gate 4: Full Autograd & Gradient Health...")
    torch.manual_seed(2024)
    config = DhruvaConfig.tiny_debug()
    model = DhruvaForCausalLM(config)
    model.train()

    batch_size, seq_len = 2, 24
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), dtype=torch.long)
    labels = input_ids.clone()

    outputs = model(input_ids, labels=labels)
    loss = outputs["loss"]
    assert loss is not None, "Loss was None during training forward pass"

    loss.backward()

    checked_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has None gradient after backward()"
            assert not torch.isnan(
                param.grad
            ).any(), f"Parameter {name} contains NaN in gradients"
            assert not torch.isinf(
                param.grad
            ).any(), f"Parameter {name} contains Inf in gradients"
            grad_sum = param.grad.abs().sum().item()
            assert (
                grad_sum > 0.0
            ), f"Parameter {name} has dead zero gradient sum after backward()"
            checked_params += 1

    assert checked_params > 0, "No trainable parameters found to check"
    print(
        f"[PASS] Gate 4 (Full Autograd & Gradient Health): All {checked_params} parameters "
        "have healthy, non-zero, finite gradients"
    )


def test_gate_5_synthetic_memorization_and_generation() -> None:
    """Gate 5: Synthetic Pattern Memorization & Autoregressive Generation."""
    print("Running Gate 5: Synthetic Pattern Memorization & Generation...")
    torch.manual_seed(999)
    config = DhruvaConfig.tiny_debug()
    model = DhruvaForCausalLM(config)
    model.train()

    # Repeating synthetic integer sequence [10, 20, 30, 40, 50, ...] of length 64
    pattern = [10, 20, 30, 40, 50]
    repeats = (64 // len(pattern)) + 2
    full_seq = (pattern * repeats)[:64]
    input_ids = torch.tensor([full_seq], dtype=torch.long)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    final_loss = 0.0
    for step in range(80):
        optimizer.zero_grad()
        out = model(input_ids, labels=input_ids)
        loss = out["loss"]
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    assert (
        final_loss < 0.05
    ), f"Pattern memorization failed to converge: final loss {final_loss:.4f} >= 0.05"

    # Generation verification: seed with [10, 20, 30], predict next tokens
    model.eval()
    seed_prompt = torch.tensor([[10, 20, 30]], dtype=torch.long)
    generated = model.generate(seed_prompt, max_new_tokens=5, sample=False)
    generated_tokens = generated[0].tolist()

    expected_continuation = [10, 20, 30, 40, 50, 10, 20, 30]
    assert (
        generated_tokens == expected_continuation
    ), f"Generated sequence {generated_tokens} does not match expected {expected_continuation}"

    print(
        f"[PASS] Gate 5 (Synthetic Pattern Memorization & Generation): Final Loss={final_loss:.4f} (< 0.05), "
        f"Continuation={generated_tokens}"
    )


def main() -> None:
    print("=" * 70)
    print("  PROJECT BHARAT-LLM: NORTHH DHRUVA NEURAL NET VERIFICATION SUITE")
    print("=" * 70)

    test_gate_1_parameter_math()
    test_gate_2_step0_theoretical_loss()
    test_gate_3_strict_causal_masking()
    test_gate_4_full_autograd_and_gradient_health()
    test_gate_5_synthetic_memorization_and_generation()

    print("=" * 70)
    print("  ALL 5 GATES PASSED: NORTHH DHRUVA ENGINE READY FOR PRE-TRAINING")
    print("=" * 70)


if __name__ == "__main__":
    main()
