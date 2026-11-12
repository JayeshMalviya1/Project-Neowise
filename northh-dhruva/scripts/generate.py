#!/usr/bin/env python3
"""Interactive Autoregressive Generation CLI for Northh Dhruva (Project BHARAT-LLM)."""

import argparse
import os
import sys
import time
from pathlib import Path

import torch

# Ensure project root is in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.model_config import DhruvaConfig
from src.evaluation.eval_harness import generate_text
from src.models.dhruva_lm import DhruvaForCausalLM
from src.tokenizer.byte_bpe import ByteBPETokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Northh Dhruva Autoregressive Generation CLI"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained PyTorch checkpoint (.pt)",
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        required=True,
        help="Path to saved ByteBPETokenizer directory",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Northh Dhruva is an independent Indian foundation model",
        help="Input text prompt for continuation",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=50,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0.0 for deterministic greedy argmax)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help="Top-k filtering threshold (0 or negative to disable)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p nucleus filtering threshold",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run generation on ('cpu', 'cuda', etc.)",
    )
    return parser.parse_args()


def load_model_from_checkpoint(
    checkpoint_path: str, device: str = "cpu"
) -> DhruvaForCausalLM:
    """Loads a DhruvaForCausalLM instance from a saved training checkpoint."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'")

    ckpt = torch.load(checkpoint_path, map_location="cpu")

    config = ckpt.get("config", None)
    if config is None:
        state_dict = ckpt.get("model_state_dict", ckpt)
        vocab_size, dim = state_dict["tok_embeddings.weight"].shape
        config = DhruvaConfig.tiny_debug()
        config.vocab_size = vocab_size
        config.dim = dim

    model = DhruvaForCausalLM(config)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("  PROJECT BHARAT-LLM: NORTHH DHRUVA GENERATION ENGINE")
    print("=" * 70)
    print(f"  Checkpoint:    {args.checkpoint}")
    print(f"  Tokenizer Dir: {args.tokenizer_dir}")
    print(f"  Device:        {args.device}")
    print(f"  Temperature:   {args.temperature}")
    print(f"  Top-k / Top-p: {args.top_k} / {args.top_p}")
    print(f"  Max Tokens:    {args.max_tokens}")
    print("-" * 70)

    # 1. Load Tokenizer
    tokenizer = ByteBPETokenizer()
    tokenizer.load(args.tokenizer_dir)
    print(f"[Generate] Tokenizer loaded (vocab_size={tokenizer.vocab_size})")

    # 2. Load Model
    model = load_model_from_checkpoint(args.checkpoint, device=args.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Generate] Model loaded ({total_params:,} parameters)")

    # 3. Generate
    print(f"\n[Prompt]:\n{args.prompt}")
    print("-" * 70)

    prompt_tokens = tokenizer.encode(args.prompt)
    t0 = time.perf_counter()

    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=args.device,
    )

    t1 = time.perf_counter()
    latency_sec = t1 - t0
    output_tokens = tokenizer.encode(generated_text)
    num_new_tokens = max(0, len(output_tokens) - len(prompt_tokens))
    tokens_per_sec = (num_new_tokens / latency_sec) if latency_sec > 0 else 0.0

    print(f"[Generated Completion]:\n{generated_text}")
    print("-" * 70)
    print(f"  Latency:        {latency_sec * 1000.0:.2f} ms")
    print(f"  Tokens Gen:     {num_new_tokens} tokens")
    print(f"  Throughput:     {tokens_per_sec:.2f} tokens/sec")
    print("=" * 70)


if __name__ == "__main__":
    main()

