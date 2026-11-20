#!/usr/bin/env python3
"""Interactive Autoregressive Generation CLI for Northh Dhruva (Project BHARAT-LLM)."""

import argparse
import os
import sys
import time
from pathlib import Path

import gc
import torch

# Cap CPU threads to 4 P-cores on laptops to prevent AVX2 thermal/power trip reboots
_default_threads = int(os.environ.get("DHRUVA_NUM_THREADS", "4"))
if torch.get_num_threads() > _default_threads:
    torch.set_num_threads(_default_threads)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")

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
        default="data/tokenizer_32k",
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

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location="cpu")

    config = ckpt.get("config", None) if isinstance(ckpt, dict) else None
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if config is None:
        vocab_size, dim = state_dict["tok_embeddings.weight"].shape
        if dim == 1280:
            config = DhruvaConfig(
                vocab_size=vocab_size,
                dim=1280,
                n_layers=24,
                n_heads=16,
                n_kv_heads=4,
                intermediate_size=3584,
                max_seq_len=2048,
            )
        else:
            config = DhruvaConfig.tiny_debug()
            config.vocab_size = vocab_size
            config.dim = dim

    model = DhruvaForCausalLM(config, init_weights=False)
    cleaned_sd = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned_sd, strict=False)
    del cleaned_sd, state_dict, ckpt
    gc.collect()
    if device == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        model.to(device=device, dtype=torch.bfloat16)
    else:
        model.to(device=device, dtype=torch.float32)
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
    tok_dir = args.tokenizer_dir if os.path.isdir(args.tokenizer_dir) else os.path.join(PROJECT_ROOT, args.tokenizer_dir)
    tokenizer = ByteBPETokenizer()
    tokenizer.load(tok_dir)
    print(f"[Generate] Tokenizer loaded (vocab_size={tokenizer.vocab_size})")

    # 2. Load Model
    model = load_model_from_checkpoint(args.checkpoint, device=args.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Generate] Model loaded ({total_params:,} parameters)")

    # 3. Generate
    user_input = args.prompt
    print(f"\n[Prompt]:\n{user_input}")
    print("-" * 70)

    t0 = time.perf_counter()

    # 1. User input ko instruction format mein wrap karo
    formatted_prompt = f"### User:\n{user_input}\n\n### Assistant:\n"

    # 2. Encode karo (BOS token sabse aage laga kar)
    input_ids = [tokenizer.bos_token_id] + tokenizer.encode(formatted_prompt)
    x = torch.tensor([input_ids], dtype=torch.long, device=args.device)

    # 3. Generation loop
    with torch.no_grad():
        for _ in range(args.max_tokens):
            logits = model(x)["logits"][:, -1, :]

            if args.temperature <= 0.0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits / args.temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            if next_id.item() == tokenizer.eos_token_id:
                break

            x = torch.cat([x, next_id], dim=-1)

    # 4. Decode karte time prompt ko hata do, sirf answer print karo
    generated_ids = x[0, len(input_ids):].tolist()
    final_output = tokenizer.decode(generated_ids).strip()

    t1 = time.perf_counter()
    latency_sec = t1 - t0
    num_new_tokens = len(generated_ids)
    tokens_per_sec = (num_new_tokens / latency_sec) if latency_sec > 0 else 0.0

    print(f"\n[Dhruva]: {final_output}")
    print("-" * 70)
    print(f"  Latency:        {latency_sec * 1000.0:.2f} ms")
    print(f"  Tokens Gen:     {num_new_tokens} tokens")
    print(f"  Throughput:     {tokens_per_sec:.2f} tokens/sec")
    print("=" * 70)


if __name__ == "__main__":
    main()

