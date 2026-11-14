#!/usr/bin/env python3
"""Interactive Testing & Inference CLI for Northh Dhruva (Project BHARAT-LLM).

Supports:
- Interactive REPL with real-time token streaming.
- Auto-detection of latest checkpoint in `checkpoints_100m/` and `data/tokenizer_16k`.
- Live runtime controls (/temp, /topk, /topp, /tokens, /ckpt, /device, /info).
- Single-prompt CLI execution with latency & throughput benchmarking.
- Checkpoint comparison mode (--compare) across training steps.
"""

import argparse
import glob
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Auto re-exec into local .venv if torch is not in current environment
try:
    import torch
except ImportError:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    venv_python = os.path.join(project_root, ".venv", "bin", "python")
    if os.path.isfile(venv_python) and sys.executable != venv_python:
        os.execv(venv_python, [venv_python] + sys.argv)
    raise

# Ensure project root is in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.model_config import DhruvaConfig
from src.evaluation.eval_harness import generate_text, stream_generate_tokens
from src.models.dhruva_lm import DhruvaForCausalLM
from src.tokenizer.byte_bpe import ByteBPETokenizer


def get_default_device() -> str:
    """Selects optimal available accelerator (mps for Apple Silicon, cuda, else cpu)."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def find_available_checkpoints(search_dirs: Optional[List[str]] = None) -> List[Tuple[int, str]]:
    """Discovers all .pt checkpoints sorted by step number."""
    if search_dirs is None:
        search_dirs = ["checkpoints_100m", "checkpoints/dhruva_t4_100m", "checkpoints"]

    found: List[Tuple[int, str]] = []
    for sdir in search_dirs:
        full_dir = os.path.join(PROJECT_ROOT, sdir) if not os.path.isabs(sdir) else sdir
        if not os.path.isdir(full_dir):
            continue
        for pt_file in glob.glob(os.path.join(full_dir, "*.pt")):
            fname = os.path.basename(pt_file)
            match = re.search(r"step_(\d+)", fname)
            step = int(match.group(1)) if match else 0
            found.append((step, pt_file))

    found.sort(key=lambda x: (x[0], x[1]))
    return found


def find_latest_checkpoint() -> Optional[str]:
    """Returns the path to the checkpoint with the highest step number."""
    ckpts = find_available_checkpoints()
    return ckpts[-1][1] if ckpts else None


def load_model(checkpoint_path: str, device: str = "cpu") -> Tuple[DhruvaForCausalLM, Dict]:
    """Loads Dhruva model and checkpoint metadata."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = ckpt.get("config", None)
    state_dict = ckpt.get("model_state_dict", ckpt)

    if config is None:
        vocab_size, dim = state_dict["tok_embeddings.weight"].shape
        config = DhruvaConfig.tiny_debug()
        config.vocab_size = vocab_size
        config.dim = dim

    model = DhruvaForCausalLM(config)
    # Strip potential compilation prefixes if present
    cleaned_sd = {}
    for k, v in state_dict.items():
        clean_k = k.replace("_orig_mod.", "")
        cleaned_sd[clean_k] = v

    model.load_state_dict(cleaned_sd, strict=False)
    model.to(device)
    model.eval()

    meta = {
        "step": ckpt.get("step", "N/A"),
        "last_loss": ckpt.get("last_loss", None),
        "config": config,
        "path": checkpoint_path,
        "params": sum(p.numel() for p in model.parameters()),
    }
    return model, meta


def print_banner(meta: Dict, device: str, tokenizer_vocab: int) -> None:
    print("\n" + "=" * 72)
    print("  🚀 NORTHH DHRUVA (PROJECT BHARAT-LLM) — INTERACTIVE TESTING CLI")
    print("=" * 72)
    print(f"  Checkpoint:   {os.path.basename(meta['path'])} (Step {meta['step']})")
    if meta["last_loss"] is not None:
        print(f"  Training Loss:{meta['last_loss']:.4f}")
    print(f"  Parameters:   {meta['params']:,}")
    print(f"  Architecture: {meta['config'].n_layers}L | {meta['config'].dim}D | {meta['config'].n_heads}H / {meta['config'].n_kv_heads}KV (GQA)")
    print(f"  Tokenizer:    16K Byte-Level BPE (Vocab: {tokenizer_vocab:,})")
    print(f"  Device:       {device.upper()}")
    print("=" * 72)
    print("  Type your prompt and press Enter to generate.")
    print("  Commands: /help, /temp <val>, /tokens <val>, /ckpt <step>, /compare, exit")
    print("=" * 72 + "\n")


def print_help() -> None:
    print("\nAvailable CLI Commands:")
    print("  /help              Show this help message")
    print("  /info              Show model architecture, parameters, and active settings")
    print("  /temp <float>      Set sampling temperature (e.g. /temp 0.7, 0.0 for greedy)")
    print("  /tokens <int>      Set maximum new tokens to generate (e.g. /tokens 120)")
    print("  /topk <int>        Set top-k filtering threshold (e.g. /topk 40, 0 to disable)")
    print("  /topp <float>      Set top-p nucleus sampling threshold (e.g. /topp 0.9)")
    print("  /device <name>     Switch device live ('cpu', 'mps', 'cuda')")
    print("  /ckpt              List all discovered checkpoints")
    print("  /ckpt <step|path>  Switch active checkpoint (e.g. /ckpt 1526 or /ckpt 600)")
    print("  /compare <prompt>  Compare generation across all training checkpoints")
    print("  /clear             Clear terminal screen")
    print("  exit, quit, q      Quit the interactive CLI\n")


def run_interactive_repl(
    model: DhruvaForCausalLM,
    meta: Dict,
    tokenizer: ByteBPETokenizer,
    device: str = "cpu",
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    max_tokens: int = 80,
) -> None:
    """Runs interactive read-eval-print loop with streaming tokens."""
    current_device = device
    current_model = model
    current_meta = meta
    current_temp = temperature
    current_top_k = top_k
    current_top_p = top_p
    current_max_tokens = max_tokens

    print_banner(current_meta, current_device, tokenizer.vocab_size)

    while True:
        try:
            prompt = input(f"\033[1;36m[dhruva step-{current_meta['step']} T={current_temp}]\033[0m >>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Dhruva session ended.")
            break

        if not prompt:
            continue

        cmd = prompt.lower()
        if cmd in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        if cmd == "/help":
            print_help()
            continue

        if cmd == "/clear":
            os.system("clear")
            print_banner(current_meta, current_device, tokenizer.vocab_size)
            continue

        if cmd == "/info":
            print("\n--- Current Session Configuration ---")
            print(f"Checkpoint:   {current_meta['path']}")
            print(f"Training Step:{current_meta['step']}")
            print(f"Parameters:   {current_meta['params']:,}")
            print(f"Device:       {current_device}")
            print(f"Temperature:  {current_temp}")
            print(f"Top-K / Top-P:{current_top_k} / {current_top_p}")
            print(f"Max Tokens:   {current_max_tokens}\n")
            continue

        if cmd.startswith("/temp"):
            parts = prompt.split()
            if len(parts) > 1:
                try:
                    current_temp = float(parts[1])
                    print(f"Temperature updated to: {current_temp}")
                except ValueError:
                    print("Usage: /temp <float> (e.g. /temp 0.7)")
            else:
                print(f"Current temperature: {current_temp}")
            continue

        if cmd.startswith("/tokens") or cmd.startswith("/max_tokens"):
            parts = prompt.split()
            if len(parts) > 1:
                try:
                    current_max_tokens = int(parts[1])
                    print(f"Max tokens updated to: {current_max_tokens}")
                except ValueError:
                    print("Usage: /tokens <int> (e.g. /tokens 100)")
            else:
                print(f"Current max tokens: {current_max_tokens}")
            continue

        if cmd.startswith("/topk"):
            parts = prompt.split()
            if len(parts) > 1:
                try:
                    current_top_k = int(parts[1])
                    print(f"Top-k updated to: {current_top_k}")
                except ValueError:
                    print("Usage: /topk <int> (e.g. /topk 40)")
            continue

        if cmd.startswith("/topp"):
            parts = prompt.split()
            if len(parts) > 1:
                try:
                    current_top_p = float(parts[1])
                    print(f"Top-p updated to: {current_top_p}")
                except ValueError:
                    print("Usage: /topp <float> (e.g. /topp 0.9)")
            continue

        if cmd.startswith("/device"):
            parts = prompt.split()
            if len(parts) > 1:
                new_dev = parts[1].lower()
                try:
                    current_model.to(new_dev)
                    current_device = new_dev
                    print(f"Moved model to device: {current_device.upper()}")
                except Exception as exc:
                    print(f"Failed to switch device to '{new_dev}': {exc}")
            else:
                print(f"Current device: {current_device}")
            continue

        if cmd.startswith("/ckpt"):
            parts = prompt.split()
            ckpts = find_available_checkpoints()
            if len(parts) == 1:
                print("\nDiscovered Checkpoints:")
                for step, path in ckpts:
                    marker = " (ACTIVE)" if path == current_meta["path"] else ""
                    print(f"  Step {step:5d} -> {path}{marker}")
                print("\nTo switch: /ckpt <step> (e.g. /ckpt 600)\n")
                continue

            target = parts[1]
            chosen_path = None
            if target.isdigit():
                t_step = int(target)
                for step, path in ckpts:
                    if step == t_step:
                        chosen_path = path
                        break
            elif os.path.isfile(target):
                chosen_path = target

            if chosen_path:
                print(f"Loading checkpoint '{chosen_path}'...")
                try:
                    del current_model
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    current_model, current_meta = load_model(chosen_path, device=current_device)
                    print(f"Loaded Step {current_meta['step']} successfully.")
                except Exception as exc:
                    print(f"Error loading checkpoint: {exc}")
            else:
                print(f"Could not find checkpoint for '{target}'. Run '/ckpt' to list available.")
            continue

        if cmd.startswith("/compare"):
            parts = prompt.split(maxsplit=1)
            cmp_prompt = parts[1] if len(parts) > 1 else "Northh Dhruva is"
            run_comparison_benchmark(cmp_prompt, tokenizer, current_device, current_temp, current_top_k, current_top_p, 40)
            continue

        # Normal text generation with real-time streaming
        print("\n\033[1;32m[Dhruva]:\033[0m ", end="", flush=True)
        t0 = time.perf_counter()
        token_count = 0

        try:
            for chunk in stream_generate_tokens(
                model=current_model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=current_max_tokens,
                temperature=current_temp,
                top_k=current_top_k,
                top_p=current_top_p,
                device=current_device,
            ):
                print(chunk, end="", flush=True)
                token_count += 1
        except Exception as exc:
            print(f"\n[Generation error]: {exc}")

        t1 = time.perf_counter()
        elapsed = t1 - t0
        tok_per_sec = (token_count / elapsed) if elapsed > 0 else 0.0
        print(f"\n\033[2m({token_count} tokens | {elapsed * 1000.0:.1f}ms | {tok_per_sec:.1f} tok/s)\033[0m\n")


def run_comparison_benchmark(
    prompt: str,
    tokenizer: ByteBPETokenizer,
    device: str,
    temperature: float,
    top_k: int,
    top_p: float,
    max_tokens: int,
) -> None:
    """Evaluates the same prompt across all available checkpoint steps."""
    ckpts = find_available_checkpoints()
    if not ckpts:
        print("No checkpoints found to compare.")
        return

    print("\n" + "=" * 72)
    print(f"  📊 CHECKPOINT EVOLUTION COMPARISON")
    print(f"  Prompt: '{prompt}'")
    print("=" * 72)

    for step, ckpt_path in ckpts:
        try:
            m, meta = load_model(ckpt_path, device=device)
            loss_str = f"loss={meta['last_loss']:.3f}" if meta['last_loss'] is not None else ""
            print(f"\n\033[1;33m▶ [Step {step:5d}] ({loss_str})\033[0m:")
            out = generate_text(
                model=m,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                device=device,
            )
            # Print only generated suffix
            completion = out[len(prompt):] if out.startswith(prompt) else out
            print(f"  {completion.strip()}")
            del m
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception as exc:
            print(f"  Error at step {step}: {exc}")

    print("\n" + "=" * 72 + "\n")


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Northh Dhruva Model Testing & Interactive CLI"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to .pt checkpoint file (defaults to latest in checkpoints_100m/)",
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default="data/tokenizer_16k",
        help="Path to tokenizer directory containing vocab.json & merges.json",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt execution mode (skips interactive REPL)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run prompt comparison across all available checkpoints",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Target device ('cpu', 'mps', 'cuda'). Defaults to best available.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0.0 for deterministic greedy)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help="Top-k filtering threshold",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling threshold",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=80,
        help="Maximum new tokens to generate",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_cli_args()

    # 1. Resolve Device
    device = args.device or get_default_device()

    # 2. Resolve Checkpoint
    ckpt_path = args.checkpoint or find_latest_checkpoint()
    if not ckpt_path or not os.path.isfile(ckpt_path):
        print("Error: No valid checkpoint found. Please specify --checkpoint <path>.")
        sys.exit(1)

    # 3. Resolve Tokenizer
    tok_dir = os.path.join(PROJECT_ROOT, args.tokenizer_dir) if not os.path.isabs(args.tokenizer_dir) else args.tokenizer_dir
    if not os.path.isdir(tok_dir) or not os.path.isfile(os.path.join(tok_dir, "vocab.json")):
        print(f"Error: Tokenizer directory not found at '{tok_dir}'.")
        sys.exit(1)

    tokenizer = ByteBPETokenizer()
    tokenizer.load(tok_dir)

    # 4. Mode: Checkpoint Comparison
    if args.compare:
        test_prompt = args.prompt or "Northh Dhruva is an independent foundation model"
        run_comparison_benchmark(
            prompt=test_prompt,
            tokenizer=tokenizer,
            device=device,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
        return

    # 5. Load Active Model
    model, meta = load_model(ckpt_path, device=device)

    # 6. Mode: Single Prompt Execution
    if args.prompt is not None:
        print_banner(meta, device, tokenizer.vocab_size)
        print(f"[Prompt]: {args.prompt}\n")
        print("[Generated Completion]:\n", end="")
        t0 = time.perf_counter()
        for chunk in stream_generate_tokens(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
        ):
            print(chunk, end="", flush=True)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        print(f"\n\nLatency: {elapsed * 1000.0:.2f} ms | Device: {device.upper()}\n")
        return

    # 7. Mode: Interactive REPL
    run_interactive_repl(
        model=model,
        meta=meta,
        tokenizer=tokenizer,
        device=device,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
