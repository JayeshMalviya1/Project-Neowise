#!/usr/bin/env python3
"""Pretraining Execution Runner for Northh Dhruva (Project BHARAT-LLM).

Usage:
    python -m src.training.pretrain --config configs/pretrain_100m.yaml
    python -m src.training.pretrain --max_steps 50 --micro_batch_size 4
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from torch.utils.data import DataLoader

# Ensure project root is in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.model_config import DhruvaConfig
from src.models.dhruva_lm import DhruvaForCausalLM
from src.tokenizer.byte_bpe import ByteBPETokenizer
from src.data.dataset import build_binary_shard, MemmapPretrainingDataset
from src.training.trainer import DhruvaTrainer

DEFAULT_SAMPLE_CORPUS = """
Northh Dhruva is an independent Indian foundation model engineered from scratch for deep reasoning.
Artificial intelligence and transformer architectures empower foundational computational systems.
Deep neural networks utilize self-attention mechanisms, rotary position embeddings, and grouped-query attention.
Pretraining at scale requires disciplined data pipelining, memory-mapped binary shards, and gradient health.
""" * 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Northh Dhruva Pretraining Runner")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pretrain_100m.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to memory-mapped binary token shard (.bin)",
    )
    parser.add_argument(
        "--text_path",
        type=str,
        default=None,
        help="Path to raw text file to tokenize if binary shard is not ready",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory to save/load model checkpoints",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Override total optimization steps",
    )
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=None,
        help="Override per-step micro batch size",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=None,
        help="Override sequence context length",
    )
    parser.add_argument(
        "--max_lr",
        type=float,
        default=None,
        help="Override maximum learning rate",
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=None,
        help="Override gradient accumulation steps",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Do not resume from existing checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Explicit compute device ('cuda', 'mps', 'cpu')",
    )
    return parser.parse_args()


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration dictionary from YAML file if exists."""
    if not os.path.isfile(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}


def prepare_dataset_and_loader(
    data_path: str,
    text_path: Optional[str],
    seq_len: int,
    batch_size: int,
    tokenizer_dir: Optional[str] = None,
) -> tuple[DataLoader, ByteBPETokenizer]:
    """Prepares tokenizer and memory-mapped DataLoader."""
    tokenizer = ByteBPETokenizer()

    # If binary shard does not exist, build it from text_path or sample corpus
    if not os.path.isfile(data_path):
        print(f"[Pretrain] Binary shard '{data_path}' not found.")
        if text_path and os.path.isfile(text_path):
            print(f"[Pretrain] Tokenizing raw corpus from '{text_path}'...")
            with open(text_path, "r", encoding="utf-8") as f:
                corpus_text = f.read()
        else:
            print("[Pretrain] Generating synthetic pretraining corpus shard...")
            corpus_text = DEFAULT_SAMPLE_CORPUS

        # Train tokenizer on corpus if target directory doesn't exist
        if tokenizer_dir and os.path.exists(os.path.join(tokenizer_dir, "vocab.json")):
            tokenizer.load(tokenizer_dir)
        else:
            tokenizer.train(corpus_text, target_vocab_size=380)
            if tokenizer_dir:
                tokenizer.save(tokenizer_dir)

        token_ids = tokenizer.encode(corpus_text)
        # Ensure enough tokens for multiple chunks
        min_tokens = seq_len * batch_size * 4
        if len(token_ids) < min_tokens:
            multiplier = (min_tokens // len(token_ids)) + 2
            token_ids = token_ids * multiplier

        meta = build_binary_shard(token_ids, data_path)
        print(f"[Pretrain] Created binary shard at '{data_path}' ({meta['num_tokens']} tokens).")
    else:
        if tokenizer_dir and os.path.exists(os.path.join(tokenizer_dir, "vocab.json")):
            tokenizer.load(tokenizer_dir)

    dataset = MemmapPretrainingDataset(bin_path=data_path, seq_len=seq_len)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=(torch.cuda.is_available()),
    )
    return loader, tokenizer


def run_pretrain() -> None:
    args = parse_args()
    yaml_config = load_yaml_config(args.config)

    model_cfg = yaml_config.get("model", {})
    train_cfg = yaml_config.get("training", {})
    data_cfg = yaml_config.get("data", {})

    # Merge configuration with CLI overrides
    data_path = args.data_path or data_cfg.get("train_bin_path", "./data/train_shard.bin")
    checkpoint_dir = args.checkpoint_dir or train_cfg.get("checkpoint_dir", "./checkpoints/dhruva_100m")
    seq_len = args.seq_len or train_cfg.get("seq_len", 256)
    micro_batch_size = args.micro_batch_size or train_cfg.get("micro_batch_size", 8)
    max_steps = args.max_steps or train_cfg.get("max_steps", 500)
    max_lr = args.max_lr or float(train_cfg.get("max_lr", 6e-4))
    min_lr = float(train_cfg.get("min_lr", 6e-5))
    warmup_steps = train_cfg.get("warmup_steps", 50)
    grad_accum_steps = args.grad_accum_steps or train_cfg.get("grad_accum_steps", 4)
    weight_decay = float(train_cfg.get("weight_decay", 0.1))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    save_interval = train_cfg.get("save_interval", 100)
    log_interval = train_cfg.get("log_interval", 10)
    resume = not args.no_resume

    print("=" * 70)
    print("  PROJECT BHARAT-LLM: NORTHH DHRUVA PRETRAINING RUNNER")
    print("=" * 70)
    print(f"Config path:      {args.config}")
    print(f"Data shard:       {data_path}")
    print(f"Checkpoint dir:   {checkpoint_dir}")
    print(f"Max steps:        {max_steps}")
    print(f"Batch size:       {micro_batch_size} (accum: {grad_accum_steps})")
    print(f"Context seq_len:  {seq_len}")

    # Prepare Data and Tokenizer
    loader, tokenizer = prepare_dataset_and_loader(
        data_path=data_path,
        text_path=args.text_path,
        seq_len=seq_len,
        batch_size=micro_batch_size,
        tokenizer_dir=data_cfg.get("tokenizer_dir", None),
    )

    # Instantiate DhruvaConfig
    vocab_size = model_cfg.get("vocab_size", tokenizer.vocab_size)
    dhruva_config = DhruvaConfig(
        vocab_size=vocab_size,
        dim=model_cfg.get("dim", 640),
        n_layers=model_cfg.get("n_layers", 12),
        n_heads=model_cfg.get("n_heads", 10),
        n_kv_heads=model_cfg.get("n_kv_heads", 2),
        intermediate_size=model_cfg.get("intermediate_size", 1792),
        max_seq_len=seq_len,
        rope_theta=model_cfg.get("rope_theta", 500000.0),
        rms_norm_eps=model_cfg.get("rms_norm_eps", 1e-5),
        tie_word_embeddings=model_cfg.get("tie_word_embeddings", True),
        initializer_range=model_cfg.get("initializer_range", 0.02),
    )

    print("\n[Pretrain] Initializing DhruvaForCausalLM...")
    model = DhruvaForCausalLM(dhruva_config)

    # Task 2.4: Wrap model with torch.compile if CUDA is available and PyTorch 2.x
    if torch.cuda.is_available() and hasattr(torch, "compile"):
        try:
            print("[Pretrain] CUDA available: compiling model with torch.compile...")
            model = torch.compile(model)
        except Exception as e:
            print(f"[Pretrain] torch.compile skipped: {e}")

    # Task 2.5: Setup DhruvaTrainer
    trainer = DhruvaTrainer(
        model=model,
        train_loader=loader,
        device=args.device,
        max_steps=max_steps,
        grad_accum_steps=grad_accum_steps,
        max_lr=max_lr,
        min_lr=min_lr,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        checkpoint_dir=checkpoint_dir,
        save_interval=save_interval,
        log_interval=log_interval,
    )

    # Task 2.6: Handle keyboard interrupts gracefully for emergency checkpointing
    try:
        trainer.fit(resume=resume)
    except KeyboardInterrupt:
        print("\n[Emergency] Caught interrupt in runner! Ensuring checkpoint saved...")
        emergency_path = os.path.join(checkpoint_dir, "emergency_checkpoint.pt")
        trainer.save_checkpoint(path=emergency_path)
        print(f"[Emergency] Emergency checkpoint saved at {emergency_path}")
        sys.exit(0)


if __name__ == "__main__":
    run_pretrain()
