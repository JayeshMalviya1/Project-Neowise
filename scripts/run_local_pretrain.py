#!/usr/bin/env python3
"""Local 24GB GPU Pretraining Runner for Northh Dhruva (Project BHARAT-LLM).

Loads `configs/train_local_24gb.yaml`, attaches `MemmapPretrainingDataset` across all
binary shards in `data/shards/train/`, instantiates `DhruvaForCausalLM` (wrapping with
`torch.compile` when CUDA is available), and runs `DhruvaTrainer.fit(resume=True)`
with live logging of loss, tokens/sec, VRAM utilization, and validation perplexity.

Usage:
    python scripts/run_local_pretrain.py
    python scripts/run_local_pretrain.py --config configs/train_local_24gb.yaml
"""

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import yaml
from torch.utils.data import ConcatDataset, DataLoader, Dataset

# Ensure project root is in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.model_config import DhruvaConfig
from src.data.dataset import MemmapPretrainingDataset
from src.evaluation.eval_harness import compute_validation_perplexity
from src.models.dhruva_lm import DhruvaForCausalLM
from src.training.trainer import DhruvaTrainer


class MultiShardMemmapDataset(Dataset):
    """Wraps multiple `MemmapPretrainingDataset` shards while exposing `seq_len`."""

    def __init__(self, shards_dir: str, seq_len: int) -> None:
        self.shards_dir = shards_dir
        self.seq_len = seq_len

        bin_files = sorted(glob.glob(os.path.join(shards_dir, "*.bin")))
        if not bin_files:
            raise FileNotFoundError(
                f"No binary shard files (*.bin) found in '{shards_dir}'. "
                "Run `scripts/prepare_data.py` first."
            )

        self.shard_datasets: List[MemmapPretrainingDataset] = []
        for bf in bin_files:
            ds = MemmapPretrainingDataset(bin_path=bf, seq_len=seq_len)
            if len(ds) > 0:
                self.shard_datasets.append(ds)

        if not self.shard_datasets:
            raise ValueError(
                f"Shards in '{shards_dir}' do not contain enough tokens for seq_len={seq_len}."
            )

        self._concat = ConcatDataset(self.shard_datasets)

    def __len__(self) -> int:
        return len(self._concat)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._concat[idx]


class Local24GBTrainer(DhruvaTrainer):
    """Extends DhruvaTrainer with live VRAM utilization & validation perplexity logging."""

    def __init__(
        self,
        *args: Any,
        val_dataset: Optional[Dataset] = None,
        eval_max_batches: int = 20,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.val_dataset = val_dataset
        self.eval_max_batches = eval_max_batches
        self._last_eval_logged_step: int = -1

    def log_vram_and_perplexity(self, step: Optional[int] = None) -> None:
        """Logs current/peak CUDA VRAM utilization and held-out validation perplexity."""
        cur_step = self.current_step if step is None else step
        if cur_step == self._last_eval_logged_step:
            return
        self._last_eval_logged_step = cur_step
        vram_str = "Memory: N/A (CPU)"
        if self.device.type == "cuda":
            alloc_gb = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
            peak_gb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
            vram_str = f"VRAM: {alloc_gb:.2f}GB (Peak: {peak_gb:.2f}GB)"
        elif (
            self.device.type == "mps"
            and hasattr(torch, "mps")
            and hasattr(torch.mps, "current_allocated_memory")
        ):
            mps_mb = torch.mps.current_allocated_memory() / (1024 ** 2)
            vram_str = f"MPS Mem: {mps_mb:.1f} MB"

        ppl_str = "ValPPL: N/A"
        if self.val_dataset is not None and len(self.val_dataset) > 0:
            was_training = self.model.training
            ppl = compute_validation_perplexity(
                model=self._get_base_model(),
                val_dataset=self.val_dataset,
                device=self.device,
                max_batches=self.eval_max_batches,
            )
            if was_training:
                self.model.train()
            ppl_str = f"ValPPL: {ppl:.2f}"

        print(f"[Metrics @ Step {cur_step}] {ppl_str} | {vram_str}")

    def save_checkpoint(self, path: Optional[str] = None, step: Optional[int] = None) -> str:
        save_path = super().save_checkpoint(path=path, step=step)
        if path is None:
            self.log_vram_and_perplexity(step=step)
        return save_path


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"YAML configuration not found at '{config_path}'")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local 24GB GPU Pretraining Runner")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_local_24gb.yaml",
        help="Path to YAML training config",
    )
    parser.add_argument(
        "--train_shards_dir",
        type=str,
        default=None,
        help="Override directory containing training .bin shards",
    )
    parser.add_argument(
        "--val_shards_dir",
        type=str,
        default=None,
        help="Override directory containing validation .bin shards",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Override checkpoint output directory",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Override maximum optimization steps",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Explicit compute device ('cuda', 'mps', 'cpu')",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Start fresh instead of resuming from checkpoint",
    )
    return parser.parse_args()


def run_local_pretrain() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})

    train_shards_dir = args.train_shards_dir or data_cfg.get("train_shards_dir", "./data/shards/train")
    val_shards_dir = args.val_shards_dir or data_cfg.get("val_shards_dir", "./data/shards/val")
    checkpoint_dir = args.checkpoint_dir or train_cfg.get("checkpoint_dir", "./checkpoints/dhruva_local_16k")

    seq_len = int(train_cfg.get("seq_len", 1024))
    micro_batch_size = int(train_cfg.get("micro_batch_size", 8))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 8))
    max_steps = int(args.max_steps or train_cfg.get("max_steps", 2000))
    max_lr = float(train_cfg.get("max_lr", 6.0e-4))
    min_lr = float(train_cfg.get("min_lr", 6.0e-5))
    warmup_steps = int(train_cfg.get("warmup_steps", 100))
    weight_decay = float(train_cfg.get("weight_decay", 0.1))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    save_interval = int(train_cfg.get("save_interval", 200))
    log_interval = int(train_cfg.get("log_interval", 10))

    effective_batch_tokens = micro_batch_size * grad_accum_steps * seq_len

    print("=" * 70)
    print("  PROJECT BHARAT-LLM: NORTHH DHRUVA LOCAL 24GB PRETRAINING")
    print("=" * 70)
    print(f"  Config:              {args.config}")
    print(f"  Train Shards Dir:    {train_shards_dir}")
    print(f"  Val Shards Dir:      {val_shards_dir}")
    print(f"  Checkpoint Dir:      {checkpoint_dir}")
    print(f"  Effective Batch:     {effective_batch_tokens:,} tokens/step ({micro_batch_size}x{grad_accum_steps}x{seq_len})")
    print(f"  Target Dtype:        {train_cfg.get('dtype', 'bfloat16')}")
    print(f"  Max Steps:           {max_steps:,}")

    # 1. Attach MemmapPretrainingDataset to training shards
    train_dataset = MultiShardMemmapDataset(shards_dir=train_shards_dir, seq_len=seq_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=micro_batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"  Train Chunks:        {len(train_dataset):,} across {len(train_dataset.shard_datasets)} shard(s)")

    val_dataset: Optional[Dataset] = None
    if val_shards_dir and os.path.isdir(val_shards_dir) and glob.glob(os.path.join(val_shards_dir, "*.bin")):
        try:
            val_dataset = MultiShardMemmapDataset(shards_dir=val_shards_dir, seq_len=seq_len)
            print(f"  Val Chunks:          {len(val_dataset):,} across {len(val_dataset.shard_datasets)} shard(s)")
        except ValueError as exc:
            print(f"[Pretrain] Warning: Validation shards skipped ({exc})")

    # 2. Instantiate DhruvaForCausalLM
    dhruva_config = DhruvaConfig(
        vocab_size=int(model_cfg.get("vocab_size", 16384)),
        dim=int(model_cfg.get("dim", 640)),
        n_layers=int(model_cfg.get("n_layers", 12)),
        n_heads=int(model_cfg.get("n_heads", 10)),
        n_kv_heads=int(model_cfg.get("n_kv_heads", 2)),
        intermediate_size=int(model_cfg.get("intermediate_size", 1792)),
        max_seq_len=int(model_cfg.get("max_seq_len", seq_len)),
        rope_theta=float(model_cfg.get("rope_theta", 500000.0)),
        rms_norm_eps=float(model_cfg.get("rms_norm_eps", 1e-5)),
        tie_word_embeddings=bool(model_cfg.get("tie_word_embeddings", True)),
        initializer_range=float(model_cfg.get("initializer_range", 0.02)),
    )
    model = DhruvaForCausalLM(dhruva_config)

    # 3. Resolve target device and strictly skip torch.compile on MPS/CPU
    if args.device:
        target_device = torch.device(args.device)
    elif torch.cuda.is_available():
        target_device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        target_device = torch.device("mps")
    else:
        target_device = torch.device("cpu")

    if target_device.type == "cuda" and hasattr(torch, "compile"):
        try:
            print("[Pretrain] CUDA detected: compiling model with torch.compile...")
            model = torch.compile(model)
        except Exception as exc:
            print(f"[Pretrain] torch.compile skipped: {exc}")
    elif target_device.type == "mps":
        print("[Pretrain] Apple Silicon MPS detected: skipping torch.compile() for backend stability.")

    # 4. Run DhruvaTrainer.fit(resume=True) with live metrics
    trainer = Local24GBTrainer(
        model=model,
        train_loader=train_loader,
        val_dataset=val_dataset,
        device=target_device,
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

    trainer.log_vram_and_perplexity(step=0)
    try:
        trainer.fit(resume=not args.no_resume)
    except KeyboardInterrupt:
        print("\n[Emergency] Interrupt caught. Saving emergency checkpoint...")
        emergency_path = os.path.join(checkpoint_dir, "emergency_checkpoint.pt")
        trainer.save_checkpoint(path=emergency_path)
        print(f"[Emergency] Saved to '{emergency_path}'")
        sys.exit(0)


if __name__ == "__main__":
    run_local_pretrain()