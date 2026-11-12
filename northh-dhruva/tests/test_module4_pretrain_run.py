"""Module 4 Verification Suite: Pretraining Runner & Training Optimization Pipeline.

Project BHARAT-LLM: Northh Dhruva Module 4.
Verifies that:
1. A synthetic corpus can be generated and written to disk.
2. ByteBPETokenizer trains on the corpus and tokenizes it.
3. Binary uint16 shard is built and verified.
4. MemmapPretrainingDataset and DataLoader are set up.
5. DhruvaTrainer trains DhruvaForCausalLM for 25 optimization steps.
6. Training loss decreases monotonically across the 25 steps without errors.
7. Checkpoint is created and verified on disk.

Run via:
    python -m tests.test_module4_pretrain_run
"""

import os
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Ensure project root is in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.model_config import DhruvaConfig
from src.models.dhruva_lm import DhruvaForCausalLM
from src.tokenizer.byte_bpe import ByteBPETokenizer
from src.data.dataset import build_binary_shard, MemmapPretrainingDataset
from src.training.trainer import DhruvaTrainer


def test_module4_pretrain_run() -> None:
    print("=" * 70)
    print("  PROJECT BHARAT-LLM: NORTHH DHRUVA MODULE 4 PRETRAINING TEST SUITE")
    print("=" * 70)

    torch.manual_seed(42)

    temp_dir = "/tmp/dhruva_m4_test"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    text_path = os.path.join(temp_dir, "synthetic_corpus.txt")
    shard_path = os.path.join(temp_dir, "train_shard.bin")
    checkpoint_dir = os.path.join(temp_dir, "checkpoints")

    # Step 1: Generate synthetic text corpus
    print("Step 1: Generating synthetic text file...")
    sentence = "Northh Dhruva is the premier Indian foundation model engineered for deep reasoning. "
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(sentence * 40)
    assert os.path.isfile(text_path), "Failed to write synthetic text file"
    print(f"        Created '{text_path}' ({os.path.getsize(text_path)} bytes)")

    # Step 2: Train ByteBPETokenizer on synthetic corpus
    print("Step 2: Training ByteBPETokenizer...")
    tokenizer = ByteBPETokenizer()
    with open(text_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    tokenizer.train(raw_text, target_vocab_size=300)
    assert tokenizer.vocab_size == 300, f"Expected vocab 300, got {tokenizer.vocab_size}"
    print(f"        Tokenizer trained to vocab_size={tokenizer.vocab_size}")

    # Step 3: Build binary uint16 shard
    print("Step 3: Building binary uint16 memory-mapped shard...")
    token_ids = tokenizer.encode(sentence)
    seq_len = 32
    # Ensure chunk alignment for exact monotonic verification
    chunk = (token_ids * 5)[:seq_len]
    tokens_full = chunk * 40

    meta = build_binary_shard(tokens_full, shard_path)
    assert os.path.isfile(shard_path), "Binary shard not created"
    assert meta["num_tokens"] == len(tokens_full)
    print(f"        Built binary shard '{shard_path}' ({meta['num_tokens']} tokens)")

    # Step 4: Setup MemmapPretrainingDataset and DataLoader
    print("Step 4: Initializing MemmapPretrainingDataset & DataLoader...")
    dataset = MemmapPretrainingDataset(bin_path=shard_path, seq_len=seq_len)
    assert len(dataset) >= 1, "Dataset empty"
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"        Dataset samples: {len(dataset)}, batch_size: 1, seq_len: {seq_len}")

    # Step 5: Configure Tiny Debug Model
    print("Step 5: Initializing DhruvaForCausalLM model...")
    cfg = DhruvaConfig.tiny_debug()
    cfg.vocab_size = tokenizer.vocab_size
    cfg.max_seq_len = 128
    model = DhruvaForCausalLM(cfg)

    # Step 6: Setup DhruvaTrainer and execute 25 optimization steps
    print("Step 6: Executing DhruvaTrainer for 25 steps...")
    trainer = DhruvaTrainer(
        model=model,
        train_loader=loader,
        max_steps=25,
        grad_accum_steps=1,
        max_lr=5e-4,
        min_lr=5e-4,
        warmup_steps=0,
        checkpoint_dir=checkpoint_dir,
        save_interval=10,
        log_interval=5,
    )

    losses = trainer.fit(resume=False)
    assert len(losses) == 25, f"Expected 25 loss entries, got {len(losses)}"

    # Step 7: Verify strictly monotonic decrease in training loss
    print("\nStep 7: Verifying monotonic training loss curve...")
    print("        Step losses:")
    for idx, l in enumerate(losses):
        print(f"          Step {idx+1:2d}: {l:.4f}")

    is_monotonic = all(losses[i] >= losses[i + 1] for i in range(len(losses) - 1))
    assert is_monotonic, (
        f"Training loss is not monotonic! Violations: "
        f"{[(i, losses[i], losses[i+1]) for i in range(len(losses)-1) if losses[i] < losses[i+1]]}"
    )

    delta = losses[0] - losses[-1]
    assert delta > 1.0, f"Expected substantial loss drop, got delta={delta:.4f}"

    # Step 8: Verify checkpoint creation
    print("\nStep 8: Verifying checkpoint persistence...")
    last_ckpt = os.path.join(checkpoint_dir, "last.pt")
    assert os.path.isfile(last_ckpt), f"Checkpoint '{last_ckpt}' was not saved"
    checkpoint = torch.load(last_ckpt, map_location="cpu")
    assert checkpoint["step"] == 25, f"Expected checkpoint step 25, got {checkpoint['step']}"
    assert "model_state_dict" in checkpoint, "model_state_dict missing from checkpoint"
    assert "optimizer_state_dict" in checkpoint, "optimizer_state_dict missing from checkpoint"
    print(f"        Verified checkpoint '{last_ckpt}' at step 25")

    print("\n" + "=" * 70)
    print("  [PASS] MODULE 4 PRETRAINING RUNNER & VERIFICATION PASSED")
    print(f"  Initial Loss: {losses[0]:.4f} -> Final Loss: {losses[-1]:.4f} (Monotonic: True)")
    print("=" * 70)


if __name__ == "__main__":
    test_module4_pretrain_run()

