"""Standalone Tokenizer, Data Pipeline, and End-to-End Training Verification Suite.

Project BHARAT-LLM: Northh Dhruva Module 2.
Runs 6 strict engineering gates to verify Byte-Level BPE, lossless normalization,
zero-UNK multilingual fallback, binary dataset memmap loading, and causal language model
end-to-end memorization & autoregressive text generation.

Run via:
    python -m tests.test_tokenizer_and_data
"""

import os
import shutil
import sys
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Ensure project root is in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.model_config import DhruvaConfig
from src.models.dhruva_lm import DhruvaForCausalLM
from src.tokenizer.normalizer import normalize_text
from src.tokenizer.byte_bpe import ByteBPETokenizer
from src.data.dataset import build_binary_shard, MemmapPretrainingDataset

SAMPLE_CORPUS = """
The quick brown fox jumps over the lazy dog. Artificial intelligence and machine learning are transforming the modern computational world.
Northh Dhruva represents an ambitious foundation model engineered from scratch for deep reasoning, mathematical problem solving, and multilingual processing across Indic languages.
Deep neural networks utilize transformer architectures with self-attention mechanisms, rotary position embeddings, and grouped-query attention to process sequential data efficiently.
Memory-mapped datasets allow massive text corpora to be streamed directly from storage into GPU memory buffers without high RAM overhead.
Byte-level byte-pair encoding compresses vocabulary representations while retaining complete round-trip fidelity and zero out-of-vocabulary tokens for arbitrary UTF-8 byte streams.
Training neural systems requires stable gradients, careful learning rate schedules, weight decay regularization, and verified loss curves.
""" * 5


def test_gate_1_base_vocab_and_merge_training() -> ByteBPETokenizer:
    """Gate 1: Verify base vocabulary of 260 and training compression to vocab_size 380."""
    print("Running Gate 1: Base Vocab & Merge Training...")
    tok = ByteBPETokenizer()
    assert tok.vocab_size == 260, f"Expected initial vocab 260, got {tok.vocab_size}"

    tok.train(SAMPLE_CORPUS, target_vocab_size=380)
    assert tok.vocab_size == 380, f"Expected trained vocab 380, got {tok.vocab_size}"

    # Verify compression
    raw_byte_count = len(normalize_text(SAMPLE_CORPUS).encode("utf-8"))
    token_ids = tok.encode(SAMPLE_CORPUS)
    assert len(token_ids) < raw_byte_count, (
        f"Expected compression: {len(token_ids)} tokens should be < {raw_byte_count} raw bytes"
    )
    ratio = raw_byte_count / len(token_ids)
    print(
        f"[PASS] Gate 1 (Base Vocab & Merge Training): Initial=260, Final={tok.vocab_size}, "
        f"Compression: {raw_byte_count} bytes -> {len(token_ids)} tokens ({ratio:.2f}x)"
    )
    return tok


def test_gate_2_lossless_round_trip(tokenizer: ByteBPETokenizer) -> None:
    """Gate 2: Verify 100% lossless round-trip reconstruction on unseen English text."""
    print("Running Gate 2: 100% Lossless Round-Trip...")
    test_cases = [
        "Unseen sentence 1: Testing BPE tokenization with numbers 1234567890!",
        "Special symbols: @#$%^&*()_+-=[]{}|;':\",./<>? and spaces.",
        "Multiple lines:\nFirst line.\nSecond line with tabs:\tand spaces.",
        "Case sensitivity and contractions: It's isn't won't shouldn't Dhruva DHRUVA dhruva.",
    ]

    for text in test_cases:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        expected = normalize_text(text)
        assert decoded == expected, (
            f"Round-trip mismatch!\nExpected: {expected!r}\nGot:      {decoded!r}"
        )

    print("[PASS] Gate 2 (100% Lossless Round-Trip): Verified exact round-trip reconstruction on unseen English text")


def test_gate_3_zero_unk_multilingual_fallback(tokenizer: ByteBPETokenizer) -> None:
    """Gate 3: Verify 0 <unk> tokens and exact round-trip on Indic text with emojis."""
    print("Running Gate 3: Zero <unk> Multilingual Byte Fallback...")
    indic_text = "Northh Dhruva: नमस्ते भारत 🇮🇳"
    encoded = tokenizer.encode(indic_text)

    # UNK ID is 3; verify it is never generated
    assert tokenizer.unk_token_id not in encoded, (
        f"Found UNK token ({tokenizer.unk_token_id}) in encoded sequence: {encoded}"
    )

    decoded = tokenizer.decode(encoded)
    assert decoded == indic_text, (
        f"Multilingual fallback mismatch!\nExpected: {indic_text!r}\nGot:      {decoded!r}"
    )

    print(
        f"[PASS] Gate 3 (Zero <unk> Multilingual Byte Fallback): 0 <unk> tokens, "
        f"exact round-trip reconstruction for '{indic_text}'"
    )


def test_gate_4_save_load_serialization_parity(tokenizer: ByteBPETokenizer) -> None:
    """Gate 4: Verify saving to disk and reloading gives 100% identical outputs."""
    print("Running Gate 4: Save/Load Serialization Parity...")
    save_dir = "/tmp/dhruva_tok"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    tokenizer.save(save_dir)
    assert os.path.exists(os.path.join(save_dir, "vocab.json")), "vocab.json missing"
    assert os.path.exists(os.path.join(save_dir, "merges.json")), "merges.json missing"

    loaded_tok = ByteBPETokenizer.from_pretrained(save_dir)
    assert loaded_tok.vocab_size == tokenizer.vocab_size, (
        f"Vocab size mismatch after reload: {loaded_tok.vocab_size} vs {tokenizer.vocab_size}"
    )

    sample_sentences = [
        "Northh Dhruva foundation model.",
        "Testing parity across reload: नमस्ते भारत 🇮🇳 with numbers 42 and punctuation!",
    ]
    for sentence in sample_sentences:
        orig_ids = tokenizer.encode(sentence, add_bos=True, add_eos=True)
        loaded_ids = loaded_tok.encode(sentence, add_bos=True, add_eos=True)
        assert orig_ids == loaded_ids, (
            f"Encoding disparity after disk reload!\nOriginal: {orig_ids}\nLoaded:   {loaded_ids}"
        )
        assert loaded_tok.decode(loaded_ids) == tokenizer.decode(orig_ids)

    print(
        f"[PASS] Gate 4 (Save/Load Serialization Parity): Verified identical token IDs "
        f"and text reconstruction after disk reload from '{save_dir}'"
    )


def test_gate_5_binary_shard_and_memmap_loader(tokenizer: ByteBPETokenizer) -> None:
    """Gate 5: Verify binary uint16 shard creation and memmap DataLoader shapes and dtypes."""
    print("Running Gate 5: Binary uint16 Shard & np.memmap Loader...")
    shard_path = "/tmp/train_shard.bin"
    if os.path.exists(shard_path):
        os.remove(shard_path)

    # Encode a substantial corpus to build the binary shard
    large_text = SAMPLE_CORPUS * 4
    token_ids = tokenizer.encode(large_text)
    assert len(token_ids) >= 256, f"Expected at least 256 tokens, got {len(token_ids)}"

    meta = build_binary_shard(token_ids, shard_path)
    assert meta["num_tokens"] == len(token_ids)
    assert meta["file_size_bytes"] == len(token_ids) * 2  # 2 bytes per uint16

    seq_len = 64
    batch_size = 4
    dataset = MemmapPretrainingDataset(bin_path=shard_path, seq_len=seq_len)
    assert len(dataset) >= batch_size, f"Dataset length ({len(dataset)}) is too small for batch_size 4"

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    x, y = next(iter(loader))

    assert x.shape == (batch_size, seq_len), f"Expected x shape ({batch_size}, {seq_len}), got {x.shape}"
    assert y.shape == (batch_size, seq_len), f"Expected y shape ({batch_size}, {seq_len}), got {y.shape}"
    assert x.dtype == torch.int64, f"Expected x dtype torch.int64, got {x.dtype}"
    assert y.dtype == torch.int64, f"Expected y dtype torch.int64, got {y.dtype}"
    # Verify alignment: y is chunk.clone() for internal shift in forward
    assert torch.equal(x, y), "Expected x and y to be identical chunk tensors for model-internal shift"

    print(
        f"[PASS] Gate 5 (Binary uint16 Shard & np.memmap Loader): Shard size={meta['num_tokens']} tokens, "
        f"DataLoader shape={tuple(x.shape)}, dtype={x.dtype}"
    )


def test_gate_6_end_to_end_training_and_generation(tokenizer: ByteBPETokenizer) -> None:
    """Gate 6: End-to-end text memorization and English generation with DhruvaForCausalLM."""
    print("Running Gate 6: End-to-End Text Memorization & English Generation...")
    torch.manual_seed(42)

    target_sentence = (
        "Northh Dhruva is an independent Indian foundation model engineered from scratch for deep reasoning. "
    )
    repeated_corpus = target_sentence * 50
    token_ids = tokenizer.encode(repeated_corpus)

    memorize_shard_path = "/tmp/memorize_shard.bin"
    if os.path.exists(memorize_shard_path):
        os.remove(memorize_shard_path)
    build_binary_shard(token_ids, memorize_shard_path)

    seq_len = 32
    dataset = MemmapPretrainingDataset(bin_path=memorize_shard_path, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Configure tiny debug model with vocab_size=380
    config = DhruvaConfig.tiny_debug()
    config.vocab_size = tokenizer.vocab_size
    config.max_seq_len = 256
    model = DhruvaForCausalLM(config)

    optimizer = optim.AdamW(model.parameters(), lr=3e-3)
    model.train()

    data_iter = iter(loader)
    final_loss = None
    for step in range(100):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)

        optimizer.zero_grad()
        outputs = model(x, labels=y)
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    assert final_loss is not None and final_loss < 0.05, (
        f"Expected final loss < 0.05 after 100 steps, got {final_loss:.4f}"
    )

    # Test generation from prompt
    prompt = "Northh"
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    generated_ids = model.generate(prompt_ids, max_new_tokens=25, sample=False)
    generated_text = tokenizer.decode(generated_ids[0].tolist())

    print(f"       Prompt:    {prompt!r}")
    print(f"       Generated: {generated_text!r}")

    assert (
        "independent" in generated_text or "foundation" in generated_text
    ), f"Expected 'independent' or 'foundation' in generated text, got: {generated_text!r}"

    print(
        f"[PASS] Gate 6 (End-to-End Text Memorization & English Generation): "
        f"Final Loss={final_loss:.4f} (< 0.05), Keyword verified in output"
    )


def main() -> None:
    print("=" * 70)
    print("  PROJECT BHARAT-LLM: NORTHH DHRUVA TOKENIZER & DATA PIPELINE SUITE")
    print("=" * 70)

    tok = test_gate_1_base_vocab_and_merge_training()
    test_gate_2_lossless_round_trip(tok)
    test_gate_3_zero_unk_multilingual_fallback(tok)
    test_gate_4_save_load_serialization_parity(tok)
    test_gate_5_binary_shard_and_memmap_loader(tok)
    test_gate_6_end_to_end_training_and_generation(tok)

    print("=" * 70)
    print("  ALL 6 GATES PASSED: NORTHH DHRUVA TOKENIZER & DATA PIPELINE READY")
    print("=" * 70)


if __name__ == "__main__":
    main()

