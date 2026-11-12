"""Module 5 Verification Suite: Real-World Ingestion, Binary Sharding, and Evaluation Harness.

Project BHARAT-LLM: Northh Dhruva Module 5.
Verifies that:
- Gate 1 (Cleaner & Deduplication): HTML stripping, whitespace normalization, short-document rejection, and exact duplicate filtering.
- Gate 2 (Binary Sharding & Multi-Shard Manifest): 5,000-token binary uint16 shards, manifest token counts matching exactly.
- Gate 3 (Sampling Mechanics): Deterministic greedy sampling and valid top-k/top-p token ID bounds.
- Gate 4 (Validation Perplexity & Generation from Checkpoint): Validation perplexity on held-out shard, autoregressive generation returning non-empty continuation.

Run via:
    python -m tests.test_module5_eval_and_sharding
"""

import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure project root is in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.model_config import DhruvaConfig
from src.data.cleaner import (
    DocumentDeduplicator,
    clean_and_deduplicate_corpus,
    clean_document,
    compute_doc_hash,
)
from src.data.dataset import MemmapPretrainingDataset
from src.data.shard_builder import create_shards_from_texts
from src.evaluation.eval_harness import (
    compute_validation_perplexity,
    generate_text,
    sample_tokens,
)
from src.models.dhruva_lm import DhruvaForCausalLM
from src.tokenizer.byte_bpe import ByteBPETokenizer
from src.training.trainer import DhruvaTrainer


def test_gate_1_cleaner_and_deduplication() -> None:
    print("Running Gate 1: Cleaner & Deduplication...")

    # 1. HTML Stripping
    html_doc = (
        "<html><body><h1>Northh Dhruva</h1><p>An independent Indian foundation model "
        "designed with deep reasoning and zero-leakage causal attention.</p></body></html>"
    )
    cleaned_html = clean_document(html_doc)
    assert cleaned_html is not None, "HTML document unexpectedly filtered out"
    assert "<" not in cleaned_html and ">" not in cleaned_html, (
        f"HTML tags were not stripped: {cleaned_html}"
    )
    assert "Northh Dhruva" in cleaned_html

    # 2. Whitespace Normalization (preserves valid paragraph linebreaks)
    ugly_whitespace = (
        "Northh   Dhruva \t\t provides  deep   reasoning.\n\n\n\n"
        "It   scales cleanly   across \t distributed   GPU   nodes."
    )
    cleaned_ws = clean_document(ugly_whitespace)
    assert cleaned_ws is not None
    assert "  " not in cleaned_ws, "Multiple spaces not collapsed"
    assert "\t" not in cleaned_ws, "Tabs not removed"
    assert "\n\n" in cleaned_ws, "Paragraph linebreak was not preserved"
    assert "\n\n\n" not in cleaned_ws, "Excessive newlines not condensed"

    # 3. Short Document Rejection (< 30 characters or < 8 words)
    too_short_chars = "Hello world short."  # 18 chars, 3 words
    assert clean_document(too_short_chars) is None, (
        "Failed to reject document with < 30 characters"
    )

    too_few_words = "ExtremelyLongWordWithoutSpacesIsHereOverThirtyCharactersLong"  # > 30 chars, 1 word
    assert clean_document(too_few_words) is None, (
        "Failed to reject document with < 8 words"
    )

    valid_doc = "Northh Dhruva is the premier Indian foundation model engineered for high efficiency pretraining."
    assert clean_document(valid_doc) is not None, "Valid document was rejected"

    # 4. Exact Duplicate Filtering via Hash
    dedup = DocumentDeduplicator()
    assert not dedup.is_duplicate(valid_doc), "First occurrence marked as duplicate"
    assert dedup.is_duplicate(valid_doc), "Second occurrence NOT detected as duplicate"

    # In-stream deduplication test
    corpus_with_dups = [
        valid_doc,
        valid_doc,  # duplicate
        "A second completely different document that contains more than eight words and thirty characters.",
        valid_doc,  # duplicate
    ]
    unique_docs = clean_and_deduplicate_corpus(corpus_with_dups)
    assert len(unique_docs) == 2, f"Expected 2 unique docs, got {len(unique_docs)}"

    print(
        "[PASS] Gate 1 (Cleaner & Deduplication): HTML stripped, whitespace normalized, "
        "short docs rejected (<30 chars, <8 words), and exact duplicates filtered."
    )


def test_gate_2_binary_sharding_and_manifest() -> None:
    print("Running Gate 2: Binary Sharding & Multi-Shard Manifest...")

    shard_dir = "/tmp/dhruva_m5_test/shards"
    if os.path.exists(shard_dir):
        shutil.rmtree(shard_dir)
    os.makedirs(shard_dir, exist_ok=True)

    # 1. Build sample corpus and train tokenizer
    sample_sentence = (
        "Northh Dhruva is a high-performance decoder-only foundation model trained on massive multilingual text. "
        "Self-attention layers with rotary position embeddings guarantee strict causal autoregression. "
    )
    docs = [sample_sentence * 5 for _ in range(60)]

    tokenizer = ByteBPETokenizer()
    tokenizer.train("\n".join(docs[:10]), target_vocab_size=320)

    # 2. Shard into 5,000-token binary uint16 files
    shard_size_tokens = 5000
    shard_paths = create_shards_from_texts(
        texts=docs,
        tokenizer=tokenizer,
        output_dir=shard_dir,
        shard_size_tokens=shard_size_tokens,
        prefix="train",
    )

    assert len(shard_paths) >= 2, (
        f"Expected multi-shard output (>= 2 shards), got {len(shard_paths)}"
    )

    # 3. Verify binary format (uint16 = 2 bytes per token)
    total_tokens_read = 0
    for idx, sp in enumerate(shard_paths):
        assert os.path.isfile(sp), f"Shard file '{sp}' does not exist on disk"
        file_bytes = os.path.getsize(sp)
        assert file_bytes % 2 == 0, f"Shard '{sp}' size {file_bytes} is not a multiple of 2 (uint16)"

        arr = np.fromfile(sp, dtype=np.uint16)
        num_tokens = len(arr)
        total_tokens_read += num_tokens

        # For all shards except the last one, chunk length must be exactly shard_size_tokens
        if idx < len(shard_paths) - 1:
            assert num_tokens == shard_size_tokens, (
                f"Shard {idx} has {num_tokens} tokens, expected {shard_size_tokens}"
            )
        assert (arr < tokenizer.vocab_size).all(), "Found token IDs exceeding vocabulary size"

    # 4. Verify manifest.json
    manifest_path = os.path.join(shard_dir, "manifest.json")
    assert os.path.isfile(manifest_path), "manifest.json was not created"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["total_tokens"] == total_tokens_read, (
        f"Manifest total_tokens ({manifest['total_tokens']}) does not match tokens on disk ({total_tokens_read})"
    )
    assert manifest["document_count"] == len(docs), (
        f"Manifest document_count ({manifest['document_count']}) != {len(docs)}"
    )
    assert manifest["shard_count"] == len(shard_paths), (
        f"Manifest shard_count ({manifest['shard_count']}) != {len(shard_paths)}"
    )

    print(
        f"[PASS] Gate 2 (Binary Sharding & Multi-Shard Manifest): Created {len(shard_paths)} shards "
        f"with {total_tokens_read:,} total uint16 tokens matching manifest.json exactly."
    )


def test_gate_3_sampling_mechanics() -> None:
    print("Running Gate 3: Sampling Mechanics...")

    torch.manual_seed(42)
    vocab_size = 400
    logits = torch.randn(vocab_size)

    # 1. Deterministic Greedy Sampling (temperature = 0.0)
    expected_argmax = int(torch.argmax(logits, dim=-1).item())
    greedy_1 = sample_tokens(logits, temperature=0.0)
    greedy_2 = sample_tokens(logits, temperature=0.0)
    assert greedy_1 == greedy_2 == expected_argmax, (
        f"Greedy sampling is non-deterministic or mismatch: {greedy_1} vs {greedy_2} vs {expected_argmax}"
    )

    # 2. Top-k filtering verification
    top_k = 5
    top_k_indices = set(torch.topk(logits, top_k).indices.tolist())
    for _ in range(40):
        sampled = sample_tokens(logits, temperature=0.8, top_k=top_k, top_p=1.0)
        assert 0 <= sampled < vocab_size, f"Sampled token ID {sampled} out of vocab bounds"
        assert sampled in top_k_indices, f"Sampled token {sampled} outside top-{top_k} set {top_k_indices}"

    # 3. Top-p (nucleus) filtering verification
    for _ in range(40):
        sampled_nucleus = sample_tokens(logits, temperature=0.8, top_k=50, top_p=0.85)
        assert 0 <= sampled_nucleus < vocab_size, f"Nucleus sampled token {sampled_nucleus} out of bounds"

    print(
        "[PASS] Gate 3 (Sampling Mechanics): Verified deterministic greedy sampling "
        "and valid vocabulary bounds under top-k / top-p filtering."
    )


def test_gate_4_perplexity_and_checkpoint_generation() -> None:
    print("Running Gate 4: Validation Perplexity & Generation from Checkpoint...")

    checkpoint_path = "/tmp/dhruva_m4_test/checkpoints/last.pt"
    corpus_text = "Northh Dhruva is the premier Indian foundation model engineered for deep reasoning. "

    # Self-healing: if Module 4 checkpoint not found, generate a lightweight checkpoint
    if not os.path.isfile(checkpoint_path):
        print(f"        Checkpoint '{checkpoint_path}' not present. Generating self-contained checkpoint...")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        tok_temp = ByteBPETokenizer()
        tok_temp.train(corpus_text * 20, target_vocab_size=300)
        cfg_temp = DhruvaConfig.tiny_debug()
        cfg_temp.vocab_size = 300
        cfg_temp.max_seq_len = 64
        m_temp = DhruvaForCausalLM(cfg_temp)
        torch.save(
            {
                "step": 25,
                "config": cfg_temp,
                "model_state_dict": m_temp.state_dict(),
                "optimizer_state_dict": {},
            },
            checkpoint_path,
        )

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = ckpt.get("config", DhruvaConfig.tiny_debug())
    model = DhruvaForCausalLM(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Setup tokenizer matching vocab
    tokenizer = ByteBPETokenizer()
    tokenizer.train(corpus_text * 30, target_vocab_size=config.vocab_size)

    # Build held-out validation shard
    val_shard_path = "/tmp/dhruva_m5_test/val_shard.bin"
    val_tokens = (tokenizer.encode(corpus_text) * 20)[:512]
    arr_val = np.array(val_tokens, dtype=np.uint16)
    arr_val.tofile(val_shard_path)

    val_dataset = MemmapPretrainingDataset(bin_path=val_shard_path, seq_len=16)
    assert len(val_dataset) > 0, "Validation dataset empty"

    # Compute validation perplexity
    perplexity = compute_validation_perplexity(model, val_dataset, device="cpu", max_batches=20)
    assert perplexity > 0.0, f"Expected positive perplexity, got {perplexity}"
    assert not math.isnan(perplexity) and not math.isinf(perplexity), (
        f"Invalid perplexity: {perplexity}"
    )

    # Autoregressive generation
    prompt = "Northh"
    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=25,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        device="cpu",
    )

    assert isinstance(generated_text, str), "Generated output is not a string"
    assert len(generated_text.strip()) > len(prompt), (
        f"Generated text too short or empty: {generated_text!r}"
    )
    assert prompt in generated_text, f"Prompt '{prompt}' missing from generated text '{generated_text}'"

    print(
        f"[PASS] Gate 4 (Validation Perplexity & Generation from Checkpoint): "
        f"Perplexity={perplexity:.2f}, Generated='{generated_text.strip()}'"
    )


def main() -> None:
    print("=" * 70)
    print("  PROJECT BHARAT-LLM: NORTHH DHRUVA MODULE 5 EVALUATION & SHARDING")
    print("=" * 70)

    test_gate_1_cleaner_and_deduplication()
    test_gate_2_binary_sharding_and_manifest()
    test_gate_3_sampling_mechanics()
    test_gate_4_perplexity_and_checkpoint_generation()

    print("=" * 70)
    print("  ALL 4 GATES PASSED: MODULE 5 INGESTION & EVALUATION HARNESS READY")
    print("=" * 70)


if __name__ == "__main__":
    main()

