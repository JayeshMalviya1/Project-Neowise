#!/usr/bin/env python3
"""Production Data Ingestion, 95/5 Split, and Multi-Shard Builder for Northh Dhruva.

Default pipeline (Module 6):
1. Loads the trained `data/tokenizer_16k/` tokenizer.
2. Reads `data/raw/pilot_corpus.txt` (splitting on `<|endoftext|>`).
3. Performs deterministic 95/5 train/val split.
4. Writes uint16 binary shards (`train_00000.bin`, `val_00000.bin`) at 500,000 tokens/shard.
5. Verifies `manifest.json` token counts against actual `.bin` files on disk.
"""

import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Ensure project root is in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cleaner import clean_and_deduplicate_corpus
from src.data.shard_builder import create_shards_from_texts
from src.tokenizer.byte_bpe import ByteBPETokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Northh Dhruva Production Data Sharding Pipeline")
    parser.add_argument(
        "--input_path",
        "--corpus_path",
        dest="input_path",
        type=str,
        default="data/raw/pilot_corpus.txt",
        help="Path to raw text corpus (default: data/raw/pilot_corpus.txt)",
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default="data/tokenizer_16k",
        help="Directory to load (or save) ByteBPETokenizer (default: data/tokenizer_16k)",
    )
    parser.add_argument(
        "--shards_dir",
        type=str,
        default=None,
        help="Base shards directory (sets train_output_dir=<shards_dir>/train and val_output_dir=<shards_dir>/val)",
    )
    parser.add_argument(
        "--train_output_dir",
        type=str,
        default="data/shards/train",
        help="Destination directory for training binary shards",
    )
    parser.add_argument(
        "--val_output_dir",
        type=str,
        default="data/shards/val",
        help="Destination directory for validation binary shards",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional single output directory (overrides 95/5 split mode)",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=16384,
        help="Target vocabulary size if training fallback tokenizer",
    )
    parser.add_argument(
        "--shard_size_tokens",
        "--shard_size",
        dest="shard_size_tokens",
        type=int,
        default=500_000,
        help="Number of uint16 tokens per binary shard (default: 500,000)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.05,
        help="Validation split fraction (default: 0.05 for 95/5 split)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="train",
        help="Shard filename prefix when using single --output_dir mode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic train/val split",
    )
    return parser.parse_args()


def load_raw_documents(input_path: Optional[str]) -> List[str]:
    """Loads raw documents from a file (splitting on `<|endoftext|>` or `\\n\\n`) or directory."""
    if input_path and os.path.isfile(input_path):
        with open(input_path, "r", encoding="utf-8") as f:
            content = f.read()
        if "<|endoftext|>" in content:
            docs = [doc.strip() for doc in content.split("<|endoftext|>") if doc.strip()]
        else:
            docs = [doc.strip() for doc in content.split("\n\n") if doc.strip()]
        return docs or [content]

    if input_path and os.path.isdir(input_path):
        docs: List[str] = []
        for file_path in sorted(glob.glob(os.path.join(input_path, "**", "*.txt"), recursive=True)):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                if "<|endoftext|>" in content:
                    docs.extend(d.strip() for d in content.split("<|endoftext|>") if d.strip())
                else:
                    docs.append(content)
        return docs

    print(f"[PrepareData] Input '{input_path}' not found. Using default synthetic fallback corpus.")
    default_corpus = [
        "Northh Dhruva is an independent Indian foundation model engineered from scratch for deep reasoning.",
        "Deep neural networks utilize transformer architectures with self-attention mechanisms and rotary embeddings.",
        "Memory-mapped datasets allow massive text corpora to be streamed directly into GPU memory buffers without RAM overhead.",
        "Byte-level BPE compresses vocabulary representations while retaining complete round-trip fidelity across Indic languages.",
    ] * 50
    return default_corpus


def verify_manifest(output_dir: str) -> Dict[str, int]:
    """Reads manifest.json and verifies exact uint16 token counts across all shards on disk."""
    manifest_path = os.path.join(output_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Expected manifest.json not found at '{manifest_path}'")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    shard_paths = manifest.get("shard_paths") or manifest.get("shards") or []
    disk_tokens = 0
    for sp in shard_paths:
        if not os.path.isfile(sp):
            raise FileNotFoundError(f"Shard listed in manifest missing on disk: '{sp}'")
        arr = np.fromfile(sp, dtype=np.uint16)
        disk_tokens += len(arr)

    manifest_tokens = manifest["total_tokens"]
    assert disk_tokens == manifest_tokens, (
        f"Manifest mismatch in '{output_dir}': manifest says {manifest_tokens:,} tokens, "
        f"disk shards contain {disk_tokens:,} tokens."
    )

    print(
        f"[PrepareData] Verified manifest '{manifest_path}': "
        f"{len(shard_paths)} shards, {manifest['document_count']:,} docs, "
        f"{disk_tokens:,} uint16 tokens (exact match)."
    )
    return {
        "shards": len(shard_paths),
        "documents": manifest["document_count"],
        "tokens": disk_tokens,
    }


def split_train_val(
    docs: List[str], val_ratio: float = 0.05, seed: int = 42
) -> Tuple[List[str], List[str]]:
    """Splits documents deterministically into train and validation subsets."""
    if len(docs) <= 1:
        return docs, docs

    shuffled = list(docs)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_ratio)))
    val_count = min(val_count, len(shuffled) - 1)
    train_docs = shuffled[:-val_count]
    val_docs = shuffled[-val_count:]
    return train_docs, val_docs


def main() -> None:
    args = parse_args()
    if args.shards_dir is not None:
        args.train_output_dir = os.path.join(args.shards_dir, "train")
        args.val_output_dir = os.path.join(args.shards_dir, "val")

    print("=" * 70)
    print("  PROJECT BHARAT-LLM: PRODUCTION DATA SHARDING PIPELINE")
    print("=" * 70)

    # 1. Load & clean documents
    raw_docs = load_raw_documents(args.input_path)
    print(f"[PrepareData] Loaded {len(raw_docs):,} raw document candidates.")

    cleaned_docs = clean_and_deduplicate_corpus(raw_docs)
    print(f"[PrepareData] Cleaned & deduplicated: {len(cleaned_docs):,} unique documents retained.")
    if not cleaned_docs:
        print("[PrepareData] Error: No documents survived cleaning filters.")
        sys.exit(1)

    # 2. Load trained tokenizer
    tokenizer = ByteBPETokenizer()
    vocab_file = os.path.join(args.tokenizer_dir, "vocab.json")
    if os.path.isfile(vocab_file):
        print(f"[PrepareData] Loading trained tokenizer from '{args.tokenizer_dir}'...")
        tokenizer.load(args.tokenizer_dir)
        print(f"[PrepareData] Loaded tokenizer (vocab_size={tokenizer.vocab_size:,}).")
    else:
        print(
            f"[PrepareData] Warning: Tokenizer not found at '{args.tokenizer_dir}'. "
            f"Training fallback tokenizer to vocab_size={args.vocab_size}..."
        )
        tokenizer.train("\n".join(cleaned_docs[:2000]), target_vocab_size=args.vocab_size)
        tokenizer.save(args.tokenizer_dir)
        print(f"[PrepareData] Fallback tokenizer saved to '{args.tokenizer_dir}'.")

    # 3. Single output directory mode vs 95/5 split mode
    if args.output_dir is not None:
        shard_paths = create_shards_from_texts(
            texts=cleaned_docs,
            tokenizer=tokenizer,
            output_dir=args.output_dir,
            shard_size_tokens=args.shard_size_tokens,
            prefix=args.prefix,
        )
        verify_manifest(args.output_dir)
        print(f"[PrepareData] Generated {len(shard_paths)} shards in '{args.output_dir}'.")
        return

    train_docs, val_docs = split_train_val(cleaned_docs, val_ratio=args.val_ratio, seed=args.seed)
    print(
        f"[PrepareData] 95/5 Train/Val split: "
        f"{len(train_docs):,} train docs | {len(val_docs):,} val docs"
    )

    print(
        f"[PrepareData] Building train shards in '{args.train_output_dir}' "
        f"(shard_size={args.shard_size_tokens:,} tokens)..."
    )
    create_shards_from_texts(
        texts=train_docs,
        tokenizer=tokenizer,
        output_dir=args.train_output_dir,
        shard_size_tokens=args.shard_size_tokens,
        prefix="train",
    )
    verify_manifest(args.train_output_dir)

    print(
        f"[PrepareData] Building val shards in '{args.val_output_dir}' "
        f"(shard_size={args.shard_size_tokens:,} tokens)..."
    )
    create_shards_from_texts(
        texts=val_docs,
        tokenizer=tokenizer,
        output_dir=args.val_output_dir,
        shard_size_tokens=args.shard_size_tokens,
        prefix="val",
    )
    verify_manifest(args.val_output_dir)
    print("=" * 70)


if __name__ == "__main__":
    main()
