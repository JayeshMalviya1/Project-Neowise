#!/usr/bin/env python3
"""Streaming Data Ingestion, 32K BBPE Tokenizer, and 5.3B Shard Builder for Northh Dhruva 500M.

Features:
- Pure streaming via `datasets.load_dataset(..., streaming=True)` with zero massive raw text on disk.
- Interleaves 60% English (FineWeb-Edu) and 40% Hindi (Sangraha / Wikipedia).
- Mode 1: `--mode tokenizer` streams ~150 MB balanced text, trains 32,768-vocab ByteBPETokenizer,
  and saves `vocab.json` & `merges.json` to `data/tokenizer_32k/`.
- Mode 2: `--mode shards` streams and tokenizes directly into uint16 binary shards
  (`val_00000.bin`..`val_00004.bin` = 50M val tokens first, then `train_00000.bin`... up to 5.3B tokens),
  updating `manifest.json` after every shard for rock-solid resumability.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

# Unconditionally re-exec into project's dedicated .venv if not already inside it
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
venv_python = os.path.join(project_root, ".venv", "bin", "python")
if os.path.isfile(venv_python) and os.path.realpath(sys.executable) != os.path.realpath(venv_python):
    os.execv(venv_python, [venv_python] + sys.argv)

import numpy as np

# Ensure project root is in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cleaner import clean_document
from src.tokenizer.byte_bpe import ByteBPETokenizer
from scripts.train_tokenizer_production import train_fast_bpe, FERTILITY_SAMPLES


def _extract_text(record: Any) -> Optional[str]:
    """Extracts text content from heterogeneous dataset records."""
    if isinstance(record, str):
        return record
    if isinstance(record, dict):
        for key in ("text", "content", "article", "body", "document"):
            val = record.get(key)
            if isinstance(val, str) and val.strip():
                return val
    return None


def stream_english_dataset() -> Iterator[str]:
    """Streams English documents from FineWeb-Edu (with wikitext fallback)."""
    from datasets import load_dataset

    candidates = [
        ("HuggingFaceFW/fineweb-edu", {"name": "sample-10BT", "split": "train", "streaming": True}),
        ("wikitext", {"name": "wikitext-103-raw-v1", "split": "train", "streaming": True}),
    ]

    for ds_name, kwargs in candidates:
        try:
            print(f"[Stream] Initializing English dataset '{ds_name}'...")
            ds = load_dataset(ds_name, **kwargs)
            for item in ds:
                text = _extract_text(item)
                if text:
                    yield text
            return
        except Exception as exc:
            print(f"[Stream] English streaming candidate '{ds_name}' unavailable: {exc}")

    raise RuntimeError("All English dataset streaming candidates failed.")


def stream_hindi_dataset() -> Iterator[str]:
    """Streams Hindi documents from AI4Bharat Sangraha (with Wikipedia fallback)."""
    from datasets import load_dataset

    candidates = [
        ("ai4bharat/sangraha", {"name": "verified", "split": "hi", "streaming": True}),
        ("ai4bharat/sangraha", {"name": "verified", "data_dir": "verified/hi", "split": "train", "streaming": True}),
        ("wikimedia/wikipedia", {"name": "20231101.hi", "split": "train", "streaming": True}),
    ]

    for ds_name, kwargs in candidates:
        try:
            print(f"[Stream] Initializing Hindi/Indic dataset '{ds_name}'...")
            ds = load_dataset(ds_name, **kwargs)
            for item in ds:
                text = _extract_text(item)
                if text:
                    yield text
            return
        except Exception as exc:
            print(f"[Stream] Hindi candidate '{ds_name}' unavailable: {exc}")

    raise RuntimeError("All Hindi dataset streaming candidates failed.")


def stream_interleaved_corpus(en_ratio: float = 0.6) -> Iterator[Tuple[str, str]]:
    """Yields (lang, cleaned_doc) interleaving 60% English and 40% Hindi."""
    en_gen = stream_english_dataset()
    hi_gen = stream_hindi_dataset()
    seen_hashes: Set[str] = set()

    # Ratio: 3 English docs to 2 Hindi docs (3/5 = 60%, 2/5 = 40%)
    en_batch = int(round(en_ratio * 10))
    hi_batch = 10 - en_batch

    while True:
        # English batch
        for _ in range(en_batch):
            try:
                raw = next(en_gen)
                cleaned = clean_document(raw, seen_hashes=seen_hashes)
                if cleaned:
                    yield ("en", cleaned)
            except StopIteration:
                pass

        # Hindi batch
        for _ in range(hi_batch):
            try:
                raw = next(hi_gen)
                cleaned = clean_document(raw, seen_hashes=seen_hashes)
                if cleaned:
                    yield ("hi", cleaned)
            except StopIteration:
                pass


def build_32k_tokenizer(
    output_dir: str = "data/tokenizer_32k",
    vocab_size: int = 32768,
    target_mb: float = 150.0,
) -> None:
    """Streams ~150 MB of balanced text (90 MB En + 60 MB Hi) and trains 32K ByteBPETokenizer."""
    print("=" * 72)
    print("  PROJECT BHARAT-LLM: 32K MULTILINGUAL TOKENIZER TRAINING")
    print(f"  Target Vocab: {vocab_size:,} | Stream Target: {target_mb:.1f} MB (60% En / 40% Hi)")
    print(f"  Output Dir:   {output_dir}")
    print("=" * 72)

    os.makedirs(output_dir, exist_ok=True)
    en_target_bytes = int(target_mb * 0.60 * 1024 * 1024)
    hi_target_bytes = int(target_mb * 0.40 * 1024 * 1024)

    en_bytes = 0
    hi_bytes = 0
    text_chunks: List[str] = []

    print("[Tokenizer] Streaming balanced corpus directly into memory...")
    for lang, doc in stream_interleaved_corpus(en_ratio=0.6):
        doc_bytes = len(doc.encode("utf-8"))
        if lang == "en" and en_bytes < en_target_bytes:
            text_chunks.append(doc)
            en_bytes += doc_bytes
        elif lang == "hi" and hi_bytes < hi_target_bytes:
            text_chunks.append(doc)
            hi_bytes += doc_bytes

        if (en_bytes + hi_bytes) % (10 * 1024 * 1024) < doc_bytes:
            print(
                f"[Tokenizer Stream] English: {en_bytes / (1024 * 1024):.1f}/{en_target_bytes / (1024 * 1024):.1f} MB | "
                f"Hindi: {hi_bytes / (1024 * 1024):.1f}/{hi_target_bytes / (1024 * 1024):.1f} MB"
            )

        if en_bytes >= en_target_bytes and hi_bytes >= hi_target_bytes:
            break

    total_mb = (en_bytes + hi_bytes) / (1024 * 1024)
    print(f"[Tokenizer] Accumulated {len(text_chunks):,} documents ({total_mb:.2f} MB total).")

    # Combine text and train BPE
    training_text = "\n\n".join(text_chunks)
    tokenizer = ByteBPETokenizer()

    t0 = time.time()
    train_fast_bpe(tokenizer, training_text, target_vocab_size=vocab_size, log_interval=1000)
    elapsed = time.time() - t0

    tokenizer.save(output_dir)
    print(f"[Tokenizer] Saved 32K tokenizer to '{output_dir}' (Training took {elapsed:.1f}s).")

    # Benchmark fertility
    print("\n" + "=" * 70)
    print("  TOKENIZER FERTILITY & COMPRESSION BENCHMARK (32K VOCAB)")
    print("=" * 70)
    print(f"  {'Domain':<10} | {'Words':>6} | {'Bytes':>6} | {'Tokens':>6} | {'Fertility':>9} | {'Bytes/Tok':>9}")
    print("-" * 70)
    for domain, sample_text in FERTILITY_SAMPLES.items():
        words = len(sample_text.split())
        raw_bytes = len(sample_text.encode("utf-8"))
        tokens = len(tokenizer.encode(sample_text))
        fertility = tokens / words if words > 0 else 0.0
        bytes_per_tok = raw_bytes / tokens if tokens > 0 else 0.0
        print(f"  {domain:<10} | {words:6d} | {raw_bytes:6d} | {tokens:6d} | {fertility:9.2f} | {bytes_per_tok:9.2f}")
    print("=" * 70 + "\n")


class ShardManager:
    """Manages streaming binary uint16 shards and updates manifest.json atomically."""

    def __init__(self, output_dir: str, prefix: str, shard_size_tokens: int):
        self.output_dir = output_dir
        self.prefix = prefix
        self.shard_size_tokens = shard_size_tokens
        os.makedirs(output_dir, exist_ok=True)

        self.manifest_path = os.path.join(output_dir, "manifest.json")
        self.shard_paths: List[str] = []
        self.total_tokens: int = 0
        self.doc_count: int = 0
        self.shard_idx: int = 0

        # Load existing manifest for seamless resumption
        if os.path.isfile(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.shard_paths = data.get("shard_paths", [])
                self.total_tokens = data.get("total_tokens", 0)
                self.doc_count = data.get("document_count", 0)
                self.shard_idx = len(self.shard_paths)
                print(
                    f"[Resuming {prefix}] Found {len(self.shard_paths)} existing shards "
                    f"({self.total_tokens:,} tokens) in '{output_dir}'."
                )
            except Exception as exc:
                print(f"[Warning] Failed reading existing manifest '{self.manifest_path}': {exc}")

    def write_shard(self, token_chunk: List[int], docs_in_shard: int) -> str:
        shard_fname = f"{self.prefix}_{self.shard_idx:05d}.bin"
        shard_path = os.path.join(self.output_dir, shard_fname)

        arr = np.array(token_chunk, dtype=np.uint16)
        arr.tofile(shard_path)

        self.shard_paths.append(shard_path)
        self.shard_idx += 1
        self.total_tokens += len(token_chunk)
        self.doc_count += docs_in_shard

        # Update manifest.json immediately after every shard
        manifest_data = {
            "prefix": self.prefix,
            "total_tokens": self.total_tokens,
            "document_count": self.doc_count,
            "shard_count": len(self.shard_paths),
            "shard_size_tokens": self.shard_size_tokens,
            "shard_paths": self.shard_paths,
            "shards": self.shard_paths,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        return shard_path


def build_500m_shards(
    target_tokens: int = 5_300_000_000,
    shard_size: int = 10_000_000,
    val_tokens: int = 50_000_000,
    tokenizer_dir: str = "data/tokenizer_32k",
    shards_dir: str = "data/shards_500m",
    batch_size: int = 500,
    max_workers: int = 8,
) -> None:
    """Streams and encodes interleaved corpus into uint16 binary shards with resumability."""
    val_dir = os.path.join(shards_dir, "val")
    train_dir = os.path.join(shards_dir, "train")

    print("=" * 72)
    print("  PROJECT BHARAT-LLM: 5.3B TOKEN STREAMING BINARY SHARD BUILDER")
    print(f"  Target Tokens: {target_tokens:,} | Shard Size: {shard_size:,} tokens")
    print(f"  Validation:    {val_tokens:,} tokens ({val_tokens // shard_size} shards) -> {val_dir}")
    print(f"  Training:      {target_tokens - val_tokens:,} tokens -> {train_dir}")
    print(f"  Tokenizer Dir: {tokenizer_dir}")
    print("=" * 72)

    # 1. Load trained 32K tokenizer
    if not os.path.isfile(os.path.join(tokenizer_dir, "vocab.json")):
        raise FileNotFoundError(
            f"32K Tokenizer not found at '{tokenizer_dir}'. Run with '--mode tokenizer' first."
        )

    tokenizer = ByteBPETokenizer()
    tokenizer.load(tokenizer_dir)
    eos_id = getattr(tokenizer, "eos_token_id", 2)
    print(f"[Shards] Loaded tokenizer (vocab_size={tokenizer.vocab_size:,})")

    # 2. Initialize Shard Managers (with auto-resume)
    val_mgr = ShardManager(val_dir, prefix="val", shard_size_tokens=shard_size)
    train_mgr = ShardManager(train_dir, prefix="train", shard_size_tokens=shard_size)

    stream = stream_interleaved_corpus(en_ratio=0.6)
    token_buffer: List[int] = []
    docs_in_buffer = 0
    t_start = time.time()

    def encode_doc(text: str) -> List[int]:
        toks = tokenizer.encode(text)
        if not toks or toks[-1] != eos_id:
            toks.append(eos_id)
        return toks

    print("\n[Shards] Starting parallel streaming & binary sharding...")

    while (val_mgr.total_tokens + train_mgr.total_tokens) < target_tokens:
        # Collect batch of documents
        batch_docs: List[str] = []
        for _ in range(batch_size):
            try:
                _, doc = next(stream)
                batch_docs.append(doc)
            except StopIteration:
                break

        if not batch_docs:
            print("[Shards] Stream exhausted.")
            break

        # Parallel batch tokenization
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tokenized_batch = list(executor.map(encode_doc, batch_docs))

        for toks in tokenized_batch:
            token_buffer.extend(toks)
            docs_in_buffer += 1

            # Decide active manager: fill val first (50M tokens), then train
            active_mgr = val_mgr if val_mgr.total_tokens < val_tokens else train_mgr

            while len(token_buffer) >= shard_size:
                shard_chunk = token_buffer[:shard_size]
                token_buffer = token_buffer[shard_size:]

                sp = active_mgr.write_shard(shard_chunk, docs_in_buffer)
                docs_in_buffer = 0

                elapsed = time.time() - t_start
                overall_tokens = val_mgr.total_tokens + train_mgr.total_tokens
                tok_per_sec = overall_tokens / elapsed if elapsed > 0 else 0.0

                print(
                    f"[{active_mgr.prefix.upper()} Shard {active_mgr.shard_idx:04d}] "
                    f"Saved {os.path.basename(sp)} | "
                    f"Tokens: {overall_tokens:,}/{target_tokens:,} ({(overall_tokens / target_tokens) * 100:.2f}%) | "
                    f"Speed: {tok_per_sec:,.0f} tok/s"
                )

                # Switch to train if val is complete
                if active_mgr == val_mgr and val_mgr.total_tokens >= val_tokens:
                    active_mgr = train_mgr

    # Flush remaining tokens to final train shard if any
    if token_buffer:
        sp = train_mgr.write_shard(token_buffer, docs_in_buffer)
        print(f"[FINAL Shard] Flushed {len(token_buffer):,} remaining tokens to {os.path.basename(sp)}")

    print("\n" + "=" * 72)
    print("  500M BINARY SHARDING PIPELINE COMPLETE")
    print(f"  Val Shards:   {val_mgr.shard_idx} shards ({val_mgr.total_tokens:,} tokens)")
    print(f"  Train Shards: {train_mgr.shard_idx} shards ({train_mgr.total_tokens:,} tokens)")
    print(f"  Total Tokens: {val_mgr.total_tokens + train_mgr.total_tokens:,} tokens")
    print("=" * 72 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Northh Dhruva 500M Streaming Data & Tokenizer Pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["tokenizer", "shards"],
        help="Pipeline execution mode ('tokenizer' or 'shards')",
    )
    # Tokenizer mode args
    parser.add_argument(
        "--target_mb",
        type=float,
        default=150.0,
        help="Corpus volume in MB to stream for tokenizer training (default: 150.0 MB)",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=32768,
        help="Target Byte-Level BPE vocabulary size (default: 32768)",
    )
    parser.add_argument(
        "--output_dir",
        "--tokenizer_dir",
        dest="tokenizer_dir",
        type=str,
        default="data/tokenizer_32k",
        help="Directory to save or load 32K tokenizer (default: data/tokenizer_32k)",
    )
    # Shards mode args
    parser.add_argument(
        "--target_tokens",
        type=int,
        default=5_300_000_000,
        help="Target total token count across val + train (default: 5,300,000,000)",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=10_000_000,
        help="Number of uint16 tokens per binary shard (default: 10,000,000)",
    )
    parser.add_argument(
        "--val_tokens",
        type=int,
        default=50_000_000,
        help="Number of validation tokens to generate first (default: 50,000,000)",
    )
    parser.add_argument(
        "--shards_dir",
        type=str,
        default="data/shards_500m",
        help="Base directory for shards (contains train/ and val/ subdirectories)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=500,
        help="Parallel document batch size for tokenization (default: 500)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Number of parallel worker threads for tokenization (default: 8)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "tokenizer":
        build_32k_tokenizer(
            output_dir=args.tokenizer_dir,
            vocab_size=args.vocab_size,
            target_mb=args.target_mb,
        )
    elif args.mode == "shards":
        build_500m_shards(
            target_tokens=args.target_tokens,
            shard_size=args.shard_size,
            val_tokens=args.val_tokens,
            tokenizer_dir=args.tokenizer_dir,
            shards_dir=args.shards_dir,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
        )


if __name__ == "__main__":
    main()
