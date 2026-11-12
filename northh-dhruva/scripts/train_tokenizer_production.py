#!/usr/bin/env python3
"""Production 16K Byte-Level BPE Tokenizer Trainer for Northh Dhruva (Project BHARAT-LLM).

Trains a 16,384-vocabulary ByteBPETokenizer on `data/raw/pilot_corpus.txt` using an
inverted-index + max-heap BPE algorithm, logs merge milestones every 500 merges,
saves `vocab.json` & `merges.json` to `data/tokenizer_16k/`, and reports fertility
metrics across English, Hindi, and Python code samples.

Usage:
    python scripts/train_tokenizer_production.py
    python scripts/train_tokenizer_production.py --input_path data/raw/pilot_corpus.txt --vocab_size 16384
"""

import argparse
import heapq
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Ensure project root is in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tokenizer.byte_bpe import SPLIT_PATTERN, ByteBPETokenizer
from src.tokenizer.normalizer import normalize_text

FERTILITY_SAMPLES: Dict[str, str] = {
    "English": (
        "Northh Dhruva is an independent foundation model engineered from scratch "
        "for deep reasoning, multi-step mathematics, and sovereign multilingual understanding."
    ),
    "Hindi": (
        "नॉर्थ ध्रुव एक स्वतंत्र भारतीय फाउंडेशन मॉडल है जिसे गहन तर्कशक्ति और "
        "बहुभाषी समझ के लिए शून्य से तैयार किया गया है।"
    ),
    "Code": (
        "def scaled_dot_product(q, k, v, mask=None):\n"
        "    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))\n"
        "    return torch.matmul(torch.softmax(scores, dim=-1), v)\n"
    ),
}


def train_fast_bpe(
    tokenizer: ByteBPETokenizer,
    text: str,
    target_vocab_size: int = 16384,
    log_interval: int = 500,
) -> None:
    """Fast incremental BPE merge training populating a ByteBPETokenizer instance."""
    tokenizer._init_base_vocab()
    if target_vocab_size <= tokenizer.vocab_size:
        return

    cleaned_text = text.replace("<|endoftext|>", "\n")
    normalized = normalize_text(cleaned_text)
    words = SPLIT_PATTERN.findall(normalized)
    if not words:
        return

    raw_counts: Dict[Tuple[bytes, ...], int] = Counter(
        tuple(bytes([b]) for b in w.encode("utf-8")) for w in words
    )

    word_seqs: List[List[bytes]] = []
    word_freqs: List[int] = []
    pair_counts: Dict[Tuple[bytes, bytes], int] = defaultdict(int)
    pair_to_words: Dict[Tuple[bytes, bytes], Set[int]] = defaultdict(set)

    for wid, (seq, freq) in enumerate(raw_counts.items()):
        word_seqs.append(list(seq))
        word_freqs.append(freq)
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            pair_counts[pair] += freq
            pair_to_words[pair].add(wid)

    heap: List[Tuple[int, Tuple[bytes, bytes]]] = [
        (-cnt, pair) for pair, cnt in pair_counts.items() if cnt > 0
    ]
    heapq.heapify(heap)

    total_merges = target_vocab_size - tokenizer.vocab_size
    merges_done = 0
    t_start = time.time()

    print(
        f"[Tokenizer] Starting production BPE training: "
        f"unique_words={len(word_seqs):,}, target_vocab={target_vocab_size:,}, "
        f"merges_needed={total_merges:,}"
    )

    while tokenizer.vocab_size < target_vocab_size and heap:
        neg_cnt, best_pair = heapq.heappop(heap)
        cnt = -neg_cnt
        if pair_counts.get(best_pair, 0) != cnt or cnt <= 0:
            continue

        p0, p1 = best_pair
        new_token = p0 + p1
        new_id = len(tokenizer.vocab)
        tokenizer.vocab[new_token] = new_id
        tokenizer.id_to_token[new_id] = new_token
        tokenizer.merges[best_pair] = len(tokenizer.merges)
        merges_done += 1

        affected_wids = list(pair_to_words.get(best_pair, ()))
        dirty_pairs: Set[Tuple[bytes, bytes]] = set()

        for wid in affected_wids:
            seq = word_seqs[wid]
            freq = word_freqs[wid]
            if len(seq) < 2:
                continue

            old_pairs = [(seq[i], seq[i + 1]) for i in range(len(seq) - 1)]

            new_seq: List[bytes] = []
            i = 0
            n = len(seq)
            changed = False
            while i < n:
                if i < n - 1 and seq[i] == p0 and seq[i + 1] == p1:
                    new_seq.append(new_token)
                    i += 2
                    changed = True
                else:
                    new_seq.append(seq[i])
                    i += 1

            if not changed:
                continue

            word_seqs[wid] = new_seq
            for op in old_pairs:
                pair_counts[op] -= freq
                pair_to_words[op].discard(wid)
                dirty_pairs.add(op)

            for i in range(len(new_seq) - 1):
                np_pair = (new_seq[i], new_seq[i + 1])
                pair_counts[np_pair] += freq
                pair_to_words[np_pair].add(wid)
                dirty_pairs.add(np_pair)

        for dp in dirty_pairs:
            dp_cnt = pair_counts[dp]
            if dp_cnt > 0:
                heapq.heappush(heap, (-dp_cnt, dp))

        if log_interval > 0 and merges_done % log_interval == 0:
            elapsed = time.time() - t_start
            print(
                f"[Tokenizer] Milestone {merges_done:5d}/{total_merges} | "
                f"Vocab: {tokenizer.vocab_size:5d} | "
                f"Latest: {new_token!r} (freq={cnt:,}) | "
                f"Elapsed: {elapsed:.1f}s"
            )

    tokenizer.cache = {}
    total_elapsed = time.time() - t_start
    print(
        f"[Tokenizer] Training finished in {total_elapsed:.2f}s. "
        f"Final vocab_size={tokenizer.vocab_size:,}"
    )


def compute_fertility_metrics(tokenizer: ByteBPETokenizer) -> Dict[str, Dict[str, float]]:
    """Computes and prints tokenizer fertility & compression across English, Hindi, and Code."""
    metrics: Dict[str, Dict[str, float]] = {}

    print("\n" + "=" * 70)
    print("  TOKENIZER FERTILITY & COMPRESSION BENCHMARK")
    print("=" * 70)
    print(f"  {'Domain':<10} | {'Words':>6} | {'Bytes':>6} | {'Tokens':>6} | {'Fertility':>9} | {'Bytes/Tok':>9}")
    print("-" * 70)

    for domain, sample in FERTILITY_SAMPLES.items():
        token_ids = tokenizer.encode(sample)
        num_tokens = max(1, len(token_ids))
        num_words = max(1, len(sample.split()))
        num_bytes = len(sample.encode("utf-8"))
        fertility = num_tokens / num_words
        bytes_per_token = num_bytes / num_tokens

        metrics[domain] = {
            "words": num_words,
            "bytes": num_bytes,
            "tokens": num_tokens,
            "fertility": fertility,
            "bytes_per_token": bytes_per_token,
        }
        print(
            f"  {domain:<10} | {num_words:>6d} | {num_bytes:>6d} | {num_tokens:>6d} | "
            f"{fertility:>9.2f} | {bytes_per_token:>9.2f}"
        )

    print("=" * 70)
    return metrics


def train_production_tokenizer(
    input_path: str = "data/raw/pilot_corpus.txt",
    output_dir: str = "data/tokenizer_16k",
    vocab_size: int = 16384,
    log_interval: int = 500,
    max_chars: int = 20_000_000,
) -> ByteBPETokenizer:
    """Loads pilot corpus, trains 16K ByteBPETokenizer, saves artifacts, and logs fertility."""
    if not os.path.isfile(input_path):
        raise FileNotFoundError(
            f"Pilot corpus not found at '{input_path}'. Run `scripts/download_pilot_data.py` first."
        )

    print(f"[Tokenizer] Reading corpus from '{input_path}'...")
    with open(input_path, "r", encoding="utf-8") as f:
        corpus_text = f.read(max_chars) if max_chars > 0 else f.read()

    print(f"[Tokenizer] Loaded {len(corpus_text):,} characters for training.")

    tokenizer = ByteBPETokenizer()
    train_fast_bpe(
        tokenizer=tokenizer,
        text=corpus_text,
        target_vocab_size=vocab_size,
        log_interval=log_interval,
    )

    tokenizer.save(output_dir)
    print(f"[Tokenizer] Saved vocab.json & merges.json to '{output_dir}'.")

    compute_fertility_metrics(tokenizer)
    return tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train production 16K ByteBPETokenizer")
    parser.add_argument(
        "--input_path",
        "--corpus_path",
        dest="input_path",
        type=str,
        default="data/raw/pilot_corpus.txt",
        help="Path to raw cleaned pilot corpus",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/tokenizer_16k",
        help="Destination directory for vocab.json and merges.json",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=16384,
        help="Target vocabulary size (default: 16384)",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=500,
        help="Merge milestone logging frequency",
    )
    parser.add_argument(
        "--max_chars",
        type=int,
        default=20_000_000,
        help="Max characters read from corpus for tokenizer training (<=0 for full file)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_production_tokenizer(
        input_path=args.input_path,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        log_interval=args.log_interval,
        max_chars=args.max_chars,
    )


if __name__ == "__main__":
    main()

