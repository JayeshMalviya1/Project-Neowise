#!/usr/bin/env python3
"""Streaming Pilot Corpus Downloader & Cleaner for Northh Dhruva (Project BHARAT-LLM).

Streams balanced English and Indic text corpora via Hugging Face `datasets`
(with direct HTTP fallback if `datasets` is unavailable), passes every document
through `src.data.cleaner.clean_document` for quality filtering and SHA-256
deduplication, and saves the unified corpus with `<|endoftext|>` separators.

Usage:
    python scripts/download_pilot_data.py
    python scripts/download_pilot_data.py --en_limit 10000 --indic_limit 10000
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set

# Ensure project root is in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cleaner import clean_document

SEPARATOR = "\n<|endoftext|>\n"

# Fallback direct HTTP raw text URLs when Hugging Face `datasets` is unavailable
HTTP_FALLBACK_SOURCES: Dict[str, List[str]] = {
    "en": [
        "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",  # Pride and Prejudice
        "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",  # Sherlock Holmes
    ],
    "indic": [
        "https://raw.githubusercontent.com/cltk/hindi_text_ltrc/master/README.md",
    ],
}


def _extract_text(record: dict) -> Optional[str]:
    """Extracts text field from heterogeneous Hugging Face dataset records."""
    for key in ("text", "content", "article", "body", "document"):
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return None


def stream_english_hf(limit: int, dataset_name: str = "HuggingFaceFW/fineweb-edu") -> Iterator[str]:
    """Streams English documents from Hugging Face `datasets`."""
    from datasets import load_dataset

    candidates = [
        (dataset_name, {"name": "sample-10BT", "split": "train", "streaming": True}),
        ("wikitext", {"name": "wikitext-103-raw-v1", "split": "train", "streaming": True}),
    ]
    if dataset_name == "wikitext-103-raw-v1":
        candidates.reverse()

    last_err: Optional[Exception] = None
    for ds_name, kwargs in candidates:
        try:
            print(f"[Downloader] Streaming English dataset '{ds_name}' ({kwargs.get('name')})...")
            ds = load_dataset(ds_name, **kwargs)
            count = 0
            for item in ds:
                text = _extract_text(item)
                if text:
                    yield text
                    count += 1
                    if count >= limit:
                        return
            return
        except Exception as exc:
            last_err = exc
            print(f"[Downloader] Warning: Failed streaming '{ds_name}': {exc}")

    if last_err is not None:
        raise last_err


def stream_indic_hf(
    limit: int,
    dataset_name: str = "wikimedia/wikipedia",
    langs: Optional[List[str]] = None,
) -> Iterator[str]:
    """Streams balanced Indic documents across Hindi, Marathi, and Tamil."""
    from datasets import load_dataset

    target_langs = langs or ["hi", "mr", "ta"]
    per_lang_limit = max(1, limit // len(target_langs))

    for lang in target_langs:
        lang_candidates = []
        if dataset_name == "ai4bharat/sangraha":
            lang_candidates.append(("ai4bharat/sangraha", {"name": "verified", "data_dir": f"verified/{lang}", "split": "train", "streaming": True}))
        lang_candidates.append(("wikimedia/wikipedia", {"name": f"20231101.{lang}", "split": "train", "streaming": True}))
        lang_candidates.append(("ai4bharat/sangraha", {"name": "verified", "split": lang, "streaming": True}))

        yielded_for_lang = 0
        for ds_name, kwargs in lang_candidates:
            try:
                print(f"[Downloader] Streaming Indic ({lang}) from '{ds_name}'...")
                ds = load_dataset(ds_name, **kwargs)
                for item in ds:
                    text = _extract_text(item)
                    if text:
                        yield text
                        yielded_for_lang += 1
                        if yielded_for_lang >= per_lang_limit:
                            break
                if yielded_for_lang > 0:
                    break
            except Exception as exc:
                print(f"[Downloader] Warning: Failed Indic '{ds_name}' ({lang}): {exc}")


def stream_http_fallback(urls: List[str], limit: int) -> Iterator[str]:
    """Fallback direct HTTP downloader for raw text files when `datasets` is unavailable."""
    count = 0
    for url in urls:
        try:
            print(f"[Downloader] HTTP fallback downloading: {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "NorthhDhruva/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw_text = resp.read().decode("utf-8", errors="ignore")
            for chunk in raw_text.split("\n\n"):
                chunk = chunk.strip()
                if chunk:
                    yield chunk
                    count += 1
                    if count >= limit:
                        return
        except Exception as exc:
            print(f"[Downloader] HTTP fallback failed for '{url}': {exc}")


def _resolve_stream(stream_fn, fallback_urls: List[str], limit: int) -> Iterator[str]:
    """Uses Hugging Face `datasets` if installed, else falls back to direct HTTP download."""
    try:
        import datasets  # noqa: F401

        yield from stream_fn()
    except ImportError:
        print("[Downloader] `datasets` package not installed. Falling back to direct HTTP download.")
        yield from stream_http_fallback(fallback_urls, limit)
    except Exception as exc:
        print(f"[Downloader] HF streaming error ({exc}). Falling back to direct HTTP download.")
        yield from stream_http_fallback(fallback_urls, limit)


def download_pilot_corpus(
    output_path: str = "data/raw/pilot_corpus.txt",
    en_limit: int = 50_000,
    indic_limit: int = 50_000,
    en_dataset: str = "HuggingFaceFW/fineweb-edu",
    indic_dataset: str = "wikimedia/wikipedia",
    indic_langs: Optional[List[str]] = None,
    target_mb: Optional[float] = None,
) -> Dict[str, float]:
    """Downloads, cleans, deduplicates, and writes the English + Indic pilot corpus."""
    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    seen_hashes: Set[str] = set()
    en_kept = 0
    indic_kept = 0
    total_chars = 0
    total_bytes = 0
    target_bytes = int(target_mb * 1024 * 1024) if target_mb and target_mb > 0 else None

    # When target_mb is specified without explicit doc limits, raise stream cap so byte target drives stop condition
    effective_en_limit = max(en_limit, 1_000_000) if target_bytes else en_limit
    effective_indic_limit = max(indic_limit, 1_000_000) if target_bytes else indic_limit
    en_byte_budget = (target_bytes // 2) if target_bytes else None

    with open(output_path, "w", encoding="utf-8") as out_f:
        first_doc = True

        def write_cleaned(raw_text: str) -> bool:
            nonlocal first_doc, total_chars, total_bytes
            cleaned = clean_document(raw_text, seen_hashes=seen_hashes)
            if cleaned is None:
                return False
            payload = (SEPARATOR + cleaned) if not first_doc else cleaned
            out_f.write(payload)
            first_doc = False
            total_chars += len(cleaned)
            total_bytes += len(payload.encode("utf-8"))
            return True

        # 1. Stream & clean English documents
        if effective_en_limit > 0:
            en_stream = _resolve_stream(
                lambda: stream_english_hf(limit=effective_en_limit, dataset_name=en_dataset),
                HTTP_FALLBACK_SOURCES["en"],
                effective_en_limit,
            )
            for doc in en_stream:
                if write_cleaned(doc):
                    en_kept += 1
                    if en_kept % 5000 == 0:
                        print(
                            f"[Downloader] English kept: {en_kept:,} "
                            f"({total_bytes / (1024 * 1024):.1f} MB)"
                        )
                if en_kept >= effective_en_limit or (en_byte_budget and total_bytes >= en_byte_budget):
                    break

        # 2. Stream & clean Indic documents
        if effective_indic_limit > 0 and (target_bytes is None or total_bytes < target_bytes):
            indic_stream = _resolve_stream(
                lambda: stream_indic_hf(
                    limit=effective_indic_limit,
                    dataset_name=indic_dataset,
                    langs=indic_langs,
                ),
                HTTP_FALLBACK_SOURCES["indic"],
                effective_indic_limit,
            )
            for doc in indic_stream:
                if write_cleaned(doc):
                    indic_kept += 1
                    if indic_kept % 5000 == 0:
                        print(
                            f"[Downloader] Indic kept: {indic_kept:,} "
                            f"({total_bytes / (1024 * 1024):.1f} MB)"
                        )
                if indic_kept >= effective_indic_limit or (target_bytes and total_bytes >= target_bytes):
                    break

    total_docs = en_kept + indic_kept
    raw_size_mb = os.path.getsize(output_path) / (1024 * 1024) if os.path.isfile(output_path) else 0.0

    print("=" * 70)
    print("  PILOT CORPUS DOWNLOAD & CLEANING SUMMARY")
    print("=" * 70)
    print(f"  Output Path:         {output_path}")
    print(f"  English Documents:   {en_kept:,}")
    print(f"  Indic Documents:     {indic_kept:,}")
    print(f"  Total Unique Docs:   {total_docs:,}")
    print(f"  Total Characters:    {total_chars:,}")
    print(f"  Raw File Size:       {raw_size_mb:.2f} MB")
    print("=" * 70)

    return {
        "english_docs": en_kept,
        "indic_docs": indic_kept,
        "total_docs": total_docs,
        "total_chars": total_chars,
        "raw_size_mb": raw_size_mb,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download balanced English + Indic pilot corpus")
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/raw/pilot_corpus.txt",
        help="Destination file for cleaned pilot corpus",
    )
    parser.add_argument(
        "--en_limit",
        type=int,
        default=50_000,
        help="Maximum English documents to sample",
    )
    parser.add_argument(
        "--indic_limit",
        type=int,
        default=50_000,
        help="Maximum Indic documents to sample",
    )
    parser.add_argument(
        "--en_dataset",
        type=str,
        default="HuggingFaceFW/fineweb-edu",
        help="Hugging Face English dataset name (fallback: wikitext-103-raw-v1)",
    )
    parser.add_argument(
        "--indic_dataset",
        type=str,
        default="wikimedia/wikipedia",
        help="Hugging Face Indic dataset name (e.g. ai4bharat/sangraha, wikimedia/wikipedia)",
    )
    parser.add_argument(
        "--indic_langs",
        nargs="+",
        default=["hi", "mr", "ta"],
        help="Indic language codes to sample",
    )
    parser.add_argument(
        "--target_mb",
        type=float,
        default=None,
        help="Target corpus size in MB (streams until reached, e.g. 420 for ~100M tokens)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_pilot_corpus(
        output_path=args.output_path,
        en_limit=args.en_limit,
        indic_limit=args.indic_limit,
        en_dataset=args.en_dataset,
        indic_dataset=args.indic_dataset,
        indic_langs=args.indic_langs,
        target_mb=args.target_mb,
    )


if __name__ == "__main__":
    main()

