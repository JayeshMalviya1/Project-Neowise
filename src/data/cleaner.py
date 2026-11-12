"""Fast, robust text cleaner and deduplicator for Northh Dhruva data ingestion."""

import hashlib
import re
from typing import Iterable, List, Optional, Set

from src.tokenizer.normalizer import normalize_text

HTML_TAG_PATTERN = re.compile(r"<[^<]+?>")


def compute_doc_hash(text: str) -> str:
    """Computes a 64-bit / SHA-256 hash string for document deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class DocumentDeduplicator:
    """In-stream document deduplicator using SHA-256 hashes."""

    def __init__(self) -> None:
        self.seen_hashes: Set[str] = set()

    def is_duplicate(self, text: str) -> bool:
        """Returns True if the document has been seen before, otherwise registers it."""
        doc_hash = compute_doc_hash(text)
        if doc_hash in self.seen_hashes:
            return True
        self.seen_hashes.add(doc_hash)
        return False

    def clear(self) -> None:
        self.seen_hashes.clear()


def clean_document(text: str, seen_hashes: Optional[Set[str]] = None) -> Optional[str]:
    """Clean, normalize, and quality-filter an incoming raw document.

    Pipeline:
    1. Apply Unicode NFC normalization & invisible char stripping via normalize_text.
    2. Strip residual HTML tags (<[^<]+?>).
    3. Normalize whitespace while preserving valid paragraph linebreaks.
    4. Quality filter: Discard documents with fewer than 30 characters or 8 words.
    5. Deduplication: Compute 64-bit hash (SHA-256) to eliminate duplicate documents if seen_hashes is provided.

    Args:
        text: Raw document text string.
        seen_hashes: Optional set of seen document hashes for in-stream deduplication.

    Returns:
        Cleaned text string, or None if the document failed quality or deduplication filters.
    """
    if not text or not isinstance(text, str):
        return None

    # 1. Unicode NFC normalization & invisible char stripping
    text = normalize_text(text)

    # 2. Strip residual HTML tags (<[^<]+?>)
    text = HTML_TAG_PATTERN.sub("", text)

    # 3. Normalize whitespace while preserving valid paragraph linebreaks
    raw_lines = text.split("\n")
    cleaned_lines: List[str] = []
    prev_was_empty = False

    for line in raw_lines:
        collapsed = re.sub(r"[ \t]+", " ", line).strip()
        if collapsed:
            cleaned_lines.append(collapsed)
            prev_was_empty = False
        elif not prev_was_empty:
            cleaned_lines.append("")
            prev_was_empty = True

    text = "\n".join(cleaned_lines).strip()

    # 4. Quality filter: Discard documents with fewer than 30 characters or 8 words
    words = text.split()
    if len(text) < 30 or len(words) < 8:
        return None

    # 5. Deduplication filter using SHA-256 hash
    if seen_hashes is not None:
        doc_hash = compute_doc_hash(text)
        if doc_hash in seen_hashes:
            return None
        seen_hashes.add(doc_hash)

    return text


def clean_and_deduplicate_corpus(texts: Iterable[str]) -> List[str]:
    """Cleans an iterable of documents and deduplicates them in-stream.

    Args:
        texts: Iterable of raw document strings.

    Returns:
        List of cleaned, unique, high-quality documents.
    """
    seen_hashes: Set[str] = set()
    cleaned_docs: List[str] = []
    for text in texts:
        cleaned = clean_document(text, seen_hashes=seen_hashes)
        if cleaned is not None:
            cleaned_docs.append(cleaned)
    return cleaned_docs
