import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Union

from src.tokenizer.normalizer import normalize_text

SPLIT_PATTERN = re.compile(r"\s+\S+|\S+|\s+")


class VocabDict(dict):
    """Dictionary supporting both bytes and latin-1 string / special token lookups."""

    def __getitem__(self, key: Union[bytes, str]) -> int:
        if isinstance(key, str) and not super().__contains__(key):
            byte_key = key.encode("latin-1")
            if super().__contains__(byte_key):
                return super().__getitem__(byte_key)
        return super().__getitem__(key)

    def get(self, key: Union[bytes, str], default: Optional[int] = None) -> Optional[int]:
        if isinstance(key, str) and not super().__contains__(key):
            byte_key = key.encode("latin-1")
            if super().__contains__(byte_key):
                return super().get(byte_key, default)
        return super().get(key, default)

    def __contains__(self, key: object) -> bool:
        if super().__contains__(key):
            return True
        if isinstance(key, str):
            return super().__contains__(key.encode("latin-1"))
        return False


class ByteBPETokenizer:
    """Byte-Level Byte-Pair Encoding (BPE) Tokenizer for Northh Dhruva (Project BHARAT-LLM).

    Features:
    - Base vocabulary of 260 tokens: 4 special tokens (<pad>, <bos>, <eos>, <unk>)
      and 256 single-byte tokens (bytes([0]) .. bytes([255])).
    - Zero out-of-vocabulary (<unk>) fallback for arbitrary multilingual UTF-8 text.
    - Lossless round-trip text reconstruction preserving whitespace, newlines, and punctuation.
    - Fast BPE training and rank-based greedy merging with word-level caching.
    - Latin-1 serialization for transparent JSON storage of raw byte vocabulary and merge pairs.
    """

    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3

    def __init__(self) -> None:
        self.pad_token_id: int = self.PAD_ID
        self.bos_token_id: int = self.BOS_ID
        self.eos_token_id: int = self.EOS_ID
        self.unk_token_id: int = self.UNK_ID

        self.vocab: VocabDict = VocabDict()
        self.id_to_token: Dict[int, bytes] = {}
        self.merges: Dict[Tuple[bytes, bytes], int] = {}
        self.cache: Dict[bytes, List[bytes]] = {}

        self._init_base_vocab()

    def _init_base_vocab(self) -> None:
        """Initializes the base 260 vocabulary: 4 special tokens + 256 raw bytes."""
        self.vocab = VocabDict()
        self.id_to_token = {}
        self.merges = {}
        self.cache = {}

        # 4 special tokens
        special_tokens = [
            (b"<pad>", self.PAD_ID),
            (b"<bos>", self.BOS_ID),
            (b"<eos>", self.EOS_ID),
            (b"<unk>", self.UNK_ID),
        ]
        for tok_bytes, tid in special_tokens:
            self.vocab[tok_bytes] = tid
            self.id_to_token[tid] = tok_bytes

        # 256 single-byte values (IDs 4..259)
        for b in range(256):
            byte_val = bytes([b])
            tid = 4 + b
            self.vocab[byte_val] = tid
            self.id_to_token[tid] = byte_val

    @property
    def vocab_size(self) -> int:
        """Current vocabulary size."""
        return len(self.vocab)

    def train(self, text: str, target_vocab_size: int, verbose: bool = False) -> None:
        """Train Byte-Level BPE merges up to target_vocab_size.

        Args:
            text: Training corpus string.
            target_vocab_size: Desired total vocabulary size (>= 260).
            verbose: If True, prints merge progress.
        """
        self._init_base_vocab()

        if target_vocab_size <= self.vocab_size:
            return

        normalized = normalize_text(text)
        words = SPLIT_PATTERN.findall(normalized)
        if not words:
            return

        # Pre-segment each word into tuples of single bytes and count word frequencies
        word_counts: Dict[Tuple[bytes, ...], int] = Counter(
            tuple(bytes([b]) for b in w.encode("utf-8")) for w in words
        )

        num_merges_to_do = target_vocab_size - self.vocab_size
        if verbose:
            print(f"[Tokenizer] Starting BPE training: target={target_vocab_size}, merges={num_merges_to_do}")

        for merge_idx in range(num_merges_to_do):
            # Count frequencies of all adjacent pairs across word segments
            pair_counts: Dict[Tuple[bytes, bytes], int] = defaultdict(int)
            for w_tuple, count in word_counts.items():
                for i in range(len(w_tuple) - 1):
                    pair_counts[(w_tuple[i], w_tuple[i + 1])] += count

            if not pair_counts:
                if verbose:
                    print(f"[Tokenizer] No more pairs to merge at vocab size {self.vocab_size}.")
                break

            best_pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[best_pair] <= 0:
                break

            new_token = best_pair[0] + best_pair[1]
            new_id = len(self.vocab)
            self.vocab[new_token] = new_id
            self.id_to_token[new_id] = new_token
            self.merges[best_pair] = len(self.merges)

            # Update word counts by replacing best_pair in-place
            p0, p1 = best_pair
            new_word_counts: Dict[Tuple[bytes, ...], int] = {}
            for w_tuple, count in word_counts.items():
                if p0 in w_tuple:
                    new_pieces: List[bytes] = []
                    i = 0
                    n = len(w_tuple)
                    while i < n:
                        if i < n - 1 and w_tuple[i] == p0 and w_tuple[i + 1] == p1:
                            new_pieces.append(new_token)
                            i += 2
                        else:
                            new_pieces.append(w_tuple[i])
                            i += 1
                    new_word_counts[tuple(new_pieces)] = count
                else:
                    new_word_counts[w_tuple] = count
            word_counts = new_word_counts

            if verbose and (merge_idx + 1) % 50 == 0:
                print(f"[Tokenizer] Merge {merge_idx + 1}/{num_merges_to_do}: '{new_token!r}' -> ID {new_id}")

        self.cache = {}
        if verbose:
            print(f"[Tokenizer] Completed training. Final vocab_size: {self.vocab_size}")

    def _merge_word(self, word_bytes: bytes) -> List[bytes]:
        """Apply learned BPE merges greedily by merge rank to a byte sequence."""
        if word_bytes in self.cache:
            return self.cache[word_bytes]

        pieces: List[bytes] = [bytes([b]) for b in word_bytes]
        while len(pieces) > 1:
            # Find the adjacent pair with the lowest merge rank
            min_rank = float("inf")
            best_pair: Optional[Tuple[bytes, bytes]] = None
            for i in range(len(pieces) - 1):
                pair = (pieces[i], pieces[i + 1])
                rank = self.merges.get(pair)
                if rank is not None and rank < min_rank:
                    min_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            # Merge all occurrences of best_pair in this word
            p0, p1 = best_pair
            merged_token = p0 + p1
            new_pieces: List[bytes] = []
            i = 0
            n = len(pieces)
            while i < n:
                if i < n - 1 and pieces[i] == p0 and pieces[i + 1] == p1:
                    new_pieces.append(merged_token)
                    i += 2
                else:
                    new_pieces.append(pieces[i])
                    i += 1
            pieces = new_pieces

        self.cache[word_bytes] = pieces
        return pieces

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Encode text to token IDs.

        Args:
            text: Input string.
            add_bos: Whether to prepend the <bos> token ID (1).
            add_eos: Whether to append the <eos> token ID (2).

        Returns:
            List of integer token IDs.
        """
        normalized = normalize_text(text)
        words = SPLIT_PATTERN.findall(normalized)
        token_ids: List[int] = []

        if add_bos:
            token_ids.append(self.BOS_ID)

        for w in words:
            word_bytes = w.encode("utf-8")
            pieces = self._merge_word(word_bytes)
            for p in pieces:
                tid = self.vocab.get(p)
                if tid is None:
                    # In a complete byte tokenizer, every single byte is in vocab (4..259)
                    # If an unmerged piece somehow misses, fall back to individual raw bytes
                    for b in p:
                        token_ids.append(self.vocab.get(bytes([b]), self.UNK_ID))
                else:
                    token_ids.append(tid)

        if add_eos:
            token_ids.append(self.EOS_ID)

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Decode a list of token IDs back into a UTF-8 string.

        Special tokens (<pad>=0, <bos>=1, <eos>=2, <unk>=3) are ignored during reconstruction.

        Args:
            token_ids: List of integer token IDs.

        Returns:
            Decoded Unicode string with lossless whitespace and punctuation.
        """
        byte_chunks: List[bytes] = []
        for tid in token_ids:
            if tid in (self.PAD_ID, self.BOS_ID, self.EOS_ID, self.UNK_ID):
                continue
            tok_bytes = self.id_to_token.get(tid)
            if tok_bytes is not None:
                byte_chunks.append(tok_bytes)

        return b"".join(byte_chunks).decode("utf-8", errors="replace")

    def save(self, directory: str) -> None:
        """Serialize tokenizer vocabulary and merges to directory.

        Raw byte tokens and merge pairs are serialized using latin-1 encoding
        to ensure 100% bijective JSON string serialization.

        Args:
            directory: Directory to save vocab.json and merges.json.
        """
        os.makedirs(directory, exist_ok=True)
        vocab_path = os.path.join(directory, "vocab.json")
        merges_path = os.path.join(directory, "merges.json")

        vocab_serialized = {
            (k.decode("latin-1") if isinstance(k, bytes) else str(k)): v
            for k, v in self.vocab.items()
        }
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab_serialized, f, indent=2, ensure_ascii=False)

        sorted_merges = sorted(self.merges.items(), key=lambda x: x[1])
        merges_serialized = [
            [p0.decode("latin-1"), p1.decode("latin-1")]
            for (p0, p1), _ in sorted_merges
        ]
        with open(merges_path, "w", encoding="utf-8") as f:
            json.dump(merges_serialized, f, indent=2, ensure_ascii=False)

    def load(self, directory: str) -> None:
        """Load tokenizer vocabulary and merges from directory.

        Args:
            directory: Directory containing vocab.json and merges.json.
        """
        vocab_path = os.path.join(directory, "vocab.json")
        merges_path = os.path.join(directory, "merges.json")

        with open(vocab_path, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)

        self.vocab = VocabDict()
        self.id_to_token = {}
        for k_str, tid in raw_vocab.items():
            k_bytes = k_str.encode("latin-1")
            self.vocab[k_bytes] = tid
            self.id_to_token[tid] = k_bytes

        with open(merges_path, "r", encoding="utf-8") as f:
            raw_merges = json.load(f)

        self.merges = {}
        if isinstance(raw_merges, list):
            for rank, pair in enumerate(raw_merges):
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    p0 = pair[0].encode("latin-1")
                    p1 = pair[1].encode("latin-1")
                    self.merges[(p0, p1)] = rank
        elif isinstance(raw_merges, dict):
            for k, rank in raw_merges.items():
                if " " in k:
                    parts = k.split(" ", 1)
                    p0 = parts[0].encode("latin-1")
                    p1 = parts[1].encode("latin-1")
                    self.merges[(p0, p1)] = int(rank)

        self.cache = {}

    @classmethod
    def from_pretrained(cls, directory: str) -> "ByteBPETokenizer":
        """Factory method to instantiate and load a tokenizer from disk."""
        tokenizer = cls()
        tokenizer.load(directory)
        return tokenizer

