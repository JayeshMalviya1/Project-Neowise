"""Production binary dataset sharding pipeline for Northh Dhruva."""

import json
import os
from typing import Any, Dict, List
import numpy as np


def create_shards_from_texts(
    texts: List[str],
    tokenizer: Any,
    output_dir: str,
    shard_size_tokens: int = 250_000,
    prefix: str = "train",
) -> List[str]:
    """Encodes documents, appends <eos>, and writes fixed-size binary uint16 shards.

    Args:
        texts: List of input document strings.
        tokenizer: Tokenizer instance with an encode() method.
        output_dir: Destination directory for shard files and manifest.json.
        shard_size_tokens: Number of uint16 tokens per binary shard file.
        prefix: Filename prefix for shard files (e.g. 'train' or 'val').

    Returns:
        List of created shard file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    buffer: List[int] = []
    shard_paths: List[str] = []
    shard_idx: int = 0
    total_tokens: int = 0
    doc_count: int = 0

    eos_token_id = getattr(tokenizer, "eos_token_id", 2)

    for text in texts:
        if not text:
            continue

        # Encode document and append <eos> (token ID 2)
        try:
            tokens = tokenizer.encode(text, add_eos=True)
        except TypeError:
            tokens = tokenizer.encode(text)

        if not tokens or tokens[-1] != eos_token_id:
            tokens.append(eos_token_id)

        buffer.extend(tokens)
        total_tokens += len(tokens)
        doc_count += 1

        while len(buffer) >= shard_size_tokens:
            shard_chunk = buffer[:shard_size_tokens]
            buffer = buffer[shard_size_tokens:]

            shard_filename = f"{prefix}_{shard_idx:05d}.bin"
            shard_path = os.path.join(output_dir, shard_filename)
            arr = np.array(shard_chunk, dtype=np.uint16)
            arr.tofile(shard_path)

            shard_paths.append(shard_path)
            shard_idx += 1

    # Flush remaining tokens to final shard
    if buffer:
        shard_filename = f"{prefix}_{shard_idx:05d}.bin"
        shard_path = os.path.join(output_dir, shard_filename)
        arr = np.array(buffer, dtype=np.uint16)
        arr.tofile(shard_path)
        shard_paths.append(shard_path)
        shard_idx += 1

    # Write manifest.json recording total tokens, document count, and shard file paths
    manifest: Dict[str, Any] = {
        "prefix": prefix,
        "total_tokens": total_tokens,
        "document_count": doc_count,
        "shard_count": len(shard_paths),
        "shard_size_tokens": shard_size_tokens,
        "shard_paths": shard_paths,
        "shards": shard_paths,
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return shard_paths
