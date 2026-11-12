"""Northh Dhruva: Custom Decoder-Only Foundation Model Engine (Project BHARAT-LLM)."""

from src.models import (
    RMSNorm,
    RotaryEmbedding,
    apply_rotary_pos_emb,
    rotate_half,
    GroupedQueryAttention,
    SwiGLUMLP,
    TransformerBlock,
    DhruvaForCausalLM,
)
from src.tokenizer import normalize_text, ByteBPETokenizer
from src.data import (
    build_binary_shard,
    MemmapPretrainingDataset,
    clean_document,
    clean_and_deduplicate_corpus,
    compute_doc_hash,
    create_shards_from_texts,
)
from src.training import DhruvaTrainer
from src.evaluation import (
    compute_validation_perplexity,
    sample_tokens,
    generate_text,
)

__all__ = [
    "RMSNorm",
    "RotaryEmbedding",
    "apply_rotary_pos_emb",
    "rotate_half",
    "GroupedQueryAttention",
    "SwiGLUMLP",
    "TransformerBlock",
    "DhruvaForCausalLM",
    "normalize_text",
    "ByteBPETokenizer",
    "build_binary_shard",
    "MemmapPretrainingDataset",
    "clean_document",
    "clean_and_deduplicate_corpus",
    "compute_doc_hash",
    "create_shards_from_texts",
    "DhruvaTrainer",
    "compute_validation_perplexity",
    "sample_tokens",
    "generate_text",
]
