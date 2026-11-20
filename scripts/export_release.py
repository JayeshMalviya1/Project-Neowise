#!/usr/bin/env python3
"""Model Compression & Release Exporter for Northh Dhruva 500M (Stage 3 Instruct + JSON).

Loads raw .pt checkpoint (`dhruva_500m_instruct_json.pt`), extracts config and weights,
and builds a self-contained release package at `release/dhruva-500m-instruct-json/`:
1. `config.json`: Full model hyperparameters + architecture & dtype metadata.
2. `model.safetensors`: Contiguous BF16 weights (~940 MB - 978 MB) with cloned tied tensors.
3. `dhruva_500m_int8.pt`: Dynamic INT8 quantized checkpoint (~475 MB) for fast CPU inference.
4. Tokenizer files (`vocab.json`, `merges.json`) copied from `data/tokenizer_32k/`.
"""

import argparse
import gc
import json
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import torch
from safetensors.torch import save_file

# Cap CPU threads to 4 P-cores on laptops to prevent AVX2 thermal/power trip reboots
_default_threads = int(os.environ.get("DHRUVA_NUM_THREADS", "4"))
if torch.get_num_threads() > _default_threads:
    torch.set_num_threads(_default_threads)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.model_config import DhruvaConfig
from src.models.dhruva_lm import DhruvaForCausalLM


def resolve_checkpoint_path(ckpt_arg: Optional[str]) -> Path:
    """Finds `dhruva_500m_raft_final.pt` (preferred) or `dhruva_500m_instruct_json.pt` across standard project locations."""
    candidates = []
    if ckpt_arg:
        candidates.extend([
            Path(ckpt_arg),
            PROJECT_ROOT / ckpt_arg,
            PROJECT_ROOT.parent / ckpt_arg,
        ])

    candidates.extend([
        PROJECT_ROOT / "dhruva_500m_raft_final.pt",
        PROJECT_ROOT / "checkpoints" / "dhruva_500m_raft_final.pt",
        PROJECT_ROOT / "instruct_500m" / "dhruva_500m_raft_final.pt",
        PROJECT_ROOT.parent / "instruct_500m" / "dhruva_500m_raft_final.pt",
        PROJECT_ROOT.parent / "dhruva_500m_raft_final.pt",
        PROJECT_ROOT / "checkpoints" / "dhruva_500m_instruct_json.pt",
        PROJECT_ROOT / "dhruva_500m_instruct_json.pt",
        PROJECT_ROOT / "instruct_500m" / "dhruva_500m_instruct_json.pt",
        PROJECT_ROOT.parent / "instruct_500m" / "dhruva_500m_instruct_json.pt",
        PROJECT_ROOT.parent / "dhruva_500m_instruct_json.pt",
        PROJECT_ROOT.parent / "instruct_500m-20261117T050325Z-1-003" / "instruct_500m" / "dhruva_500m_instruct_final.pt",
    ])

    for cand in candidates:
        if cand.is_file():
            return cand.resolve()

    raise FileNotFoundError(
        f"Could not find checkpoint '{ckpt_arg or 'dhruva_500m_raft_final.pt / dhruva_500m_instruct_json.pt'}'. "
        f"Searched: {[str(c) for c in candidates[:8]]}"
    )


def resolve_tokenizer_dir(tok_arg: str) -> Path:
    """Resolves the source 32K tokenizer directory."""
    for cand in (Path(tok_arg), PROJECT_ROOT / tok_arg, PROJECT_ROOT / "data" / "tokenizer_32k"):
        if cand.is_dir() and (cand / "vocab.json").is_file():
            return cand.resolve()
    raise FileNotFoundError(f"Tokenizer directory not found: {tok_arg}")


def format_mb(size_bytes: int) -> str:
    """Formats byte count in both decimal MB (10^6) and binary MiB (2^20)."""
    mb_dec = size_bytes / 1_000_000
    mib = size_bytes / (1024 * 1024)
    return f"{mib:.2f} MiB ({mb_dec:.2f} MB)"


def export_release(
    checkpoint_path: Path,
    tokenizer_dir: Path,
    output_dir: Path,
) -> Dict[str, int]:
    """Exports self-contained release directory with BF16 safetensors, INT8 checkpoint, config, and tokenizer."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_size_bytes = checkpoint_path.stat().st_size

    print("=" * 76)
    print("  🚀 NORTHH DHRUVA 500M — STAGE 3 RELEASE EXPORTER")
    print("=" * 76)
    print(f"  Source Checkpoint: {checkpoint_path}")
    print(f"  Raw Size:          {format_mb(raw_size_bytes)}")
    print(f"  Target Release:    {output_dir}")
    print("-" * 76)

    # 1. Load Raw Checkpoint
    print("[1/4] Loading raw PyTorch checkpoint...")
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    config: Optional[DhruvaConfig] = ckpt.get("config", None) if isinstance(ckpt, dict) else None
    raw_sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    cleaned_sd: Dict[str, torch.Tensor] = {}
    for k, v in raw_sd.items():
        clean_k = k.replace("_orig_mod.", "")
        cleaned_sd[clean_k] = v

    if config is None:
        vocab_size, dim = cleaned_sd["tok_embeddings.weight"].shape
        config = DhruvaConfig(
            vocab_size=vocab_size,
            dim=dim,
            n_layers=24,
            n_heads=16,
            n_kv_heads=4,
            intermediate_size=3584,
            max_seq_len=2048,
            rope_theta=500000.0,
            rms_norm_eps=1e-5,
            tie_word_embeddings=True,
        )

    # 2. Export config.json
    print("[2/4] Exporting config.json...")
    config_dict = asdict(config)
    config_payload = {
        "architectures": ["DhruvaForCausalLM"],
        "model_type": "dhruva",
        "torch_dtype": "bfloat16",
        "head_dim": config.head_dim,
        "hidden_act": "swiglu",
        "position_embedding_type": "rope",
        **config_dict,
    }
    config_json_path = output_dir / "config.json"
    with open(config_json_path, "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)

    # 3. Export BF16 Safetensors (cloning shared/tied tensors)
    print("[3/4] Exporting BF16 Safetensors (model.safetensors)...")
    bf16_sd: Dict[str, torch.Tensor] = {}
    seen_data_ptrs = set()

    for key, tensor in cleaned_sd.items():
        ptr = tensor.data_ptr()
        if ptr in seen_data_ptrs or key == "lm_head.weight":
            t_bf16 = tensor.to(dtype=torch.bfloat16).clone().contiguous()
        else:
            seen_data_ptrs.add(ptr)
            t_bf16 = tensor.to(dtype=torch.bfloat16).contiguous()
            if t_bf16.data_ptr() == ptr and key != "tok_embeddings.weight":
                t_bf16 = t_bf16.clone().contiguous()
        bf16_sd[key] = t_bf16

    # Ensure tied weights do not share storage pointers when passed to save_file
    if "tok_embeddings.weight" in bf16_sd and "lm_head.weight" in bf16_sd:
        if bf16_sd["tok_embeddings.weight"].data_ptr() == bf16_sd["lm_head.weight"].data_ptr():
            bf16_sd["lm_head.weight"] = bf16_sd["lm_head.weight"].clone().contiguous()

    safetensors_path = output_dir / "model.safetensors"
    save_file(bf16_sd, str(safetensors_path))
    safetensors_size_bytes = safetensors_path.stat().st_size
    del bf16_sd

    # 4. Export INT8 Dynamic Quantized Checkpoint (dhruva_500m_int8.pt)
    print("[4/4] Applying PyTorch Dynamic INT8 Quantization (dhruva_500m_int8.pt)...")
    fp32_model = DhruvaForCausalLM(config, init_weights=False)
    fp32_sd = {k: v.to(dtype=torch.float32) for k, v in cleaned_sd.items()}
    fp32_model.load_state_dict(fp32_sd, strict=False)
    fp32_model.eval()
    del fp32_sd
    del cleaned_sd
    del ckpt
    gc.collect()

    # Ensure a valid quantization engine is active
    supported_engines = [e for e in torch.backends.quantized.supported_engines if e != "none"]
    if torch.backends.quantized.engine == "none" and supported_engines:
        torch.backends.quantized.engine = "fbgemm" if "fbgemm" in supported_engines else supported_engines[0]

    int8_model = torch.ao.quantization.quantize_dynamic(
        fp32_model, {torch.nn.Linear}, dtype=torch.qint8
    )
    del fp32_model

    int8_sd = int8_model.state_dict()
    # Since tie_word_embeddings=True, `lm_head` already stores the qint8 quantized weights of
    # `tok_embeddings.weight`. Omitting the duplicate FP32 `tok_embeddings.weight` saves ~168 MB
    # on disk (bringing the checkpoint to ~471 MB) and is reconstructed on load via dequantize().
    if getattr(config, "tie_word_embeddings", True) and "lm_head._packed_params._packed_params" in int8_sd:
        int8_sd.pop("tok_embeddings.weight", None)

    int8_path = output_dir / "dhruva_500m_int8.pt"
    torch.save(
        {
            "config": config_payload,
            "quantized": True,
            "quantization": "dynamic_int8_linear",
            "quantized_engine": torch.backends.quantized.engine,
            "model_state_dict": int8_sd,
        },
        str(int8_path),
    )
    int8_size_bytes = int8_path.stat().st_size

    # 5. Package Tokenizer Files
    copied_tok_files = []
    for item in tokenizer_dir.iterdir():
        if item.is_file():
            dest = output_dir / item.name
            shutil.copy2(str(item), str(dest))
            copied_tok_files.append(item.name)

    # Also mirror release dir at workspace root if run from outside Project-Neowise
    cwd_release = Path.cwd() / "release" / "dhruva-500m-instruct-json"
    if cwd_release.resolve() != output_dir.resolve() and Path.cwd().resolve() == PROJECT_ROOT.parent.resolve():
        # Create a lightweight pointer or copy only if explicitly requested
        pass

    print("-" * 76)
    print("  📦 RELEASE EXPORT COMPLETE — SIZE SUMMARY")
    print("-" * 76)
    print(f"  Raw Checkpoint (.pt):         {format_mb(raw_size_bytes)}")
    print(f"  BF16 Safetensors:             {format_mb(safetensors_size_bytes)}")
    print(f"  INT8 Quantized (.pt):         {format_mb(int8_size_bytes)}")
    reduction_pct = (1.0 - (int8_size_bytes / raw_size_bytes)) * 100.0 if raw_size_bytes > 0 else 0.0
    print(f"  INT8 Compression Ratio:       {reduction_pct:.1f}% size reduction")
    print(f"  Config File:                  {config_json_path.name}")
    print(f"  Packaged Tokenizer Files:     {', '.join(sorted(copied_tok_files))}")
    print("=" * 76)

    return {
        "raw_bytes": raw_size_bytes,
        "safetensors_bytes": safetensors_size_bytes,
        "int8_bytes": int8_size_bytes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Northh Dhruva 500M Stage 3 checkpoint to BF16 Safetensors & INT8 release package"
    )
    parser.add_argument(
        "--ckpt",
        "--checkpoint",
        dest="ckpt",
        type=str,
        default=None,
        help="Path to raw checkpoint (.pt). Defaults to dhruva_500m_raft_final.pt if present, else dhruva_500m_instruct_json.pt",
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default="data/tokenizer_32k",
        help="Path to 32K ByteBPETokenizer directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="release/dhruva-500m-instruct-json",
        help="Target release directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = resolve_checkpoint_path(args.ckpt)
    tok_dir = resolve_tokenizer_dir(args.tokenizer_dir)

    out_path = Path(args.output_dir)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path

    export_release(ckpt_path, tok_dir, out_path)


if __name__ == "__main__":
    main()
