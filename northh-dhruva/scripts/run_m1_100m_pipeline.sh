#!/usr/bin/env bash
set -euo pipefail

# One-command 100M-token Apple Silicon (M1 Pro MPS) data ingestion & pretraining pipeline
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"

if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    PYTHON_BIN="$(command -v python)"
fi

cd "${PROJECT_DIR}"

echo "======================================================================"
echo "  PROJECT BHARAT-LLM: NORTHH DHRUVA M1 PRO 100M-TOKEN MPS PIPELINE"
echo "  Project Dir: ${PROJECT_DIR}"
echo "  Python Bin:  ${PYTHON_BIN}"
echo "======================================================================"

echo "[1/4] Downloading ~420 MB of cleaned English + Indic text..."
"${PYTHON_BIN}" scripts/download_pilot_data.py \
    --output_path data/raw/pilot_corpus.txt \
    --target_mb 420

echo "[2/4] Training 16,384-vocab ByteBPETokenizer..."
"${PYTHON_BIN}" scripts/train_tokenizer_production.py \
    --corpus_path data/raw/pilot_corpus.txt \
    --output_dir data/tokenizer_16k \
    --vocab_size 16384

echo "[3/4] Building uint16 memory-mapped binary shards (1M tokens/shard)..."
"${PYTHON_BIN}" scripts/prepare_data.py \
    --corpus_path data/raw/pilot_corpus.txt \
    --tokenizer_dir data/tokenizer_16k \
    --shards_dir data/shards \
    --shard_size 1000000

echo "[4/4] Launching M1 Pro MPS pretraining (100M tokens)..."
"${PYTHON_BIN}" scripts/run_local_pretrain.py \
    --config configs/train_m1_pro_100m.yaml

echo "======================================================================"
echo "  PIPELINE COMPLETE"
echo "======================================================================"

