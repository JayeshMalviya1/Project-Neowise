#!/usr/bin/env bash
set -euo pipefail

# Northh Dhruva (Project BHARAT-LLM) Local Data Prep & Colab Tesla T4 Handoff
# Performs CPU-intensive ingestion, BBPE training, and binary sharding locally on M1 Pro,
# then packages compiled artifacts into dhruva_ready_data.zip for high-speed cloud GPU training.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -x "${PROJECT_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
elif [[ -x "${PROJECT_DIR}/../.venv/bin/python" ]]; then
    PYTHON_BIN="${PROJECT_DIR}/../.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    PYTHON_BIN="$(command -v python)"
fi

cd "${PROJECT_DIR}"

RAW_CORPUS="data/raw/pilot_corpus.txt"
TOK_SAMPLE="data/raw/tok_sample.txt"
MIN_BYTES=$((400 * 1024 * 1024)) # 400 MB threshold
SAMPLE_BYTES=41943040            # 40 MB slice for fast representative tokenizer training

echo "======================================================================"
echo "  PROJECT BHARAT-LLM: LOCAL DATA PREP & COLAB T4 PACKAGING PIPELINE"
echo "  Project Dir: ${PROJECT_DIR}"
echo "  Python Bin:  ${PYTHON_BIN}"
echo "======================================================================"

# Step 1: Check existing pilot corpus (>400 MB) or download ~420 MB
NEED_DOWNLOAD=true
if [[ -f "${RAW_CORPUS}" ]]; then
    FILE_SIZE=$(wc -c < "${RAW_CORPUS}" | tr -d '[:space:]')
    if [[ ${FILE_SIZE} -ge ${MIN_BYTES} ]]; then
        echo "[1/4] Found existing pilot corpus: ${RAW_CORPUS} (${FILE_SIZE} bytes >= 400 MB). Skipping download."
        NEED_DOWNLOAD=false
    else
        echo "[1/4] Existing pilot corpus is under 400 MB (${FILE_SIZE} bytes). Re-downloading..."
    fi
else
    echo "[1/4] Pilot corpus not found at ${RAW_CORPUS}. Starting download (~420 MB)..."
fi

if [[ "${NEED_DOWNLOAD}" = true ]]; then
    mkdir -p data/raw
    "${PYTHON_BIN}" scripts/download_pilot_data.py \
        --output_path "${RAW_CORPUS}" \
        --target_mb 420
fi

# Step 2: Extract 40 MB slice and train 16,384-vocab ByteBPETokenizer
echo "[2/4] Extracting 40 MB slice and training 16,384-vocab ByteBPETokenizer..."
mkdir -p data/raw data/tokenizer_16k
head -c "${SAMPLE_BYTES}" "${RAW_CORPUS}" > "${TOK_SAMPLE}"

"${PYTHON_BIN}" scripts/train_tokenizer_production.py \
    --corpus_path "${TOK_SAMPLE}" \
    --output_dir data/tokenizer_16k \
    --vocab_size 16384

# Step 3: Encode full corpus into uint16 binary shards (1M tokens/shard)
echo "[3/4] Encoding corpus into uint16 binary shards (1M tokens/shard)..."
"${PYTHON_BIN}" scripts/prepare_data.py \
    --corpus_path "${RAW_CORPUS}" \
    --tokenizer_dir data/tokenizer_16k \
    --shards_dir data/shards \
    --shard_size 1000000

# Step 4: Package artifacts into dhruva_ready_data.zip
echo "[4/4] Packaging tokenizer, shards, and T4 config into dhruva_ready_data.zip..."
rm -f dhruva_ready_data.zip
zip -r dhruva_ready_data.zip data/tokenizer_16k data/shards configs/train_t4_100m.yaml

ARCHIVE_SIZE=$(ls -lh dhruva_ready_data.zip | awk '{print $5}')

echo "======================================================================"
echo "  DATA PREPARATION & PACKAGING COMPLETE"
echo "  Archive: ${PROJECT_DIR}/dhruva_ready_data.zip (${ARCHIVE_SIZE})"
echo "======================================================================"
echo ""
echo "Google Colab T4 Launch Instructions:"
echo "----------------------------------------------------------------------"
echo "1. In Google Colab, select: Runtime > Change runtime type > T4 GPU."
echo "2. Upload 'dhruva_ready_data.zip' to the Colab environment."
echo "3. Run the following cells in Colab:"
echo ""
echo "   # Cell 1: Clone repo and unpack prepared data"
echo "   !git clone https://github.com/JayeshMalviya1/Project-Neowise.git"
echo "   %cd Project-Neowise"
echo "   !unzip -q /content/dhruva_ready_data.zip -d ."
echo ""
echo "   # Cell 2: Install dependencies"
echo "   !pip install -r requirements.txt"
echo ""
echo "   # Cell 3: Launch 100M-token Tesla T4 pretraining"
echo "   !python scripts/run_local_pretrain.py --config configs/train_t4_100m.yaml"
echo "----------------------------------------------------------------------"

