"""Unified Production Inference Engine for Northh Dhruva 500M (Stage 3 Instruct + JSON).

Features:
- Self-contained loading from `release/dhruva-500m-instruct-json/`.
- Hardware-aware precision: BF16/FP16 `model.safetensors` on CUDA/MPS, INT8/FP32 on CPU.
- Stage 3 ChatML prompt formatting (with optional `Context:` + `Instruction:` blocks).
- Auto-Mode Repetition Penalty: 1.0 when first generated token starts with `{` (JSON mode),
  1.15 otherwise (conversational mode).
- Structured Output Parser returning `ui_component` (`table`, `specs_card`, `json`, or `markdown`).
"""

import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

# Cap default CPU threads to 4 P-cores on laptops to prevent AVX2 thermal/power trip reboots
_default_threads = int(os.environ.get("DHRUVA_NUM_THREADS", "4"))
if torch.get_num_threads() > _default_threads:
    torch.set_num_threads(_default_threads)

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.model_config import DhruvaConfig
from src.models.dhruva_lm import DhruvaForCausalLM
from src.tokenizer.byte_bpe import ByteBPETokenizer


def _select_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


class DhruvaEngine:
    """Production inference engine for Northh Dhruva 500M (Chat + Structured JSON)."""

    def __init__(
        self,
        model_dir: str = "release/dhruva-500m-instruct-json",
        quant: str = "bf16",
        context_window: int = 2048,
        device: Optional[str] = None,
        release_dir: Optional[str] = None,
        prefer_format: Optional[str] = None,
    ) -> None:
        target_dir = release_dir or model_dir
        self.device = device or _select_default_device()
        self.release_dir = self._resolve_release_dir(target_dir)
        self.model_dir = str(self.release_dir)
        self.context_window = context_window
        self.loaded_format: str = "unknown"

        # Normalize quant / prefer_format
        if prefer_format in ("int8", "safetensors"):
            self.quant = "int8" if prefer_format == "int8" else "bf16"
        else:
            self.quant = quant.lower() if quant else "bf16"

        # 1. Load Config
        config_path = self.release_dir / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing config.json in release directory: {self.release_dir}")

        with open(config_path, "r", encoding="utf-8") as f:
            raw_cfg = json.load(f)

        valid_fields = {
            "vocab_size",
            "dim",
            "n_layers",
            "n_heads",
            "n_kv_heads",
            "intermediate_size",
            "max_seq_len",
            "rope_theta",
            "rms_norm_eps",
            "tie_word_embeddings",
            "initializer_range",
            "use_checkpoint",
        }
        filtered_cfg = {k: v for k, v in raw_cfg.items() if k in valid_fields}
        filtered_cfg["max_seq_len"] = context_window
        self.config = DhruvaConfig(**filtered_cfg)

        # 2. Load Tokenizer
        self.tokenizer = ByteBPETokenizer()
        self.tokenizer.load(str(self.release_dir))

        # 3. Load Model Weights (Safetensors for bf16, dhruva_500m_int8.pt for int8)
        chosen_fmt = "int8" if self.quant == "int8" else "safetensors"
        self.model = self._load_model(prefer_format=chosen_fmt)

    @staticmethod
    def _resolve_release_dir(release_dir: str) -> Path:
        candidates = [
            Path(release_dir),
            PROJECT_ROOT / release_dir,
            PROJECT_ROOT.parent / release_dir,
        ]
        for cand in candidates:
            if cand.is_dir() and (cand / "config.json").is_file():
                return cand.resolve()
        raise FileNotFoundError(
            f"Release directory '{release_dir}' not found or incomplete. "
            f"Run `python scripts/export_release.py` first."
        )

    def _load_model(self, prefer_format: Optional[str] = None) -> torch.nn.Module:
        safetensors_path = self.release_dir / "model.safetensors"
        int8_path = self.release_dir / "dhruva_500m_int8.pt"

        # Determine target format: prefer safetensors (BF16/FP16 on GPU/MPS, FP32 with KV-cache on CPU),
        # falling back to dhruva_500m_int8.pt when requested or when safetensors is absent.
        if prefer_format in ("int8", "safetensors"):
            target_fmt = prefer_format
        elif safetensors_path.is_file():
            target_fmt = "safetensors"
        elif int8_path.is_file():
            target_fmt = "int8"
        else:
            raise FileNotFoundError(f"No model weights found in {self.release_dir}")

        if target_fmt == "int8" and self.device == "cpu" and int8_path.is_file():
            supported_engines = [e for e in torch.backends.quantized.supported_engines if e != "none"]
            if torch.backends.quantized.engine == "none" and supported_engines:
                torch.backends.quantized.engine = (
                    "fbgemm" if "fbgemm" in supported_engines else supported_engines[0]
                )

            ckpt = torch.load(str(int8_path), map_location="cpu", weights_only=False, mmap=True)
            if isinstance(ckpt, dict) and "quantized_engine" in ckpt:
                eng = ckpt["quantized_engine"]
                if eng in supported_engines:
                    torch.backends.quantized.engine = eng

            base_model = DhruvaForCausalLM(self.config, init_weights=False)
            base_model.eval()
            q_model = torch.ao.quantization.quantize_dynamic(
                base_model, {torch.nn.Linear}, dtype=torch.qint8
            )
            sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            q_model.load_state_dict(sd, strict=False)

            # Reconstruct tied token embeddings from quantized lm_head if omitted on disk
            if "tok_embeddings.weight" not in sd and hasattr(q_model.lm_head, "weight"):
                q_model.tok_embeddings.weight.data.copy_(q_model.lm_head.weight().dequantize())

            del sd, ckpt
            gc.collect()
            q_model.eval()
            self.loaded_format = "int8"
            return q_model

        # Load from model.safetensors (zero-copy mmap via safetensors)
        model = DhruvaForCausalLM(self.config, init_weights=False)
        sd = load_file(str(safetensors_path), device="cpu")
        model.load_state_dict(sd, strict=False)
        del sd
        gc.collect()
        if self.config.tie_word_embeddings:
            model.lm_head.weight = model.tok_embeddings.weight

        if self.device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            model.to(device=self.device, dtype=dtype)
        elif self.device == "mps":
            model.to(device=self.device, dtype=torch.float16)
        else:
            model.to(device="cpu", dtype=torch.float32)

        model.eval()
        self.loaded_format = "safetensors"
        return model

    @staticmethod
    def format_prompt(
        prompt: str,
        context: Optional[str] = None,
        retrieved_context: Optional[str] = None,
    ) -> str:
        """Formats prompt using the exact Stage 3/4 ChatML template.

        - `retrieved_context`: RAFT Q&A format (`Retrieved Context:\\n...\\n\\nQuestion:\\n...`)
        - `context`: Direct JSON extraction format (`Context:\\n...\\n\\nInstruction:\\n...`)
        - neither: Standalone conversational turn (`### User:\\n{prompt}\\n\\n### Assistant:\\n`)
        """
        clean_prompt = prompt.strip()
        if retrieved_context is not None and str(retrieved_context).strip():
            clean_rc = str(retrieved_context).strip()
            return f"### User:\nRetrieved Context:\n{clean_rc}\n\nQuestion:\n{clean_prompt}\n\n### Assistant:\n"
        if context is not None and str(context).strip():
            clean_context = str(context).strip()
            return f"### User:\nContext:\n{clean_context}\n\nInstruction:\n{clean_prompt}\n\n### Assistant:\n"
        return f"### User:\n{clean_prompt}\n\n### Assistant:\n"

    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor,
        generated_ids: List[int],
        penalty: float,
    ) -> torch.Tensor:
        """Applies Keskar/HuggingFace repetition penalty to already generated tokens."""
        if penalty == 1.0 or not generated_ids:
            return logits
        unique_ids = torch.tensor(list(set(generated_ids)), dtype=torch.long, device=logits.device)
        score = logits[:, unique_ids]
        score = torch.where(score < 0, score * penalty, score / penalty)
        logits = logits.clone()
        logits[:, unique_ids] = score
        return logits

    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        retrieved_context: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.9,
        top_k: int = 40,
    ) -> Dict[str, Any]:
        """Runs Stage 3/4 inference with Auto-Mode Repetition Penalty and Structured Output Parsing."""
        t0 = time.perf_counter()

        formatted_prompt = self.format_prompt(
            prompt=prompt,
            context=context,
            retrieved_context=retrieved_context,
        )
        input_ids = [self.tokenizer.bos_token_id] + self.tokenizer.encode(formatted_prompt)

        exec_device = "cpu" if self.loaded_format == "int8" else self.device
        x = torch.tensor([input_ids], dtype=torch.long, device=exec_device)

        generated_ids: List[int] = []
        repetition_penalty: float = 1.0
        eos_id = self.tokenizer.eos_token_id
        past_key_values = None
        curr_input = x

        with torch.no_grad():
            for step_idx in range(max_new_tokens):
                out = self.model(
                    curr_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                logits = out["logits"][:, -1, :]
                past_key_values = out.get("past_key_values", None)

                # Apply repetition penalty after inspecting the first generated token
                if step_idx > 0 and repetition_penalty != 1.0:
                    logits = self._apply_repetition_penalty(logits, generated_ids, repetition_penalty)

                # Sample or Greedy Argmax
                if temperature <= 0.0 or temperature < 1e-5:
                    next_id = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    scaled_logits = logits / temperature
                    if top_k is not None and top_k > 0:
                        k = min(top_k, scaled_logits.size(-1))
                        v, _ = torch.topk(scaled_logits, k)
                        scaled_logits[scaled_logits < v[:, [-1]]] = -float("Inf")
                    if top_p is not None and 0.0 < top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                        sorted_indices_to_remove[:, 0] = False
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        scaled_logits[:, indices_to_remove] = -float("Inf")
                    probs = F.softmax(scaled_logits, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)

                tok_id = int(next_id.item())
                if tok_id == eos_id:
                    break

                generated_ids.append(tok_id)
                curr_input = next_id

                # Auto-Mode Repetition Penalty detection on the very first token
                if step_idx == 0:
                    first_token_text = self.tokenizer.decode(generated_ids).lstrip()
                    if first_token_text.startswith("{"):
                        repetition_penalty = 1.0
                    else:
                        repetition_penalty = 1.15

        t1 = time.perf_counter()
        latency_ms = round((t1 - t0) * 1000.0, 2)
        raw_text = self.tokenizer.decode(generated_ids).strip()
        token_count = len(generated_ids)

        # Structured Output Parser
        if raw_text.startswith("{"):
            parsed = self._try_parse_json(raw_text)
            if isinstance(parsed, dict):
                return {
                    "format": "structured_json",
                    "ui_component": parsed.get("type", "json"),
                    "data": parsed,
                    "raw": raw_text,
                    "tokens_generated": token_count,
                    "latency_ms": latency_ms,
                }

        return {
            "format": "text",
            "ui_component": "markdown",
            "data": raw_text,
            "raw": raw_text,
            "tokens_generated": token_count,
            "latency_ms": latency_ms,
        }

    @staticmethod
    def _try_parse_json(raw_text: str) -> Optional[Dict[str, Any]]:
        """Parses JSON string with fallback extraction up to last closing brace."""
        try:
            return json.loads(raw_text, strict=False)
        except json.JSONDecodeError:
            last_brace = raw_text.rfind("}")
            if last_brace != -1:
                try:
                    return json.loads(raw_text[: last_brace + 1], strict=False)
                except json.JSONDecodeError:
                    return None
            return None
