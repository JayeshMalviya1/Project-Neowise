#!/usr/bin/env python3
"""Unified Production CLI & FastAPI Server for Northh Dhruva 500M (Stage 3 Instruct + JSON).

Usage:
- Interactive CLI:
    python scripts/serve_dhruva.py --mode cli
- FastAPI Server (port 8000):
    python scripts/serve_dhruva.py --mode api --port 8000
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.engine import DhruvaEngine


def run_cli_mode(engine: DhruvaEngine, default_max_tokens: int = 256) -> None:
    """Interactive terminal loop for English/Hindi chat and `/json` structured extraction."""
    print("\n" + "=" * 76)
    print("  🚀 NORTHH DHRUVA 500M (STAGE 3 INSTRUCT + JSON) — PRODUCTION CLI")
    print("=" * 76)
    print(f"  Release Dir:   {engine.release_dir}")
    print(f"  Active Device: {engine.device.upper()}")
    print(f"  Loaded Format: {engine.loaded_format.upper()}")
    print(f"  Architecture:  {engine.config.n_layers}L | {engine.config.dim}D | {engine.config.n_heads}H/{engine.config.n_kv_heads}KV GQA")
    print("-" * 76)
    print("  Commands:")
    print("    • Type any prompt in English or Hindi for standard chat.")
    print("    • /json   — Enter Context + Instruction separately for structured JSON extraction.")
    print("    • /clear  — Clear screen.")
    print("    • /exit   — Exit the CLI.")
    print("=" * 76 + "\n")

    while True:
        try:
            user_in = input("\033[1;36m[Dhruva CLI]\033[0m >>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting Dhruva CLI.")
            break

        if not user_in:
            continue

        cmd = user_in.lower()
        if cmd in ("/exit", "exit", "quit", "q"):
            print("Goodbye!")
            break

        if cmd == "/clear":
            os.system("cls" if os.name == "nt" else "clear")
            continue

        if cmd == "/json":
            print("\n--- Structured JSON Extraction Mode ---")
            try:
                context_val = input("\033[1;33mContext\033[0m     >>> ").strip()
                prompt_val = input("\033[1;33mInstruction\033[0m >>> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nCancelled /json input.\n")
                continue

            if not prompt_val:
                print("Instruction cannot be empty.\n")
                continue

            result = engine.generate(
                prompt=prompt_val,
                context=context_val,
                max_new_tokens=default_max_tokens,
            )
        else:
            result = engine.generate(
                prompt=user_in,
                context=None,
                max_new_tokens=default_max_tokens,
            )

        ui_comp = result["ui_component"]
        fmt = result["format"]
        latency = result["latency_ms"]
        tok_cnt = result["tokens_generated"]

        print(f"\n\033[1;32m[Dhruva | ui_component={ui_comp} | format={fmt} | {tok_cnt} tokens | {latency:.2f} ms]:\033[0m")
        if fmt == "structured_json":
            print(json.dumps(result["data"], indent=2, ensure_ascii=False))
        else:
            print(result["data"])
        print()


def create_fastapi_app(engine: DhruvaEngine):
    """Creates and configures the FastAPI application with Chat, JSON & Similarity-Gated RAG routes."""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    from src.rag.pipeline import DhruvaRAGPipeline

    app = FastAPI(
        title="Northh Dhruva 500M Production API",
        description="Bilingual English/Hindi Chat, Structured JSON Extraction & Similarity-Gated RAG Engine",
        version="4.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    rag_state: Dict[str, Optional[DhruvaRAGPipeline]] = {"pipeline": None}

    def _get_rag() -> DhruvaRAGPipeline:
        if rag_state["pipeline"] is None:
            rag_state["pipeline"] = DhruvaRAGPipeline(engine=engine)
        return rag_state["pipeline"]

    class GenerateRequest(BaseModel):
        prompt: str
        context: str = ""
        retrieved_context: str = ""
        max_new_tokens: int = 256
        temperature: float = 0.0

    class RAGIngestRequest(BaseModel):
        pdf_path: Optional[str] = None
        text: Optional[str] = None
        doc_name: str = "uploaded_doc.txt"
        page: int = 1

    class RAGQueryRequest(BaseModel):
        question: str
        top_k: int = 4
        max_new_tokens: int = 256
        output_mode: str = "auto"
        temperature: float = 0.0
        similarity_threshold: float = 0.38

    @app.get("/health")
    def health_check() -> Dict[str, Any]:
        return {
            "status": "ok",
            "model": "dhruva-500m-raft-json",
            "device": engine.device,
            "loaded_format": engine.loaded_format,
            "vocab_size": engine.config.vocab_size,
            "n_layers": engine.config.n_layers,
            "dim": engine.config.dim,
        }

    @app.post("/v1/generate")
    def generate_endpoint(req: GenerateRequest) -> Dict[str, Any]:
        return engine.generate(
            prompt=req.prompt,
            context=req.context if req.context.strip() else None,
            retrieved_context=req.retrieved_context if req.retrieved_context.strip() else None,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
        )

    @app.post("/v1/rag/ingest")
    def rag_ingest_endpoint(req: RAGIngestRequest) -> Dict[str, Any]:
        rag = _get_rag()
        if req.pdf_path:
            try:
                count = rag.ingest_pdf(req.pdf_path, doc_name=req.doc_name if req.doc_name != "uploaded_doc.txt" else None)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc))
            return {"status": "ingested", "source": req.pdf_path, "chunks_added": count}
        if req.text and req.text.strip():
            count = rag.ingest_text(req.text, doc_name=req.doc_name, page=req.page)
            return {"status": "ingested", "source": req.doc_name, "chunks_added": count}
        raise HTTPException(status_code=400, detail="Provide either 'pdf_path' or non-empty 'text'.")

    @app.post("/v1/rag/query")
    def rag_query_endpoint(req: RAGQueryRequest) -> Dict[str, Any]:
        rag = _get_rag()
        return rag.query(
            question=req.question,
            top_k=req.top_k,
            max_new_tokens=req.max_new_tokens,
            output_mode=req.output_mode,
            temperature=req.temperature,
            similarity_threshold=req.similarity_threshold,
        )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Northh Dhruva 500M Production CLI & FastAPI Server"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["cli", "api"],
        default="cli",
        help="Execution mode: 'cli' for interactive terminal or 'api' for FastAPI server",
    )
    parser.add_argument(
        "--release_dir",
        type=str,
        default="release/dhruva-500m-instruct-json",
        help="Path to exported release directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override target device ('cpu', 'cuda', 'mps')",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["safetensors", "int8"],
        default=None,
        help="Optional override for model weight format ('safetensors' or 'int8')",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="FastAPI bind host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="FastAPI bind port (default: 8000)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Default max_new_tokens in CLI mode",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[Dhruva] Initializing DhruvaEngine from '{args.release_dir}'...")
    engine = DhruvaEngine(
        release_dir=args.release_dir,
        device=args.device,
        prefer_format=args.format,
    )
    print(
        f"[Dhruva] Loaded model successfully (device={engine.device}, format={engine.loaded_format})."
    )

    if args.mode == "cli":
        run_cli_mode(engine, default_max_tokens=args.max_new_tokens)
    else:
        import uvicorn

        app = create_fastapi_app(engine)
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
