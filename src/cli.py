#!/usr/bin/env python3
"""One-Command Global CLI for Northh Dhruva 500M:
- `dhruva ingest <file>` — Ingest PDF or text documents into `~/.dhruva/rag_index.json`
- `dhruva ask "<question>"` — Similarity-gated bilingual RAG query with conversational fallback
- `dhruva run --quant [bf16|int8]` — Interactive chat & JSON extraction CLI
- `dhruva serve --port 8000` — FastAPI server (`/v1/generate`, `/v1/rag/ingest`, `/v1/rag/query`, `/health`)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_HF_REPO_ID = os.environ.get("DHRUVA_HF_REPO", "northh-ai/dhruva-500m-instruct-json")


def resolve_or_download_model_dir(
    model_dir: str = "release/dhruva-500m-instruct-json",
    quant: str = "bf16",
    repo_id: str = DEFAULT_HF_REPO_ID,
) -> str:
    """Finds local release directory or selectively downloads requested precision from Hugging Face Hub."""
    required_weight_file = "dhruva_500m_int8.pt" if quant == "int8" else "model.safetensors"

    local_candidates = [
        Path(model_dir),
        PROJECT_ROOT / model_dir,
        PROJECT_ROOT.parent / model_dir,
    ]
    for cand in local_candidates:
        if cand.is_dir() and (cand / "config.json").is_file() and (cand / required_weight_file).is_file():
            return str(cand.resolve())

    from huggingface_hub import snapshot_download

    if quant == "int8":
        ignore_patterns = ["*.safetensors", "model.safetensors"]
        size_hint = "~475 MB (INT8)"
    else:
        ignore_patterns = ["*.pt", "dhruva_500m_int8.pt"]
        size_hint = "~940 MB (BF16 Safetensors)"

    print(
        f"[Dhruva] Local release not found. Downloading '{repo_id}' [{size_hint}] "
        f"into Hugging Face cache (~/.cache/huggingface/hub/)..."
    )
    cached_dir = snapshot_download(
        repo_id=repo_id,
        ignore_patterns=ignore_patterns,
    )
    return str(cached_dir)


def _add_common_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--quant",
        type=str,
        choices=["bf16", "int8"],
        default="bf16",
        help="Precision format ('bf16' for model.safetensors, 'int8' for dhruva_500m_int8.pt)",
    )
    p.add_argument(
        "--model-dir",
        type=str,
        default="release/dhruva-500m-instruct-json",
        help="Local release folder (downloads from Hugging Face Hub if absent)",
    )
    p.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_HF_REPO_ID,
        help="Hugging Face Hub repository ID for auto-download fallback",
    )
    p.add_argument(
        "--context-window",
        type=int,
        default=2048,
        help="Maximum context window tokens (default: 2048)",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Max CPU threads for PyTorch (default: 4 to prevent laptop thermal/power spikes)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dhruva",
        description="Northh Dhruva 500M — Bilingual (English/Hindi), Structured JSON & RAFT RAG CLI",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # 1. `dhruva run`
    run_parser = subparsers.add_parser(
        "run",
        help="Start interactive chat / JSON extraction REPL or run a single prompt",
    )
    _add_common_model_args(run_parser)
    run_parser.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens")
    run_parser.add_argument("--prompt", type=str, default=None, help="Optional single prompt")
    run_parser.add_argument("--context", type=str, default=None, help="Optional context block")

    # 2. `dhruva serve`
    serve_parser = subparsers.add_parser(
        "serve",
        help="Launch FastAPI server (/v1/generate, /v1/rag/ingest, /v1/rag/query, /health)",
    )
    _add_common_model_args(serve_parser)
    serve_parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host (default: 0.0.0.0)")

    # 3. `dhruva ingest`
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest a PDF or text file into the local bilingual RAG index (~/.dhruva/rag_index.json)",
    )
    _add_common_model_args(ingest_parser)
    ingest_parser.add_argument("path", type=str, help="Path to PDF or UTF-8 text file to ingest")
    ingest_parser.add_argument("--doc-name", type=str, default=None, help="Optional document display name")
    ingest_parser.add_argument("--index-path", type=str, default=None, help="Optional custom RAG index path")

    # 4. `dhruva ask`
    ask_parser = subparsers.add_parser(
        "ask",
        help="Ask a question using similarity-gated RAG with automatic conversational fallback",
    )
    _add_common_model_args(ask_parser)
    ask_parser.add_argument("question", type=str, help="Question or greeting in English or Hindi")
    ask_parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve")
    ask_parser.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens")
    ask_parser.add_argument(
        "--output-mode",
        type=str,
        choices=["auto", "table", "specs_card"],
        default="auto",
        help="Output mode ('auto', 'table', or 'specs_card')",
    )
    ask_parser.add_argument(
        "--threshold",
        type=float,
        default=0.38,
        help="Cosine similarity threshold for RAG gating (default: 0.38)",
    )
    ask_parser.add_argument("--index-path", type=str, default=None, help="Optional custom RAG index path")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    import torch

    if getattr(args, "threads", None) and args.threads > 0:
        torch.set_num_threads(args.threads)

    from src.inference.engine import DhruvaEngine
    from src.rag.pipeline import DhruvaRAGPipeline
    from scripts.serve_dhruva import create_fastapi_app, run_cli_mode

    resolved_dir = resolve_or_download_model_dir(
        model_dir=args.model_dir,
        quant=args.quant,
        repo_id=args.repo_id,
    )

    engine = DhruvaEngine(
        model_dir=resolved_dir,
        quant=args.quant,
        context_window=args.context_window,
    )

    if args.subcommand == "run":
        if args.prompt is not None:
            res = engine.generate(
                prompt=args.prompt,
                context=args.context,
                max_new_tokens=args.max_new_tokens,
            )
            print(
                f"\n[Dhruva | quant={args.quant} | ui_component={res['ui_component']} | "
                f"format={res['format']} | {res['tokens_generated']} tokens | {res['latency_ms']} ms]:"
            )
            if res["format"] == "structured_json":
                print(json.dumps(res["data"], indent=2, ensure_ascii=False))
            else:
                print(res["data"])
            return

        run_cli_mode(engine, default_max_tokens=args.max_new_tokens)

    elif args.subcommand == "serve":
        import uvicorn

        app = create_fastapi_app(engine)
        uvicorn.run(app, host=args.host, port=args.port)

    elif args.subcommand == "ingest":
        rag = DhruvaRAGPipeline(engine=engine, index_path=args.index_path)
        file_path = Path(args.path)
        if not file_path.is_file():
            print(f"Error: File not found: {args.path}")
            sys.exit(1)
        if file_path.suffix.lower() == ".pdf":
            added = rag.ingest_pdf(str(file_path), doc_name=args.doc_name)
        else:
            raw_text = file_path.read_text(encoding="utf-8", errors="replace")
            added = rag.ingest_text(raw_text, doc_name=args.doc_name or file_path.name, page=1)
        print(f"[Dhruva RAG] Ingested '{file_path.name}' -> {added} chunks indexed at {rag.index_path}")

    elif args.subcommand == "ask":
        rag = DhruvaRAGPipeline(
            engine=engine,
            index_path=args.index_path,
            similarity_threshold=args.threshold,
        )
        res = rag.query(
            question=args.question,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            output_mode=args.output_mode,
        )
        print(
            f"\n[Dhruva RAG | mode={res['mode']} | best_cosine={res['best_cosine']} | "
            f"ui_component={res['ui_component']} | {res['tokens_generated']} tokens | {res['latency_ms']} ms]:"
        )
        if res["format"] == "structured_json":
            print(json.dumps(res["data"], indent=2, ensure_ascii=False))
        else:
            print(res["data"])
        if res.get("sources"):
            print("\nSources:")
            for src in res["sources"]:
                print(f"  - {src['doc']} (Page {src['page']}) | cosine={src['cosine_score']} | rrf={src['rrf_score']}")


if __name__ == "__main__":
    main()
