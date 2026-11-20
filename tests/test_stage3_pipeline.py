#!/usr/bin/env python3
"""Verification suite for Northh Dhruva 500M Stage 3 Release, Inference Engine & FastAPI."""

import json
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.engine import DhruvaEngine
from scripts.serve_dhruva import create_fastapi_app


def main() -> None:
    print("=" * 76)
    print("  🧪 NORTHH DHRUVA 500M — STAGE 3 VERIFICATION SUITE")
    print("=" * 76)

    engine = DhruvaEngine(release_dir="release/dhruva-500m-instruct-json")
    print(f"  Loaded Release: {engine.release_dir}")
    print(f"  Device:         {engine.device}")
    print(f"  Active Format:  {engine.loaded_format}")
    print("-" * 76)

    # Test 1: Multi-row FRA land claim table extraction (type: "table")
    fra_context = (
        "Forest Rights Act (FRA) Gram Sabha Claim Register - Mandla District:\n"
        "1. Claimant: Ramesh Gond, Village: Bichhiya, Area Claimed: 1.85 Ha, Status: Approved\n"
        "2. Claimant: Sukarti Baiga, Village: Mawai, Area Claimed: 2.10 Ha, Status: Pending\n"
        "3. Claimant: Mangal Singh, Village: Ghughri, Area Claimed: 1.40 Ha, Status: Approved"
    )
    fra_prompt = "Extract the FRA land claims into a structured JSON table."
    print("\n[Test 1] FRA Land Claim Table Extraction:")
    res1 = engine.generate(prompt=fra_prompt, context=fra_context, max_new_tokens=200)
    print(f"  format:       {res1['format']}")
    print(f"  ui_component: {res1['ui_component']}")
    print(f"  tokens:       {res1['tokens_generated']} | latency: {res1['latency_ms']} ms")
    print(
        f"  output:\n{json.dumps(res1['data'], indent=2, ensure_ascii=False) if isinstance(res1['data'], dict) else res1['data']}"
    )

    # Test 2: Construction quote extraction (type: "specs_card")
    quote_context = (
        "Contractor Quotation: Standard Residential Package at Rs 1650 per sq.ft. "
        "Specifications: UltraTech OPC 43 Grade Cement, Tata Tiscon Fe500D TMT Steel, "
        "Finolex Electrical Wiring, and Asian Paints Royale finish."
    )
    quote_prompt = "Extract the construction quote into a specs_card JSON."
    print("\n[Test 2] Construction Quote Extraction (specs_card):")
    res2 = engine.generate(prompt=quote_prompt, context=quote_context, max_new_tokens=200)
    print(f"  format:       {res2['format']}")
    print(f"  ui_component: {res2['ui_component']}")
    print(f"  tokens:       {res2['tokens_generated']} | latency: {res2['latency_ms']} ms")
    print(
        f"  output:\n{json.dumps(res2['data'], indent=2, ensure_ascii=False) if isinstance(res2['data'], dict) else res2['data']}"
    )

    # Test 3: Standard English/Hindi Chat Prompt (markdown)
    chat_prompt = "भारत में वन अधिकार अधिनियम (FRA) का मुख्य उद्देश्य क्या है? संक्षेप में बताएं।"
    print("\n[Test 3] Hindi/English Conversational Chat Prompt:")
    res3 = engine.generate(prompt=chat_prompt, context=None, max_new_tokens=100)
    print(f"  format:       {res3['format']}")
    print(f"  ui_component: {res3['ui_component']}")
    print(f"  tokens:       {res3['tokens_generated']} | latency: {res3['latency_ms']} ms")
    print(f"  output:\n{res3['data']}")

    # Test 4: FastAPI Endpoints (/health)
    app = create_fastapi_app(engine)
    routes = [r.path for r in app.routes]
    assert "/health" in routes and "/v1/generate" in routes, f"Missing routes: {routes}"
    print("\n[Test 4] FastAPI Endpoints Registered: /health, /v1/generate [PASS]")
    print("=" * 76)


if __name__ == "__main__":
    main()
