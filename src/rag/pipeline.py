"""Similarity-Gated Bilingual RAG Pipeline (`DhruvaRAGPipeline`) for Northh Dhruva 500M.

Features:
- PDF & text ingestion using PyMuPDF (`fitz`) and Dhruva's 32K ByteBPETokenizer
  (280-token chunks with 40-token overlap).
- Multilingual dense embeddings via `BAAI/bge-m3` (`normalize_embeddings=True`).
- Local JSON index persistence at `~/.dhruva/rag_index.json` with optional
  PostgreSQL + `pgvector` storage when `DATABASE_URL` is configured.
- Hybrid Search combining dense cosine similarity + exact keyword overlap via
  Reciprocal Rank Fusion (`rrf_k=60`), preserving each hit's raw `cosine_score`.
- Smart Conversational & Abstention Gate: bypasses chunk injection when
  `not hits` or `best_cosine < 0.38` so greetings & general chat converse naturally.
"""

from dataclasses import dataclass
import json
import math
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.inference.engine import DhruvaEngine


WORD_RE = re.compile(r"[\w\u0900-\u097F]+", re.UNICODE)


@dataclass
class SearchHit:
    chunk_id: str
    doc_name: str
    page: int
    text: str
    cosine_score: float
    keyword_score: float
    rrf_score: float


def _dot(a: List[float], b: List[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def _tokenize_keywords(text: str) -> set:
    return {w.lower() for w in WORD_RE.findall(text) if len(w) > 1}


class DhruvaRAGPipeline:
    """Similarity-gated bilingual RAG pipeline backed by BAAI/bge-m3 and DhruvaEngine."""

    def __init__(
        self,
        engine: DhruvaEngine,
        index_path: Optional[str] = None,
        embed_model_name: str = "BAAI/bge-m3",
        chunk_tokens: int = 280,
        chunk_overlap: int = 40,
        similarity_threshold: float = 0.38,
        abstain_threshold: float = 0.48,
        database_url: Optional[str] = None,
    ) -> None:
        self.engine = engine
        self.embed_model_name = embed_model_name
        self.chunk_tokens = chunk_tokens
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold
        self.abstain_threshold = abstain_threshold
        self.database_url = database_url or os.environ.get("DATABASE_URL")

        default_index = Path.home() / ".dhruva" / "rag_index.json"
        self.index_path = Path(index_path) if index_path else default_index
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        self._embedder = None
        self._records: List[Dict[str, Any]] = []
        self._load_local_index()

        if self.database_url:
            self._init_pgvector_table()

    # -------------------------------------------------------------------------
    # Embedder & Persistence Helpers
    # -------------------------------------------------------------------------

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.embed_model_name)
        return self._embedder

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        embedder = self._get_embedder()
        vecs = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vecs]

    def _load_local_index(self) -> None:
        if self.index_path.is_file():
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self._records = data
            except Exception:
                self._records = []

    def _save_local_index(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self._records, f, ensure_ascii=False)

    def _init_pgvector_table(self) -> None:
        """Initializes PostgreSQL + pgvector table when DATABASE_URL is configured."""
        try:
            import psycopg2

            with psycopg2.connect(self.database_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS dhruva_rag_chunks (
                            chunk_id TEXT PRIMARY KEY,
                            doc_name TEXT NOT NULL,
                            page INTEGER NOT NULL,
                            text TEXT NOT NULL,
                            embedding vector(1024)
                        );
                        """
                    )
                conn.commit()
        except Exception as exc:
            print(f"[RAG] PostgreSQL/pgvector initialization skipped ({exc}); using local JSON index.")
            self.database_url = None

    def _upsert_pgvector(self, new_records: List[Dict[str, Any]]) -> None:
        if not self.database_url or not new_records:
            return
        try:
            import psycopg2

            with psycopg2.connect(self.database_url) as conn:
                with conn.cursor() as cur:
                    for rec in new_records:
                        vec_str = "[" + ",".join(f"{x:.6f}" for x in rec["embedding"]) + "]"
                        cur.execute(
                            """
                            INSERT INTO dhruva_rag_chunks (chunk_id, doc_name, page, text, embedding)
                            VALUES (%s, %s, %s, %s, %s::vector)
                            ON CONFLICT (chunk_id) DO UPDATE
                            SET text = EXCLUDED.text, embedding = EXCLUDED.embedding;
                            """,
                            (rec["chunk_id"], rec["doc_name"], rec["page"], rec["text"], vec_str),
                        )
                conn.commit()
        except Exception as exc:
            print(f"[RAG] PostgreSQL upsert warning: {exc}")

    # -------------------------------------------------------------------------
    # Chunking & Ingestion
    # -------------------------------------------------------------------------

    def _chunk_by_tokens(self, text: str) -> List[str]:
        """Splits text into `chunk_tokens`-sized chunks with `chunk_overlap` using Dhruva's tokenizer."""
        clean = text.strip()
        if not clean:
            return []

        token_ids = self.engine.tokenizer.encode(clean)
        if not token_ids:
            return []

        step = max(1, self.chunk_tokens - self.chunk_overlap)
        chunks: List[str] = []
        for start in range(0, len(token_ids), step):
            window = token_ids[start : start + self.chunk_tokens]
            if not window:
                break
            chunk_text = self.engine.tokenizer.decode(window).strip()
            if chunk_text:
                chunks.append(chunk_text)
            if start + self.chunk_tokens >= len(token_ids):
                break
        return chunks

    def ingest_text(self, text: str, doc_name: str, page: int = 1) -> int:
        """Chunks raw text, computes BAAI/bge-m3 embeddings, and persists to the RAG index."""
        chunks = self._chunk_by_tokens(text)
        if not chunks:
            return 0

        embeddings = self._embed_texts(chunks)
        new_records: List[Dict[str, Any]] = []
        for ch_text, emb in zip(chunks, embeddings):
            rec = {
                "chunk_id": str(uuid.uuid4()),
                "doc_name": doc_name,
                "page": int(page),
                "text": ch_text,
                "embedding": emb,
            }
            self._records.append(rec)
            new_records.append(rec)

        self._save_local_index()
        self._upsert_pgvector(new_records)
        return len(new_records)

    def ingest_pdf(self, pdf_path: str, doc_name: Optional[str] = None) -> int:
        """Extracts text page-by-page using PyMuPDF (`fitz`), chunks, embeds, and indexes."""
        import fitz  # PyMuPDF

        path_obj = Path(pdf_path)
        if not path_obj.is_file():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        resolved_name = doc_name or path_obj.name
        total_chunks = 0
        pending_texts: List[str] = []
        pending_meta: List[Tuple[str, int]] = []

        with fitz.open(str(path_obj)) as doc:
            for page_idx, page in enumerate(doc, start=1):
                page_text = page.get_text("text").strip()
                if not page_text:
                    continue
                for ch in self._chunk_by_tokens(page_text):
                    pending_texts.append(ch)
                    pending_meta.append((resolved_name, page_idx))

        if not pending_texts:
            return 0

        embeddings = self._embed_texts(pending_texts)
        new_records: List[Dict[str, Any]] = []
        for (d_name, pg), ch_text, emb in zip(pending_meta, pending_texts, embeddings):
            rec = {
                "chunk_id": str(uuid.uuid4()),
                "doc_name": d_name,
                "page": pg,
                "text": ch_text,
                "embedding": emb,
            }
            self._records.append(rec)
            new_records.append(rec)
            total_chunks += 1

        self._save_local_index()
        self._upsert_pgvector(new_records)
        return total_chunks

    # -------------------------------------------------------------------------
    # Hybrid Search + Reciprocal Rank Fusion (RRF)
    # -------------------------------------------------------------------------

    def hybrid_retrieve(
        self,
        query: str,
        top_k: int = 4,
        rrf_k: int = 60,
    ) -> List[SearchHit]:
        """Combines dense BAAI/bge-m3 cosine similarity with exact keyword overlap via RRF."""
        if not self._records:
            return []

        query_vec = self._embed_texts([query])[0]
        query_terms = _tokenize_keywords(query)

        scored: List[Tuple[int, float, float]] = []
        for idx, rec in enumerate(self._records):
            cos_sim = _dot(query_vec, rec["embedding"])
            doc_terms = _tokenize_keywords(rec["text"])
            if query_terms and doc_terms:
                kw_overlap = len(query_terms & doc_terms) / float(len(query_terms))
            else:
                kw_overlap = 0.0
            scored.append((idx, cos_sim, kw_overlap))

        # Rank by dense cosine similarity
        dense_ranked = sorted(scored, key=lambda item: item[1], reverse=True)
        dense_rank_map = {idx: rank for rank, (idx, _, _) in enumerate(dense_ranked, start=1)}

        # Rank by exact keyword overlap
        kw_ranked = sorted(scored, key=lambda item: (item[2], item[1]), reverse=True)
        kw_rank_map = {idx: rank for rank, (idx, _, _) in enumerate(kw_ranked, start=1)}

        # Reciprocal Rank Fusion (RRF)
        fused: List[SearchHit] = []
        for idx, cos_sim, kw_score in scored:
            r_dense = dense_rank_map[idx]
            r_kw = kw_rank_map[idx]
            rrf_score = (1.0 / (rrf_k + r_dense)) + (1.0 / (rrf_k + r_kw))
            rec = self._records[idx]
            fused.append(
                SearchHit(
                    chunk_id=rec["chunk_id"],
                    doc_name=rec["doc_name"],
                    page=int(rec["page"]),
                    text=rec["text"],
                    cosine_score=float(cos_sim),
                    keyword_score=float(kw_score),
                    rrf_score=float(rrf_score),
                )
            )

        fused.sort(key=lambda h: (h.rrf_score, h.cosine_score), reverse=True)
        return fused[:top_k]

    # -------------------------------------------------------------------------
    # Similarity-Gated Query Execution
    # -------------------------------------------------------------------------

    def query(
        self,
        question: str,
        top_k: int = 4,
        max_new_tokens: int = 256,
        output_mode: str = "auto",
        temperature: float = 0.0,
        similarity_threshold: Optional[float] = None,
        abstain_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Runs 3-tier similarity-gated RAG query:
        - Tier 1 (`not hits` or `best_cosine < 0.38`): `conversational_fallback` (no chunks).
        - Tier 2 (`0.38 <= best_cosine < 0.48` and `output_mode == "auto"`): `rag_abstain`
          (returns clean abstention directly without invoking the LLM on weak chunks).
        - Tier 3 (`best_cosine >= 0.48`): `rag_grounded` (packs top chunks for cited synthesis).
        """
        t0 = time.perf_counter()
        conv_thresh = (
            self.similarity_threshold
            if similarity_threshold is None
            else float(similarity_threshold)
        )
        abs_thresh = (
            self.abstain_threshold
            if abstain_threshold is None
            else float(abstain_threshold)
        )

        hits = self.hybrid_retrieve(question, top_k=top_k)
        best_cosine = max((h.cosine_score for h in hits), default=0.0)
        mode_norm = (output_mode or "auto").lower()

        # Tier 1: Conversational Fallback (greetings / general chat unrelated to indexed docs)
        if not hits or best_cosine < conv_thresh:
            res = self.engine.generate(
                prompt=question,
                context=None,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            res["mode"] = "conversational_fallback"
            res["best_cosine"] = round(best_cosine, 4)
            res["sources"] = []
            return res

        # Tier 2: Similarity-Gated Abstention Band (0.38 <= best_cosine < 0.48 in auto mode)
        if mode_norm == "auto" and best_cosine < abs_thresh:
            has_devanagari = any("\u0900" <= ch <= "\u097F" for ch in question)
            if has_devanagari:
                abstain_msg = "क्षमा करें, उपलब्ध दस्तावेजों में इस प्रश्न का सटीक उत्तर नहीं मिला."
            else:
                abstain_msg = "I am sorry, the provided documents do not contain enough relevant information to answer this question."
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return {
                "format": "text",
                "ui_component": "markdown",
                "data": abstain_msg,
                "raw": abstain_msg,
                "tokens_generated": 0,
                "latency_ms": latency_ms,
                "mode": "rag_abstain",
                "best_cosine": round(best_cosine, 4),
                "sources": [],
            }

        # Tier 3: Grounded Cited Synthesis (best_cosine >= 0.48 or explicit structured JSON mode)
        token_budget = max(128, self.engine.context_window - max_new_tokens - 96)
        question_tokens = len(self.engine.tokenizer.encode(question))
        remaining_budget = max(64, token_budget - question_tokens)

        packed_blocks: List[str] = []
        used_hits: List[SearchHit] = []
        used_tokens = 0

        for idx, hit in enumerate(hits, start=1):
            block = f"[Chunk {idx} | Doc: {hit.doc_name} | Page {hit.page}]\n{hit.text}"
            block_tokens = len(self.engine.tokenizer.encode(block))
            if used_tokens + block_tokens > remaining_budget and packed_blocks:
                break
            packed_blocks.append(block)
            used_hits.append(hit)
            used_tokens += block_tokens

        packed_context = "\n\n".join(packed_blocks)

        if mode_norm == "table":
            res = self.engine.generate(
                prompt=f"{question}\nExtract the information into a structured JSON table.",
                context=packed_context,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        elif mode_norm == "specs_card":
            res = self.engine.generate(
                prompt=f"{question}\nExtract the specifications into a structured specs_card JSON.",
                context=packed_context,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        else:
            res = self.engine.generate(
                prompt=question,
                retrieved_context=packed_context,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

        res["mode"] = "rag_grounded"
        res["best_cosine"] = round(best_cosine, 4)
        res["sources"] = [
            {
                "doc": h.doc_name,
                "page": h.page,
                "cosine_score": round(h.cosine_score, 4),
                "rrf_score": round(h.rrf_score, 6),
            }
            for h in used_hits
        ]
        return res
