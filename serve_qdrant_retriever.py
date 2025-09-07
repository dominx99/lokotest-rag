# serve_qdrant_retriever.py — Use Qdrant + OpenAI embeddings + BM25 + reranker
import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import regex as re

from fastapi import FastAPI, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from qdrant_store import get_qdrant_store

INDEX_DIR  = Path("rag_prep/index")
BM25_PATH  = INDEX_DIR / "bm25.pkl"
INFO_PATH  = INDEX_DIR / "qdrant_info.json"

# ---------- Tunables ----------
VEC_TOPK   = int(os.environ.get("VEC_TOPK", "100"))
BM25_TOPK  = int(os.environ.get("BM25_TOPK", "150"))
MERGE_TOPK = int(os.environ.get("MERGE_TOPK", "40"))
FINAL_K    = int(os.environ.get("FINAL_K", "8"))
RRF_K      = int(os.environ.get("RRF_K", "60"))

# Reranker (can be changed via env RERANKER)
RERANKER_NAME = os.environ.get("RERANKER", "jinaai/jina-reranker-v2-base-multilingual")
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "rag_documents")

# Tokenizer for BM25 (keep numbers + km/h)
TOKEN_RE = re.compile(r"(?:\p{L}+|\d+(?:[.,]\d+)?|km/?h)", re.UNICODE | re.IGNORECASE)

def normalize_units(s: str) -> str:
    if not s:
        return ""
    s = s.replace("KM / H", "km/h").replace("KM/ H", "km/h").replace("KM /H", "km/h")
    s = s.replace("km / h", "km/h").replace("km/ h", "km/h").replace("km /h", "km/h")
    return s

def tokenize_pl(s: str):
    s = normalize_units(s)
    return [m.group(0).lower() for m in TOKEN_RE.finditer(s)]

def rrf(id_lists: List[List[str]], k: int = 60) -> Dict[str, float]:
    """Reciprocal Rank Fusion for string IDs."""
    fused: Dict[str, float] = {}
    for ranked in id_lists:
        for rank, idx in enumerate(ranked, 1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return fused

# Initialize components
print("Loading Qdrant, BM25, models…")

# Load Qdrant store
qdrant_store = get_qdrant_store(COLLECTION_NAME)

# Create metadata cache for efficiency
print("Building metadata cache...")
metadata_cache = {}
try:
    all_points = qdrant_store.scroll_all_points()
    for point in all_points:
        chunk_id = point["payload"].get("chunk_id")
        if chunk_id:
            metadata_cache[chunk_id] = point["payload"]
    print(f"Cached metadata for {len(metadata_cache)} chunks")
except Exception as e:
    print(f"Warning: Could not build metadata cache: {e}")
    print("Will build cache on-demand during searches")

# Load BM25 index
with open(BM25_PATH, "rb") as f:
    bm25_pack = pickle.load(f)
bm25 = bm25_pack["bm25"]
bm25_meta = bm25_pack["meta"]

# Load collection info
with open(INFO_PATH, "r", encoding="utf-8") as f:
    info = json.load(f)
EMB_MODEL = info["model"]
EMB_DIM   = info["dim"]
print(f"Embedding backend: Qdrant + OpenAI | model: {EMB_MODEL} | dim: {EMB_DIM}")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Reranker
reranker = SentenceTransformer(
    RERANKER_NAME,
    device=device,
    trust_remote_code=True
)
try:
    reranker.max_seq_length = 8192
except Exception:
    pass
print(f"Device: {device}")

# OpenAI client for query embeddings
oa_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def embed_query(q: str) -> np.ndarray:
    kwargs = {"model": EMB_MODEL, "input": [q]}
    resp = oa_client.embeddings.create(**kwargs)
    v = np.array(resp.data[0].embedding, dtype="float32")
    # Normalize to unit length for cosine similarity matching
    v = v / np.linalg.norm(v)
    return v

def vec_search(q: str, topk: int) -> List[Tuple[str, float]]:
    """Search Qdrant collection for similar vectors."""
    qv = embed_query(q)
    hits = qdrant_store.search(
        query_vector=qv,
        limit=topk
    )
    # Return (chunk_id, score) tuples
    return [(hit["chunk_id"], hit["score"]) for hit in hits]

def bm25_search(q: str, topk: int) -> List[Tuple[str, float]]:
    """Search BM25 index and return chunk_ids."""
    toks = tokenize_pl(q)
    scores = bm25.get_scores(toks)
    ranked = np.argsort(scores)[::-1][:topk]
    
    # Convert row indices to chunk_ids
    results = []
    for i in ranked:
        if i < len(bm25_meta):
            chunk_id = bm25_meta[i]["chunk_id"]
            score = float(scores[i])
            results.append((chunk_id, score))
    
    return results

def get_texts_for_chunk_ids(chunk_ids: List[str]) -> Dict[str, str]:
    """Get text content for chunk IDs from metadata cache or direct search."""
    texts: Dict[str, str] = {}
    
    # First try from cache
    for chunk_id in chunk_ids:
        if chunk_id in metadata_cache:
            texts[chunk_id] = metadata_cache[chunk_id].get("text", "")
    
    # If cache doesn't have all chunks, search directly
    missing_chunks = set(chunk_ids) - set(texts.keys())
    if missing_chunks:
        try:
            all_points = qdrant_store.scroll_all_points()
            for point in all_points:
                payload_chunk_id = point["payload"].get("chunk_id")
                if payload_chunk_id in missing_chunks:
                    texts[payload_chunk_id] = point["payload"].get("text", "")
        except Exception as e:
            print(f"Warning: Could not retrieve missing chunks: {e}")
    
    return texts

def text_match_search(q: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Direct text matching search like the working Qdrant query."""
    hits = qdrant_store.text_search(q, limit=limit)
    
    results = []
    for hit in hits:
        payload = hit.get("payload", {})
        results.append({
            "score": float(hit.get("score", 0.0)),
            "title": payload.get("title"),
            "source_path": payload.get("source_path"),
            "page": payload.get("page"),
            "chunk_index": payload.get("chunk_index"),
            "chunk_id": payload.get("chunk_id", ""),
            "text": payload.get("text", ""),
        })
    
    return results

def merge_and_rerank(q: str) -> List[Dict[str, Any]]:
    """Simplified search: prioritize text matching, fallback to vector search."""
    # First try text matching (like the working Qdrant query)
    text_results = text_match_search(q, limit=FINAL_K)
    
    # If we get good text matches, use them
    if text_results and any(r["score"] > 0.8 for r in text_results):
        return text_results[:FINAL_K]
    
    # Fallback to vector search only
    vec_results = vec_search(q, FINAL_K)
    results = []
    
    for chunk_id, score in vec_results:
        if chunk_id in metadata_cache:
            metadata = metadata_cache[chunk_id]
            results.append({
                "score": float(score),
                "title": metadata.get("title"),
                "source_path": metadata.get("source_path"),
                "page": metadata.get("page"),
                "chunk_index": metadata.get("chunk_index"),
                "chunk_id": metadata.get("chunk_id"),
                "text": metadata.get("text", ""),
            })
    
    return results

# ---------- CLI ----------
def cli_once():
    try:
        q = input("Zapytanie: ").strip()
        out = merge_and_rerank(q)
        for rnk, r in enumerate(out, 1):
            print(f"\n[{rnk}] score={r['score']:.3f}")
            print(f"   tytuł: {r.get('title')}")
            print(f"   plik:  {r.get('source_path')}")
            print(f"   str.:  {r.get('page')}  chunk: {r.get('chunk_index')}")
            print(f"   id:    {r.get('chunk_id')}")
            print("   ---")
            print((r.get('text') or "").replace('\n', ' ')[:600])
        print("\nDone.")
    except KeyboardInterrupt:
        pass

# ---------- HTTP ----------
class Hit(BaseModel):
    score: float
    title: str | None
    source_path: str | None
    page: int | None
    chunk_index: int | None
    chunk_id: str
    text: str

class SearchResponse(BaseModel):
    query: str
    hits: List[Hit]

app = FastAPI(title="RAG Retriever (PL) — Qdrant + OpenAI embeddings")

@app.get("/search", response_model=SearchResponse)
def search(q: str = Query(..., description="Polskie zapytanie")):
    hits = merge_and_rerank(q)
    return {"query": q, "hits": hits}

if __name__ == "__main__":
    import sys
    if "--once" in sys.argv:
        cli_once()
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)