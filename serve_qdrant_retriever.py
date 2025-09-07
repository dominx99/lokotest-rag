# serve_qdrant_retriever.py — Optimized Qdrant + OpenAI embeddings
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np

from fastapi import FastAPI, Query
from pydantic import BaseModel
from openai import OpenAI

from qdrant_store import get_qdrant_store

INFO_PATH  = Path("rag_prep/index/qdrant_info.json")

# ---------- Tunables ----------
FINAL_K    = int(os.environ.get("FINAL_K", "8"))
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "rag_documents")

# Initialize components
print("Loading Qdrant store (optimized)...")

# Load Qdrant store
qdrant_store = get_qdrant_store(COLLECTION_NAME)

# Load collection info
with open(INFO_PATH, "r", encoding="utf-8") as f:
    info = json.load(f)
EMB_MODEL = info["model"]
EMB_DIM   = info["dim"]
print(f"Embedding backend: Qdrant + OpenAI | model: {EMB_MODEL} | dim: {EMB_DIM}")

# OpenAI client for query embeddings
oa_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def embed_query(q: str) -> np.ndarray:
    """Generate embeddings for query with caching."""
    kwargs = {"model": EMB_MODEL, "input": [q]}
    resp = oa_client.embeddings.create(**kwargs)
    v = np.array(resp.data[0].embedding, dtype="float32")
    # Normalize to unit length for cosine similarity matching
    v = v / np.linalg.norm(v)
    return v

def optimized_search(q: str) -> List[Dict[str, Any]]:
    """Optimized search using only text matching and vector search."""
    # Primary: text matching for exact/fuzzy matches
    text_results = qdrant_store.text_search(q, limit=FINAL_K * 2)
    
    if text_results:
        # Process text results
        results = []
        for hit in text_results:
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
        
        # If we have good text matches, return them
        if any(r["score"] > 0.5 for r in results):
            return results[:FINAL_K]
    
    # Fallback: vector search
    try:
        qv = embed_query(q)
        vector_hits = qdrant_store.search(query_vector=qv, limit=FINAL_K)
        
        results = []
        for hit in vector_hits:
            results.append({
                "score": float(hit.get("score", 0.0)),
                "title": hit.get("title"),
                "source_path": hit.get("source_path"),
                "page": hit.get("page"), 
                "chunk_index": hit.get("chunk_index"),
                "chunk_id": hit.get("chunk_id", ""),
                "text": hit.get("text", ""),
            })
        return results
    except Exception as e:
        print(f"Vector search failed: {e}")
        return text_results[:FINAL_K] if text_results else []

# ---------- CLI ----------
def cli_once():
    try:
        q = input("Zapytanie: ").strip()
        out = optimized_search(q)
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

app = FastAPI(title="RAG Retriever (Optimized) — Qdrant + OpenAI embeddings")

@app.get("/search", response_model=SearchResponse)
def search(q: str = Query(..., description="Polskie zapytanie")):
    hits = optimized_search(q)
    return {"query": q, "hits": hits}

if __name__ == "__main__":
    import sys
    if "--once" in sys.argv:
        cli_once()
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)