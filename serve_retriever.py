# serve_retriever.py — Use OpenAI embeddings for queries + BM25 + reranker API
import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss
import torch
import regex as re

from fastapi import FastAPI, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from openai import OpenAI

INDEX_DIR  = Path("rag_prep/index")
FAISS_PATH = INDEX_DIR / "faiss.index"
META_PATH  = INDEX_DIR / "meta.jsonl"
BM25_PATH  = INDEX_DIR / "bm25.pkl"
INFO_PATH  = INDEX_DIR / "index_info.json"

# ---------- Tunables ----------
VEC_TOPK   = int(os.environ.get("VEC_TOPK", "100"))
BM25_TOPK  = int(os.environ.get("BM25_TOPK", "150"))
MERGE_TOPK = int(os.environ.get("MERGE_TOPK", "40"))
FINAL_K    = int(os.environ.get("FINAL_K", "8"))
RRF_K      = int(os.environ.get("RRF_K", "60"))

# Reranker (can be changed via env RERANKER)
RERANKER_NAME = os.environ.get("RERANKER", "jinaai/jina-reranker-v2-base-multilingual")

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

def load_meta() -> List[Dict[str, Any]]:
    items = []
    with open(META_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            m = json.loads(line)
            m["_row_id"] = i
            items.append(m)
    return items

def rrf(id_lists: List[List[int]], k: int = 60) -> Dict[int, float]:
    fused: Dict[int, float] = {}
    for ranked in id_lists:
        for rank, idx in enumerate(ranked, 1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return fused

print("Loading FAISS, BM25, models…")
index = faiss.read_index(str(FAISS_PATH))
meta = load_meta()
with open(BM25_PATH, "rb") as f:
    bm25_pack = pickle.load(f)
bm25 = bm25_pack["bm25"]

with open(INFO_PATH, "r", encoding="utf-8") as f:
    info = json.load(f)
EMB_MODEL = info["model"]
EMB_DIM   = info["dim"]
print(f"Embedding backend: OpenAI | model: {EMB_MODEL} | dim: {EMB_DIM}")

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
    # If build step used dimensions override, it’s already baked into vectors;
    # here we don’t need to repeat it (OpenAI returns the model’s default dim unless specified).
    resp = oa_client.embeddings.create(**kwargs)
    v = np.array(resp.data[0].embedding, dtype="float32")[None, :]
    # Normalize to unit length for cosine/IP matching
    faiss.normalize_L2(v)
    return v

def vec_search(q: str, topk: int) -> List[Tuple[int, float]]:
    qv = embed_query(q)
    scores, ids = index.search(qv, topk)
    return [(int(i), float(s)) for i, s in zip(ids[0], scores[0])]

def bm25_search(q: str, topk: int) -> List[Tuple[int, float]]:
    toks = tokenize_pl(q)
    scores = bm25.get_scores(toks)
    ranked = np.argsort(scores)[::-1][:topk]
    return [(int(i), float(scores[i])) for i in ranked]

def read_texts_for_ids(cand_ids: List[int]) -> Dict[int, str]:
    id_set = set(cand_ids)
    texts: Dict[int, str] = {}
    with open("rag_prep/data/chunks.jsonl", "r", encoding="utf-8") as f:
        for rid, line in enumerate(f):
            if rid in id_set:
                rec = json.loads(line)
                texts[rid] = rec.get("text", "")
                if len(texts) == len(id_set):
                    break
    return texts

def merge_and_rerank(q: str) -> List[Dict[str, Any]]:
    vec = vec_search(q, VEC_TOPK)
    bm  = bm25_search(q, BM25_TOPK)

    vec_ids = [i for i, _ in vec]
    bm_ids  = [i for i, _ in bm]

    fused_scores = rrf([vec_ids, bm_ids], k=RRF_K)
    fused_sorted = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:MERGE_TOPK]
    cand_ids = [i for i, _ in fused_sorted]

    texts = read_texts_for_ids(cand_ids)
    pairs = [(q, texts.get(i, "")) for i in cand_ids]

    # Rerank
    try:
        scores = reranker.compute_score(pairs, batch_size=8)
    except Exception:
        q_emb = reranker.encode([q], normalize_embeddings=True)
        p_emb = reranker.encode([t for _, t in pairs], normalize_embeddings=True)
        scores = (q_emb @ p_emb.T)[0].tolist()

    ranked = sorted(zip(cand_ids, scores), key=lambda x: x[1], reverse=True)[:FINAL_K]

    results = []
    for i, sc in ranked:
        m = meta[i]
        results.append({
            "score": float(sc),
            "title": m.get("title"),
            "source_path": m.get("source_path"),
            "page": m.get("page"),
            "chunk_index": m.get("chunk_index"),
            "chunk_id": m.get("chunk_id"),
            "text": texts.get(i, ""),
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

app = FastAPI(title="RAG Retriever (PL) — OpenAI embeddings")

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
        uvicorn.run(app, host="127.0.0.1", port=8000)
