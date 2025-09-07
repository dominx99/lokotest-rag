# build_index.py — Build FAISS index using OpenAI embeddings (text-embedding-3-*)
import os
import json
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
from tqdm import tqdm
from openai import OpenAI

CHUNKS_PATH = Path("rag_prep/data/chunks.jsonl")
INDEX_DIR   = Path("rag_prep/index")
FAISS_PATH  = INDEX_DIR / "faiss.index"
META_PATH   = INDEX_DIR / "meta.jsonl"
INFO_PATH   = INDEX_DIR / "index_info.json"

EMB_MODEL   = os.environ.get("EMB_MODEL", "text-embedding-3-large")   # or text-embedding-3-small
# For text-embedding-3 models you may optionally set dimensions (e.g. 1536 for 3-large)
EMB_DIM_ENV = os.environ.get("EMB_DIM")  # e.g. "1536" (optional)
EMB_DIM     = int(EMB_DIM_ENV) if EMB_DIM_ENV else None

BATCH = int(os.environ.get("EMB_BATCH", "128"))

def read_chunks() -> List[Dict[str, Any]]:
    rows = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            r = json.loads(line)
            r["_row_id"] = i
            rows.append(r)
    return rows

def embed_texts(client: OpenAI, texts: List[str]) -> np.ndarray:
    # OpenAI embeddings endpoint supports batching
    # text-embedding-3-* accepts optional 'dimensions'
    kwargs = {"model": EMB_MODEL, "input": texts}
    if EMB_DIM is not None:
        kwargs["dimensions"] = EMB_DIM
    resp = client.embeddings.create(**kwargs)
    vecs = [d.embedding for d in resp.data]
    return np.array(vecs, dtype="float32")

def main():
    if not CHUNKS_PATH.exists():
        raise SystemExit(f"Missing {CHUNKS_PATH}")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading chunks from {CHUNKS_PATH} …")
    rows = read_chunks()
    texts = [r.get("text") or "" for r in rows]

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    all_vecs = []
    print(f"Embedding {len(texts)} chunks with {EMB_MODEL}"
          f"{f' ({EMB_DIM} dims)' if EMB_DIM else ''} … (batch={BATCH})")

    for i in tqdm(range(0, len(texts), BATCH)):
        batch = texts[i:i+BATCH]
        V = embed_texts(client, batch)
        all_vecs.append(V)

    X = np.vstack(all_vecs)  # (N, d)
    dim = X.shape[1]
    print(f"Vectors shape: {X.shape}")

    # Build FAISS index (L2 on normalized not needed; embeddings are fine for IP or L2)
    index = faiss.IndexFlatIP(dim)
    # Normalize to unit length for cosine via inner product
    faiss.normalize_L2(X)
    index.add(X)

    faiss.write_index(index, str(FAISS_PATH))
    print(f"✅ Saved FAISS: {FAISS_PATH}")

    # Save meta in parallel order
    with open(META_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            meta = {
                "chunk_id": r["chunk_id"],
                "doc_id": r["doc_id"],
                "metadata": r.get("metadata", {}),
                "page": r.get("page_start"),
                "chunk_index": r.get("chunk_index"),
                "title": r.get("metadata", {}).get("title"),
                "source_path": r.get("metadata", {}).get("source_path"),
            }
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    print(f"✅ Saved meta:   {META_PATH}")

    with open(INFO_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "model": EMB_MODEL,
            "dim": dim,
            "normalized": True,
            "backend": "openai",
        }, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved info:   {INFO_PATH}")

if __name__ == "__main__":
    main()
