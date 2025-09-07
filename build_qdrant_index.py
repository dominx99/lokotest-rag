# build_qdrant_index.py — Build Qdrant collection using OpenAI embeddings
import os
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from tqdm import tqdm
from openai import OpenAI

from qdrant_store import get_qdrant_store

CHUNKS_PATH = Path("rag_prep/data/chunks.jsonl")
INDEX_DIR   = Path("rag_prep/index")
INFO_PATH   = INDEX_DIR / "qdrant_info.json"

EMB_MODEL   = os.environ.get("EMB_MODEL", "text-embedding-3-large")
EMB_DIM_ENV = os.environ.get("EMB_DIM")
EMB_DIM     = int(EMB_DIM_ENV) if EMB_DIM_ENV else None

BATCH = int(os.environ.get("EMB_BATCH", "128"))
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "rag_documents")


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

    # Generate embeddings
    for i in tqdm(range(0, len(texts), BATCH)):
        batch = texts[i:i+BATCH]
        V = embed_texts(client, batch)
        all_vecs.append(V)

    X = np.vstack(all_vecs)  # (N, d)
    dim = X.shape[1]
    print(f"Vectors shape: {X.shape}")

    # Normalize vectors for cosine similarity
    from sklearn.preprocessing import normalize
    X = normalize(X, norm='l2', axis=1)

    # Initialize Qdrant store
    qdrant_store = get_qdrant_store(COLLECTION_NAME)
    
    # Create collection
    qdrant_store.create_collection(vector_size=dim)

    # Prepare payloads
    payloads = []
    chunk_ids = []
    for r in rows:
        payload = {
            "chunk_id": r["chunk_id"],
            "doc_id": r["doc_id"],
            "text": r.get("text", ""),
            "metadata": r.get("metadata", {}),
            "page": r.get("page_start"),
            "chunk_index": r.get("chunk_index"),
            "title": r.get("metadata", {}).get("title"),
            "source_path": r.get("metadata", {}).get("source_path"),
        }
        payloads.append(payload)
        chunk_ids.append(r["chunk_id"])

    # Upsert vectors to Qdrant (UUIDs will be generated automatically)
    print(f"Uploading {len(X)} vectors to Qdrant collection '{COLLECTION_NAME}' ...")
    qdrant_store.upsert_vectors(X, payloads, chunk_ids)

    # Save collection info
    collection_info = qdrant_store.get_collection_info()
    info_data = {
        "model": EMB_MODEL,
        "dim": dim,
        "normalized": True,
        "backend": "qdrant",
        "collection_name": COLLECTION_NAME,
        "points_count": collection_info.get("points_count", len(X)),
        "vector_size": collection_info.get("config", {}).get("vector_size", dim)
    }
    
    with open(INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(info_data, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved info: {INFO_PATH}")
    
    print(f"✅ Qdrant collection '{COLLECTION_NAME}' ready with {collection_info.get('points_count', len(X))} points")


if __name__ == "__main__":
    main()