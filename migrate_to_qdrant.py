#!/usr/bin/env python3
# migrate_to_qdrant.py — Migrate existing FAISS index to Qdrant
import os
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import faiss
from tqdm import tqdm

from qdrant_store import get_qdrant_store

# Paths
INDEX_DIR = Path("rag_prep/index")
FAISS_PATH = INDEX_DIR / "faiss.index"
META_PATH = INDEX_DIR / "meta.jsonl"
CHUNKS_PATH = Path("rag_prep/data/chunks.jsonl")
FAISS_INFO_PATH = INDEX_DIR / "index_info.json"
QDRANT_INFO_PATH = INDEX_DIR / "qdrant_info.json"

COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "rag_documents")


def load_faiss_index():
    """Load FAISS index and metadata."""
    if not FAISS_PATH.exists():
        raise FileNotFoundError(f"FAISS index not found: {FAISS_PATH}")
    
    print(f"Loading FAISS index from {FAISS_PATH}")
    index = faiss.read_index(str(FAISS_PATH))
    
    # Load metadata
    meta_items = []
    with open(META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                meta_items.append(json.loads(line))
    
    return index, meta_items


def load_chunk_texts():
    """Load original chunk texts."""
    chunks = {}
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                chunk = json.loads(line)
                chunk["_row_id"] = i
                chunks[i] = chunk
    return chunks


def migrate_to_qdrant():
    """Migrate FAISS index to Qdrant."""
    
    # Load FAISS data
    faiss_index, meta_items = load_faiss_index()
    chunk_texts = load_chunk_texts()
    
    print(f"FAISS index contains {faiss_index.ntotal} vectors")
    print(f"Loaded {len(meta_items)} metadata items")
    print(f"Loaded {len(chunk_texts)} chunk texts")
    
    # Extract vectors from FAISS
    print("Extracting vectors from FAISS...")
    all_vectors = []
    batch_size = 1000
    
    for i in tqdm(range(0, faiss_index.ntotal, batch_size)):
        end_idx = min(i + batch_size, faiss_index.ntotal)
        batch_vectors = faiss_index.reconstruct_batch(range(i, end_idx))
        all_vectors.append(batch_vectors)
    
    vectors_matrix = np.vstack(all_vectors)
    print(f"Extracted vectors shape: {vectors_matrix.shape}")
    
    # Initialize Qdrant
    qdrant_store = get_qdrant_store(COLLECTION_NAME)
    
    # Create collection
    vector_size = vectors_matrix.shape[1]
    qdrant_store.create_collection(vector_size=vector_size)
    
    # Prepare payloads
    print("Preparing payloads...")
    payloads = []
    ids = []
    
    for i, meta in enumerate(tqdm(meta_items)):
        chunk_id = meta["chunk_id"]
        
        # Get text from chunks
        chunk_data = chunk_texts.get(i, {})
        text = chunk_data.get("text", "")
        
        payload = {
            "chunk_id": chunk_id,
            "doc_id": meta["doc_id"],
            "text": text,
            "metadata": chunk_data.get("metadata", {}),
            "page": meta.get("page"),
            "chunk_index": meta.get("chunk_index"),
            "title": meta.get("title"),
            "source_path": meta.get("source_path"),
        }
        
        payloads.append(payload)
        ids.append(chunk_id)
    
    # Upload to Qdrant (UUIDs will be generated automatically)
    print(f"Uploading {len(vectors_matrix)} vectors to Qdrant...")
    qdrant_store.upsert_vectors(vectors_matrix, payloads, ids)
    
    # Save Qdrant info
    with open(FAISS_INFO_PATH, "r") as f:
        faiss_info = json.load(f)
    
    qdrant_info = {
        "model": faiss_info["model"],
        "dim": faiss_info["dim"],
        "normalized": faiss_info["normalized"],
        "backend": "qdrant",
        "collection_name": COLLECTION_NAME,
        "migrated_from": "faiss",
        "points_count": len(vectors_matrix),
        "vector_size": vector_size
    }
    
    with open(QDRANT_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(qdrant_info, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Migration complete!")
    print(f"✅ Qdrant collection '{COLLECTION_NAME}' created with {len(vectors_matrix)} points")
    print(f"✅ Info saved to {QDRANT_INFO_PATH}")
    
    # Verify migration
    collection_info = qdrant_store.get_collection_info()
    print(f"✅ Verification: Collection has {collection_info.get('points_count', 0)} points")


def verify_migration():
    """Verify that migration worked correctly."""
    print("\nVerifying migration...")
    
    qdrant_store = get_qdrant_store(COLLECTION_NAME)
    collection_info = qdrant_store.get_collection_info()
    
    print(f"Collection info: {collection_info}")
    
    # Test search
    try:
        dummy_vector = np.random.rand(collection_info["config"]["vector_size"]).astype(np.float32)
        dummy_vector = dummy_vector / np.linalg.norm(dummy_vector)  # normalize
        
        results = qdrant_store.search(dummy_vector, limit=3)
        print(f"✅ Search test successful - found {len(results)} results")
        
        if results:
            print(f"Sample result: {results[0]['chunk_id']}")
    
    except Exception as e:
        print(f"❌ Search test failed: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        verify_migration()
    else:
        migrate_to_qdrant()
        verify_migration()