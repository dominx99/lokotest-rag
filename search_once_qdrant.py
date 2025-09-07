import json
import numpy as np
import torch
import os
from pathlib import Path
from openai import OpenAI
from qdrant_store import get_qdrant_store

INDEX_DIR = Path("rag_prep/index")
INFO = INDEX_DIR / "qdrant_info.json"
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "rag_documents")

def load_collection_info():
    """Load collection information."""
    with open(INFO, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    # Load collection info
    info = load_collection_info()
    EMB_MODEL = info["model"]
    
    # Initialize Qdrant store
    qdrant_store = get_qdrant_store(COLLECTION_NAME)
    
    # Initialize OpenAI client for embeddings
    oa_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    def embed_query(q: str) -> np.ndarray:
        kwargs = {"model": EMB_MODEL, "input": [q]}
        resp = oa_client.embeddings.create(**kwargs)
        v = np.array(resp.data[0].embedding, dtype="float32")
        # Normalize to unit length for cosine similarity
        v = v / np.linalg.norm(v)
        return v

    q = input("Wpisz zapytanie (po polsku): ").strip()
    
    # Embed query
    q_emb = embed_query(q)
    
    # Search Qdrant
    results = qdrant_store.search(query_vector=q_emb, limit=5)
    
    # Display results
    for rank, result in enumerate(results, 1):
        print(f"\n[{rank}] score={float(result['score']):.3f}")
        print(f"   tytuł: {result.get('title')}")
        print(f"   plik:  {result.get('source_path')}")
        print(f"   str.:  {result.get('page')}")
        print(f"   chunk: {result.get('chunk_index')}  id: {result.get('chunk_id')}")
        print("   ---")
        text = result.get('text', '')
        if text:
            text = text[:600].replace('\n', ' ')
            print(text)
        else:
            print("[No text available]")
    
    print("\nDone.")

if __name__ == "__main__":
    main()