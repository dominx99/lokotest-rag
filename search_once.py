import json, faiss, numpy as np, torch, os
from pathlib import Path
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path("rag_prep/index")
META = INDEX_DIR / "meta.jsonl"
INDEX = INDEX_DIR / "faiss.index"
INFO = INDEX_DIR / "index_info.json"
MODEL_NAME = os.environ.get("EMB_MODEL", json.loads(open(INFO).read())["model"])

def load_meta():
    metas = []
    with open(META, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            m = json.loads(line)
            m["_row_id"] = i
            metas.append(m)
    return metas

def main():
    metas = load_meta()
    index = faiss.read_index(str(INDEX))
    model = SentenceTransformer(MODEL_NAME, device="cuda" if torch.cuda.is_available() else "cpu")

    q = input("Wpisz zapytanie (po polsku): ").strip()
    q_emb = model.encode([f"query: {q}"], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    scores, ids = index.search(q_emb, 5)
    for rank, (idx, score) in enumerate(zip(ids[0], scores[0]), 1):
        m = metas[idx]
        print(f"\n[{rank}] score={float(score):.3f}")
        print(f"   tytuł: {m.get('title')}")
        print(f"   plik:  {m.get('source_path')}")
        print(f"   str.:  {m.get('page_start')}")
        print(f"   chunk: {m.get('chunk_index')}  id: {m.get('chunk_id')}")
    print("\nDone.")

if __name__ == "__main__":
    main()
