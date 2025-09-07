# build_bm25.py — BM25 index with number/unit-aware tokenizer (PL)
import json
import pickle
from pathlib import Path
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import regex as re  # supports \p{...}

CHUNKS = Path("rag_prep/data/chunks.jsonl")
OUT = Path("rag_prep/index/bm25.pkl")

# Keep letters, numbers (incl. decimals), and km/h as a single token
TOKEN_RE = re.compile(r"(?:\p{L}+|\d+(?:[.,]\d+)?|km/?h)", re.UNICODE | re.IGNORECASE)

def normalize_units(s: str) -> str:
    if not s:
        return ""
    # unify common variations of km/h
    s = s.replace("KM / H", "km/h").replace("KM/ H", "km/h").replace("KM /H", "km/h")
    s = s.replace("km / h", "km/h").replace("km/ h", "km/h").replace("km /h", "km/h")
    return s

def tokenize_pl(s: str):
    s = normalize_units(s)
    return [m.group(0).lower() for m in TOKEN_RE.finditer(s)]

def main():
    if not CHUNKS.exists():
        raise SystemExit(f"Input not found: {CHUNKS}")
    docs = []
    meta = []
    with open(CHUNKS, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading chunks"):
            if not line.strip():
                continue
            rec = json.loads(line)
            docs.append(tokenize_pl(rec.get("text", "")))
            meta.append({
                "chunk_id": rec["chunk_id"],
                "doc_id": rec["doc_id"],
                "title": rec.get("metadata", {}).get("title"),
                "source_path": rec.get("metadata", {}).get("source_path"),
                "page": rec.get("page_start"),
                "chunk_index": rec.get("chunk_index"),
            })
    bm25 = BM25Okapi(docs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "wb") as f:
        pickle.dump({"bm25": bm25, "meta": meta}, f)
    print("✅ Saved", OUT)

if __name__ == "__main__":
    main()
