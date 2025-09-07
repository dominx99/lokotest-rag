import json, math, re, os, uuid
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

# Optional token estimator: tiktoken if available, else char-based
def make_token_sizer():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return lambda s: len(enc.encode(s))
    except Exception:
        # ~4 chars ≈ 1 token (rough), good enough for chunk sizing
        return lambda s: max(1, math.ceil(len(s) / 4))

token_len = make_token_sizer()

# ---------- Config ----------
IN_JSONL = Path("rag_prep/data/pages.jsonl")
OUT_JSONL = Path("rag_prep/data/chunks.jsonl")

# Target chunk size in tokens; overlap retains context across chunks
TARGET_TOKENS = int(os.environ.get("CHUNK_TOKENS", "380"))
OVERLAP_TOKENS = int(os.environ.get("CHUNK_OVERLAP", "60"))

# Treat these as “hard” split boundaries
HARD_BOUNDARY_RE = re.compile(
    r"""(?mx)
    ^\s*                # start of line
    (?:                 # common section numbering / bullets
        \d+(?:\.\d+)*   # 1. 1.2. 1.2.3
        |[IVXLCDM]+\.)  # Roman numerals
    \s+|
    ^\s*[-–•]\s+|       # bullets
    ^\s*(?:Rozdział|Sekcja|Załącznik|Tabela|Rysunek)\b
    """
)

# Sentence-ish splitter tuned for PL punctuation (no heavy NLP deps)
SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[\.\?!…])\s+(?=[A-ZĄĆĘŁŃÓŚŹŻ])"
)

def clean_text(s: str) -> str:
    if not s:
        return ""
    # normalize line endings
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # remove spurious nulls
    s = s.replace("\x00", " ")
    # join hyphenated line breaks: słowo-\nkontynuacja
    s = re.sub(r"([A-Za-zÀ-ÿĄĆĘŁŃÓŚŹŻąęółśżź])-\n([A-Za-zÀ-ÿĄĆĘŁŃÓŚŹŻąęółśżź])", r"\1\2", s)
    # collapse newlines >2 to exactly 2 (keep paragraphs)
    s = re.sub(r"\n{3,}", "\n\n", s)
    # trim extra spaces
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()

def soft_units(paragraph: str) -> List[str]:
    """
    Split paragraph into sentence-like units without requiring NLP models.
    """
    parts = SENTENCE_SPLIT_RE.split(paragraph)
    # merge tiny fragments with neighbors
    merged = []
    buf = ""
    for p in parts:
        if token_len(buf + (" " if buf else "") + p) < 30:
            buf = (buf + " " + p).strip()
        else:
            if buf:
                merged.append(buf)
            buf = p
    if buf:
        merged.append(buf)
    return merged

def semantic_pack(units: List[str], max_tokens: int) -> List[List[str]]:
    """
    Pack units into chunks respecting semantic boundaries (paragraphs).
    """
    chunks, cur, cur_tokens = [], [], 0
    
    i = 0
    while i < len(units):
        u = units[i]
        t = token_len(u)
        
        # If this would exceed the limit and we have content
        if cur and cur_tokens + t > max_tokens:
            # Look ahead to see if there's a natural break point soon
            lookahead = 3
            found_boundary = False
            
            for j in range(i, min(i + lookahead, len(units))):
                if "\n\n" in units[j]:  # paragraph boundary
                    found_boundary = True
                    break
            
            # If we found a boundary soon or current chunk is large enough, split now
            if found_boundary or cur_tokens > max_tokens * 0.6:
                chunks.append(cur)
                cur, cur_tokens = [], 0
        
        cur.append(u)
        cur_tokens += t
        i += 1
    
    if cur:
        chunks.append(cur)
    return chunks

def greedy_pack(units: List[str], max_tokens: int) -> List[List[str]]:
    """
    Pack units into chunks not exceeding max_tokens, respecting paragraph boundaries.
    """
    # Use semantic packing by default
    return semantic_pack(units, max_tokens)

def sliding_overlap(chunks: List[List[str]], overlap_tokens: int) -> List[str]:
    out = []
    prev_tail = ""
    for i, ch in enumerate(chunks):
        body = " ".join(ch).strip()
        # prepend tail overlap from previous
        if prev_tail:
            body = (prev_tail + " " + body).strip()
        out.append(body)
        # compute new tail from current for next
        if overlap_tokens > 0:
            units = ch.copy()
            # take from end until we reach desired overlap
            tail = []
            total = 0
            for u in reversed(units):
                t = token_len(u)
                tail.append(u)
                total += t
                if total >= overlap_tokens:
                    break
            prev_tail = " ".join(reversed(tail))
        else:
            prev_tail = ""
    return out

def page_to_chunks(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    text = clean_text(rec.get("text",""))
    if not text:
        return []

    # First, split by hard boundaries while preserving order
    # We split on lines so boundaries match at line starts.
    lines = text.split("\n")
    blocks, buf = [], []
    for ln in lines:
        if HARD_BOUNDARY_RE.search(ln):
            if buf:
                blocks.append("\n".join(buf).strip())
                buf = []
        buf.append(ln)
    if buf:
        blocks.append("\n".join(buf).strip())

    # Further split blocks into paragraph-aware units
    units: List[str] = []
    for b in blocks:
        # split by double newline (paragraphs)
        paras = [p.strip() for p in b.split("\n\n") if p.strip()]
        for i, p in enumerate(paras):
            # Keep paragraph markers for boundary detection
            paragraph_units = soft_units(p)
            # Add paragraph separator to last unit (except for last paragraph)
            if paragraph_units and i < len(paras) - 1:
                paragraph_units[-1] += "\n\n"
            units.extend(paragraph_units)

    # Pack into chunks with overlap
    packed = greedy_pack(units, TARGET_TOKENS)
    texts = sliding_overlap(packed, OVERLAP_TOKENS)

    # Build records
    out = []
    for idx, chunk_text in enumerate(texts):
        out.append({
            "chunk_id": f"{rec['doc_id']}-{rec['page']:04d}-{idx:03d}-{uuid.uuid4().hex[:6]}",
            "doc_id": rec["doc_id"],
            "page_start": rec["page"],
            "page_end": rec["page"],   # page-level for now; later we can merge multi-page
            "chunk_index": idx,
            "text": chunk_text,
            "metadata": {
                **rec.get("metadata", {}),
                "source_page": rec["page"],
                "source_path": rec.get("metadata", {}).get("source_path"),
                "title": rec.get("metadata", {}).get("title"),
                "lang_hint": "pl",  # helpful for downstream LLMs
            }
        })
    return out

def main():
    if not IN_JSONL.exists():
        raise SystemExit(f"Input not found: {IN_JSONL}. Run Step 1 first.")
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    total_pages = 0
    total_chunks = 0
    with open(IN_JSONL, "r", encoding="utf-8") as f_in, \
         open(OUT_JSONL.with_suffix(".tmp"), "w", encoding="utf-8") as f_out:
        for line in tqdm(f_in, desc="Chunking pages"):
            rec = json.loads(line)
            pieces = page_to_chunks(rec)
            total_pages += 1
            total_chunks += len(pieces)
            for p in pieces:
                f_out.write(json.dumps(p, ensure_ascii=False) + "\n")

    os.replace(OUT_JSONL.with_suffix(".tmp"), OUT_JSONL)
    print(f"✅ Wrote {OUT_JSONL}")
    print(f"📄 Pages processed: {total_pages}")
    print(f"🔹 Chunks created: {total_chunks}")
    if total_chunks:
        avg = round(total_chunks / max(1,total_pages), 2)
        print(f"📏 Avg chunks per page: {avg}")
        print(f"ℹ️ You can tune size via CHUNK_TOKENS (default {TARGET_TOKENS}) and CHUNK_OVERLAP (default {OVERLAP_TOKENS}).")

if __name__ == "__main__":
    main()
