import json, os, re, uuid, shutil
from pathlib import Path
from datetime import datetime
from dateutil.parser import parse as parse_date

import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
from tqdm import tqdm

RAW_DIR = Path("rag_prep/raw_pdfs")
OUT_JSONL = Path("rag_prep/data/pages.jsonl")
OUT_IMAGES = Path("rag_prep/data/page_images")
TMP_DIR = Path("rag_prep/tmp")

OCR_LANGS = os.environ.get("OCR_LANGS", "pol")

for p in [OUT_JSONL.parent, OUT_IMAGES, TMP_DIR]:
    p.mkdir(parents=True, exist_ok=True)

def clean_ws(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def get_pdf_meta(doc: fitz.Document):
    md = doc.metadata or {}
    def norm(d, key, default=""):
        v = d.get(key) or default
        return v.strip() if isinstance(v, str) else v
    # Try to parse creation/mod dates
    def try_date(v):
        try:
            return parse_date(v).isoformat()
        except Exception:
            return None
    return {
        "title": norm(md, "title", ""),
        "author": norm(md, "author", ""),
        "subject": norm(md, "subject", ""),
        "keywords": norm(md, "keywords", ""),
        "creator": norm(md, "creator", ""),
        "producer": norm(md, "producer", ""),
        "creationDate": try_date(md.get("creationDate") or ""),
        "modDate": try_date(md.get("modDate") or ""),
        "page_count": doc.page_count,
    }

def page_has_text(page: fitz.Page) -> bool:
    txt = page.get_text("text", flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_INHIBIT_SPACES)
    return bool(txt and txt.strip())

def ocr_pdf_page(pdf_path: Path, page_index: int, dpi=300):
    images = convert_from_path(
        str(pdf_path),
        first_page=page_index+1,
        last_page=page_index+1,
        dpi=dpi,
        fmt="png",
        output_folder=TMP_DIR
    )
    if not images:
        return "", None
    img: Image.Image = images[0]
    text = pytesseract.image_to_string(img, lang=OCR_LANGS)
    return text, img

def save_page_image(image: Image.Image, doc_id: str, page_index: int):
    out = OUT_IMAGES / f"{doc_id}_p{page_index+1:04d}.png"
    image.save(out)
    return str(out)

def infer_title(fallback: str, path: Path):
    # Prefer PDF embedded title; else filename without extension
    return fallback or path.stem

def make_doc_id(path: Path):
    # Stable-ish doc id based on filename + a short uuid for collisions
    return f"{path.stem.lower().replace(' ','_')}-{uuid.uuid4().hex[:8]}"

def process_pdf(pdf_path: Path):
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Failed to open {pdf_path}: {e}")
        return []

    meta = get_pdf_meta(doc)
    doc_id = make_doc_id(pdf_path)
    title = infer_title(meta.get("title",""), pdf_path)

    recs = []
    for i in tqdm(range(doc.page_count), desc=f"Pages {pdf_path.name}"):
        page = doc.load_page(i)
        if page_has_text(page):
            txt = page.get_text("text", flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_INHIBIT_SPACES)
            txt = clean_ws(txt)
            img_path = None
            ocr_used = False
        else:
            # OCR fallback
            txt_raw, img = ocr_pdf_page(pdf_path, i)
            txt = clean_ws(txt_raw)
            img_path = save_page_image(img, doc_id, i) if img else None
            ocr_used = True

        rec = {
            "doc_id": doc_id,
            "page": i + 1,
            "text": txt,
            "metadata": {
                "source_path": str(pdf_path),
                "source_type": "pdf",
                "title": title,
                "author": meta.get("author"),
                "subject": meta.get("subject"),
                "keywords": meta.get("keywords"),
                "creator": meta.get("creator"),
                "producer": meta.get("producer"),
                "creationDate": meta.get("creationDate"),
                "modDate": meta.get("modDate"),
                "page_count": meta.get("page_count"),
                "ocr_used": ocr_used,
                "page_image": img_path,
            }
        }
        # Skip completely empty pages to keep your corpus clean
        if rec["text"]:
            recs.append(rec)

    doc.close()
    return recs

def main():
    pdfs = sorted([p for p in RAW_DIR.glob("**/*.pdf") if p.is_file()])
    if not pdfs:
        print(f"No PDFs in {RAW_DIR}. Add files and re-run.")
        return

    # Write/append atomically
    tmp_out = OUT_JSONL.with_suffix(".jsonl.tmp")
    with open(tmp_out, "w", encoding="utf-8") as f:
        for pdf in pdfs:
            recs = process_pdf(pdf)
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    shutil.move(tmp_out, OUT_JSONL)
    print(f"✅ Wrote {OUT_JSONL} with per-page records.")
    print(f"🖼️ Page images (for OCR’d pages) at {OUT_IMAGES}/")

if __name__ == "__main__":
    main()
