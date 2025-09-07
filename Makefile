# Makefile — Recreate the full RAG pipeline end-to-end
# Usage examples:
#   make help
#   make venv deps
#   make prep chunk embed bm25          # full (re)build
#   make retriever                      # start retriever API (port 8000)
#   make retriever-once                 # single interactive query in CLI
#   make answer QUESTION="Posterunek techniczny"
#   make answer-api                     # start answer API (port 8010)

# -------- Paths --------
PY        := .venv/bin/python
PIP       := .venv/bin/pip
RAW_PDFS  := rag_prep/raw_pdfs
DATA_DIR  := rag_prep/data
INDEX_DIR := rag_prep/index

# -------- Models & knobs (override with: make embed EMB_MODEL=... ) --------
# OpenAI embeddings (used by build_index.py)
EMB_MODEL ?= text-embedding-3-large
# EMB_DIM is optional; if set, must be valid for the chosen model (e.g., 3072 or 1536)
EMB_DIM   ?=

# Reranker (no-custom-code multilingual by default)
RERANKER  ?= Alibaba-NLP/gte-multilingual-reranker-base
# If you prefer Jina (requires trust_remote_code): RERANKER=jinaai/jina-reranker-v2-base-multilingual

# Hybrid recall (increase if you miss “needle” facts like 20 km/h)
VEC_TOPK   ?= 100
BM25_TOPK  ?= 150
MERGE_TOPK ?= 40
FINAL_K    ?= 8
RRF_K      ?= 60

# Answering LLM (OpenAI Chat)
RAG_CHAT_MODEL ?= gpt-5-mini

# -------- Default target --------
.PHONY: help
help:
	@echo ""
	@echo "Targets:"
	@echo "  venv            - create Python venv (.venv)"
	@echo "  deps            - install Python dependencies into venv"
	@echo "  prep            - extract text/images from PDFs  -> $(DATA_DIR)/pages.jsonl"
	@echo "  chunk           - chunk pages                     -> $(DATA_DIR)/chunks.jsonl"
	@echo "  embed           - build FAISS (OpenAI embeddings) -> $(INDEX_DIR)/faiss.index"
	@echo "  bm25            - build BM25 (tokenizer keeps numbers & km/h) -> $(INDEX_DIR)/bm25.pkl"
	@echo "  rebuild         - prep + chunk + embed + bm25"
	@echo "  quick-rebuild   - chunk + embed + bm25 (if PDFs didn’t change)"
	@echo "  retriever       - run retriever API on :8000"
	@echo "  retriever-once  - one-shot CLI retrieval test"
	@echo "  answer          - run one-shot RAG answer (use QUESTION=\"...\")"
	@echo "  answer-api      - run answer API on :8010"
	@echo "  clean-index     - remove vector/BM25 indexes"
	@echo ""
	@echo "Variables (override via 'make target VAR=value'):"
	@echo "  EMB_MODEL=$(EMB_MODEL)  EMB_DIM=$(EMB_DIM)"
	@echo "  RERANKER=$(RERANKER)"
	@echo "  VEC_TOPK=$(VEC_TOPK)  BM25_TOPK=$(BM25_TOPK)  MERGE_TOPK=$(MERGE_TOPK)  FINAL_K=$(FINAL_K)  RRF_K=$(RRF_K)"
	@echo "  RAG_CHAT_MODEL=$(RAG_CHAT_MODEL)"
	@echo ""
	@echo "Note: You don't need 'source .venv/bin/activate.fish' inside Make; we call $(PY)/$(PIP) directly."

# -------- Environment / deps --------
.PHONY: venv
venv:
	@test -d .venv || python -m venv .venv
	@echo "✅ venv ready: .venv"

.PHONY: deps
deps: venv
	@$(PY) -m pip install --upgrade pip
	@echo "📦 writing requirements.txt"
	@echo "python-dateutil" > requirements.txt
	@echo "pymupdf" >> requirements.txt
	@echo "pdf2image" >> requirements.txt
	@echo "pytesseract" >> requirements.txt
	@echo "Pillow" >> requirements.txt
	@echo "tqdm" >> requirements.txt
	@echo "regex" >> requirements.txt
	@echo "tiktoken" >> requirements.txt
	@echo "sentence-transformers>=3.0" >> requirements.txt
	@echo "faiss-cpu" >> requirements.txt
	@echo "torch>=2.2" >> requirements.txt
	@echo "rank-bm25" >> requirements.txt
	@echo "fastapi" >> requirements.txt
	@echo "uvicorn" >> requirements.txt
	@echo "pydantic" >> requirements.txt
	@echo "numpy" >> requirements.txt
	@echo "scikit-learn" >> requirements.txt
	@echo "scipy" >> requirements.txt
	@echo "httpx" >> requirements.txt
	@echo "openai" >> requirements.txt
	@echo "einops" >> requirements.txt
	@echo "sentencepiece" >> requirements.txt
	@$(PIP) install -r requirements.txt
	@echo "✅ deps installed"

# -------- Pipeline --------
# 1) Prepare PDFs -> pages.jsonl (and page images for OCR fallback)
.PHONY: prep
prep:
	@test -d $(RAW_PDFS) || (echo "❌ Missing $(RAW_PDFS). Put PDFs there."; exit 1)
	@$(PY) prep_pdfs.py
	@echo "✅ prep done -> $(DATA_DIR)/pages.jsonl"

# 2) Chunk pages -> chunks.jsonl
.PHONY: chunk
chunk:
	@$(PY) chunk_texts.py
	@echo "✅ chunk done -> $(DATA_DIR)/chunks.jsonl"

# 3) Build FAISS with OpenAI embeddings (reads chunks.jsonl)
.PHONY: embed
embed:
ifndef OPENAI_API_KEY
	$(error OPENAI_API_KEY is not set)
endif
	@EMB_MODEL="$(EMB_MODEL)" EMB_DIM="$(EMB_DIM)" $(PY) build_index.py
	@echo "✅ FAISS built -> $(INDEX_DIR)/faiss.index"

# 4) Build BM25 with number/unit-aware tokenizer
.PHONY: bm25
bm25:
	@$(PY) build_bm25.py
	@echo "✅ BM25 built -> $(INDEX_DIR)/bm25.pkl"

# Convenience bundles
.PHONY: rebuild
rebuild: prep chunk embed bm25
	@echo "✅ Rebuild complete."

.PHONY: quick-rebuild
quick-rebuild: chunk embed bm25
	@echo "✅ Quick rebuild complete."

# -------- Serving / Testing --------
# Start retriever API (FastAPI on :8000)
.PHONY: retriever
retriever:
	@RERANKER="$(RERANKER)" \
	VEC_TOPK="$(VEC_TOPK)" BM25_TOPK="$(BM25_TOPK)" MERGE_TOPK="$(MERGE_TOPK)" FINAL_K="$(FINAL_K)" RRF_K="$(RRF_K)" \
	$(PY) serve_retriever.py

# One-shot CLI retrieval test (prompts for 'Zapytanie:')
.PHONY: retriever-once
retriever-once:
	@RERANKER="$(RERANKER)" \
	VEC_TOPK="$(VEC_TOPK)" BM25_TOPK="$(BM25_TOPK)" MERGE_TOPK="$(MERGE_TOPK)" FINAL_K="$(FINAL_K)" RRF_K="$(RRF_K)" \
	$(PY) serve_retriever.py --once

# One-shot answer (requires retriever running separately)
QUESTION ?= Posterunek techniczny
.PHONY: answer
answer:
ifndef OPENAI_API_KEY
	$(error OPENAI_API_KEY is not set)
endif
	@RAG_CHAT_MODEL="$(RAG_CHAT_MODEL)" $(PY) answer_rag.py "$(QUESTION)"

# Run answer API (FastAPI on :8010)
.PHONY: answer-api
answer-api:
ifndef OPENAI_API_KEY
	$(error OPENAI_API_KEY is not set)
endif
	@RAG_CHAT_MODEL="$(RAG_CHAT_MODEL)" $(PY) answer_rag.py serve

# -------- Cleanup --------
.PHONY: clean-index
clean-index:
	@rm -f $(INDEX_DIR)/faiss.index $(INDEX_DIR)/bm25.pkl $(INDEX_DIR)/index_info.json
	@echo "🧹 Removed indexes in $(INDEX_DIR)"
