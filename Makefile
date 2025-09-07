# Makefile — Qdrant-based RAG pipeline
# Usage examples:
#   make help
#   make deps
#   make rebuild
#   make retriever
#   make answer QUESTION="..."

# -------- Paths --------
PY        := .venv/bin/python
PIP       := .venv/bin/pip
RAW_PDFS  := rag_prep/raw_pdfs
DATA_DIR  := rag_prep/data
INDEX_DIR := rag_prep/index

# -------- Qdrant Configuration --------
# Qdrant Cloud (recommended - set these in your environment):
QDRANT_URL        ?= 
QDRANT_API_KEY    ?= 
QDRANT_COLLECTION ?= rag_documents

# For self-hosted Qdrant (only if not using cloud):
QDRANT_HOST       ?= localhost
QDRANT_PORT       ?= 6333

# -------- Models & Parameters --------
# OpenAI embeddings
EMB_MODEL ?= text-embedding-3-large
EMB_DIM   ?=

# Reranker
RERANKER  ?= jinaai/jina-reranker-v2-base-multilingual

# Hybrid search parameters
VEC_TOPK   ?= 100
BM25_TOPK  ?= 150
MERGE_TOPK ?= 50
FINAL_K    ?= 10
RRF_K      ?= 60

# Answering LLM
RAG_CHAT_MODEL ?= gpt-4o-mini

# -------- Default target --------
.PHONY: help
help:
	@echo ""
	@echo "🚀 Qdrant-based RAG Pipeline"
	@echo ""
	@echo "Setup:"
	@echo "  deps            - install dependencies"
	@echo "  qdrant-health   - check Qdrant cloud connection"
	@echo ""
	@echo "Data Pipeline:"
	@echo "  prep            - extract text from PDFs -> $(DATA_DIR)/pages.jsonl"
	@echo "  chunk           - chunk pages -> $(DATA_DIR)/chunks.jsonl"
	@echo "  index           - build Qdrant collection"
	@echo "  rebuild         - prep + chunk + index + bm25"
	@echo ""
	@echo "Services:"
	@echo "  retriever       - start retriever API on :8000"
	@echo "  answer          - one-shot RAG answer (use QUESTION=\"...\")"
	@echo "  answer-api      - start answer API on :8010"
	@echo "  qa              - ask question with auto-start retriever (experimental)"
	@echo "  test-retriever  - test what retriever returns (use QUESTION=\"...\")"
	@echo ""
	@echo "Testing:"
	@echo "  test-search     - test direct Qdrant search"
	@echo "  test-integration - full integration test"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean-index     - remove indexes"
	@echo "  clean-all       - clean everything"
	@echo ""
	@echo "Environment Variables:"
	@echo "  QDRANT_URL=$(QDRANT_URL)"
	@echo "  QDRANT_COLLECTION=$(QDRANT_COLLECTION)"
	@echo "  EMB_MODEL=$(EMB_MODEL) RERANKER=$(RERANKER)"
	@echo "  VEC_TOPK=$(VEC_TOPK) FINAL_K=$(FINAL_K)"
	@echo ""

# -------- Setup --------
.venv:
	python -m venv .venv
	@echo "✅ Virtual environment created"

.PHONY: deps
deps: .venv
	@$(PY) -m pip install --upgrade pip
	@$(PIP) install -r requirements.txt
	@echo "✅ Dependencies installed"

.PHONY: qdrant-health
qdrant-health:
	@echo "🔍 Checking Qdrant connection..."
ifndef QDRANT_URL
	@echo "❌ QDRANT_URL not set. Please set your Qdrant Cloud URL."
	@echo "   Example: export QDRANT_URL='https://your-cluster.qdrant.io'"
	@exit 1
endif
ifndef QDRANT_API_KEY
	@echo "❌ QDRANT_API_KEY not set. Please set your Qdrant API key."
	@exit 1
endif
	@$(PY) -c "from qdrant_store import get_qdrant_store; store = get_qdrant_store('$(QDRANT_COLLECTION)'); print('✅ Qdrant connection successful')" || echo "❌ Qdrant connection failed"

# -------- Data Pipeline --------
.PHONY: prep
prep:
	@test -d $(RAW_PDFS) || (echo "❌ Missing $(RAW_PDFS). Put PDFs there."; exit 1)
	@$(PY) prep_pdfs.py
	@echo "✅ PDF preparation complete -> $(DATA_DIR)/pages.jsonl"

.PHONY: chunk
chunk:
	@$(PY) chunk_texts.py
	@echo "✅ Text chunking complete -> $(DATA_DIR)/chunks.jsonl"

.PHONY: index
index: qdrant-health
ifndef OPENAI_API_KEY
	$(error OPENAI_API_KEY is not set)
endif
	@echo "🚀 Building Qdrant collection '$(QDRANT_COLLECTION)'..."
	@QDRANT_URL="$(QDRANT_URL)" QDRANT_API_KEY="$(QDRANT_API_KEY)" QDRANT_COLLECTION="$(QDRANT_COLLECTION)" \
	 EMB_MODEL="$(EMB_MODEL)" EMB_DIM="$(EMB_DIM)" $(PY) build_qdrant_index.py
	@echo "✅ Qdrant collection built"

.PHONY: bm25
bm25:
	@$(PY) build_bm25.py
	@echo "✅ BM25 index built -> $(INDEX_DIR)/bm25.pkl"

.PHONY: rebuild
rebuild: prep chunk index bm25
	@echo "🎉 Complete rebuild finished!"

.PHONY: quick-rebuild
quick-rebuild: chunk index bm25
	@echo "🎉 Quick rebuild finished!"


# -------- Services --------
.PHONY: retriever
retriever: qdrant-health
	@echo "🚀 Starting Qdrant retriever service..."
	@QDRANT_URL="$(QDRANT_URL)" QDRANT_API_KEY="$(QDRANT_API_KEY)" QDRANT_COLLECTION="$(QDRANT_COLLECTION)" \
	 RERANKER="$(RERANKER)" VEC_TOPK="$(VEC_TOPK)" BM25_TOPK="$(BM25_TOPK)" \
	 MERGE_TOPK="$(MERGE_TOPK)" FINAL_K="$(FINAL_K)" RRF_K="$(RRF_K)" \
	 $(PY) serve_qdrant_retriever.py

QUESTION ?= What is the speed limit?
.PHONY: answer
answer:
ifndef OPENAI_API_KEY
	$(error OPENAI_API_KEY is not set)
endif
	@echo "❓ Asking: $(QUESTION)"
	@RAG_CHAT_MODEL="$(RAG_CHAT_MODEL)" $(PY) answer_rag.py "$(QUESTION)"

.PHONY: answer-api
answer-api:
ifndef OPENAI_API_KEY
	$(error OPENAI_API_KEY is not set)
endif
	@echo "🚀 Starting answer API service..."
	@RAG_CHAT_MODEL="$(RAG_CHAT_MODEL)" $(PY) answer_rag.py serve

.PHONY: test-retriever
test-retriever:
	@echo "🔍 Testing retriever with: $(QUESTION)"
	@curl -s "http://localhost:8000/search?q=$(shell python -c "import urllib.parse; print(urllib.parse.quote('$(QUESTION)'))")" | python -m json.tool

# Experimental: Ask question with auto-start retriever (runs in background)
.PHONY: qa
qa: qdrant-health
ifndef OPENAI_API_KEY
	$(error OPENAI_API_KEY is not set)
endif
	@echo "❓ Quick Q&A: $(QUESTION)"
	@echo "🚀 Starting retriever service in background..."
	@QDRANT_URL="$(QDRANT_URL)" QDRANT_API_KEY="$(QDRANT_API_KEY)" QDRANT_COLLECTION="$(QDRANT_COLLECTION)" \
	 RERANKER="$(RERANKER)" VEC_TOPK="$(VEC_TOPK)" BM25_TOPK="$(BM25_TOPK)" \
	 MERGE_TOPK="$(MERGE_TOPK)" FINAL_K="$(FINAL_K)" RRF_K="$(RRF_K)" \
	 $(PY) serve_qdrant_retriever.py &
	@echo "⏳ Waiting for service to start..."
	@sleep 5
	@echo "💬 Asking question..."
	@RAG_CHAT_MODEL="$(RAG_CHAT_MODEL)" $(PY) answer_rag.py "$(QUESTION)" || true
	@echo "🛑 Stopping background retriever..."
	@pkill -f serve_qdrant_retriever.py || true

# -------- Testing --------
.PHONY: test-search
test-search: qdrant-health
ifndef OPENAI_API_KEY
	$(error OPENAI_API_KEY is not set)
endif
	@echo "🔍 Testing direct Qdrant search..."
	@QDRANT_URL="$(QDRANT_URL)" QDRANT_API_KEY="$(QDRANT_API_KEY)" QDRANT_COLLECTION="$(QDRANT_COLLECTION)" \
	 $(PY) search_once_qdrant.py

.PHONY: test-integration
test-integration: qdrant-health
	@echo "🧪 Running integration tests..."
	@QDRANT_URL="$(QDRANT_URL)" QDRANT_API_KEY="$(QDRANT_API_KEY)" QDRANT_COLLECTION="$(QDRANT_COLLECTION)" \
	 $(PY) test_qdrant_integration.py

# -------- Cleanup --------
.PHONY: clean-index
clean-index:
	@echo "🧹 Removing indexes..."
	@rm -f $(INDEX_DIR)/qdrant_info.json $(INDEX_DIR)/bm25.pkl
	@echo "✅ Local indexes removed"
	@echo "⚠️  Note: Qdrant collection '$(QDRANT_COLLECTION)' still exists"
	@echo "   To remove: make clean-qdrant"

.PHONY: clean-qdrant
clean-qdrant: qdrant-health
	@echo "🧹 Removing Qdrant collection '$(QDRANT_COLLECTION)'..."
	@QDRANT_URL="$(QDRANT_URL)" QDRANT_API_KEY="$(QDRANT_API_KEY)" QDRANT_COLLECTION="$(QDRANT_COLLECTION)" \
	 $(PY) -c "from qdrant_store import get_qdrant_store; get_qdrant_store('$(QDRANT_COLLECTION)').delete_collection()"
	@echo "✅ Qdrant collection removed"

.PHONY: clean-all
clean-all: clean-index clean-qdrant
	@rm -rf $(DATA_DIR)/chunks.jsonl $(DATA_DIR)/pages.jsonl
	@rm -rf rag_prep/tmp/*
	@echo "🧹 Complete cleanup finished"

# -------- Development --------
.PHONY: shell
shell:
	@$(PY)

.PHONY: status
status: qdrant-health
	@echo ""
	@echo "📊 System Status:"
	@echo "  Qdrant: $(QDRANT_URL)"
	@echo "  Collection: $(QDRANT_COLLECTION)"
	@echo "  Data files:"
	@ls -la $(DATA_DIR)/*.jsonl 2>/dev/null || echo "    No data files found"
	@echo "  Index files:"
	@ls -la $(INDEX_DIR)/* 2>/dev/null || echo "    No index files found"
