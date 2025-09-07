# LokotestRAG - End-to-End RAG Pipeline

A comprehensive Retrieval-Augmented Generation (RAG) system that processes PDF documents and provides intelligent question-answering capabilities using hybrid retrieval (vector + BM25) and OpenAI's language models.

## Features

- **PDF Processing**: Extract text and images from PDFs with OCR fallback
- **Text Chunking**: Intelligent document segmentation for optimal retrieval
- **Hybrid Retrieval**: Combines FAISS vector search with BM25 keyword search
- **Reranking**: Multilingual document reranking for improved relevance
- **Question Answering**: OpenAI-powered conversational responses
- **APIs**: FastAPI-based retriever and answering services

## Quick Start

```bash
# Setup environment and dependencies
make venv deps

# Build the complete pipeline (place PDFs in rag_prep/raw_pdfs/ first)
make prep chunk embed bm25

# Start services
make retriever      # Retrieval API on :8000
make answer-api     # Q&A API on :8010

# Test single query
make answer QUESTION="Your question here"
```

## Prerequisites

- Python 3.8+
- OpenAI API key (set `OPENAI_API_KEY` environment variable)
- PDFs placed in `rag_prep/raw_pdfs/` directory

## Pipeline Components

### 1. Document Preparation (`make prep`)
Extracts text and images from PDFs, saving structured data to `rag_prep/data/pages.jsonl`

### 2. Text Chunking (`make chunk`)
Segments documents into retrievable chunks, output: `rag_prep/data/chunks.jsonl`

### 3. Vector Indexing (`make embed`)
Builds FAISS index using OpenAI embeddings, creates `rag_prep/index/faiss.index`

### 4. BM25 Indexing (`make bm25`)
Creates keyword search index with number/unit awareness, saves to `rag_prep/index/bm25.pkl`

## Configuration

Override default settings via environment variables or make arguments:

```bash
# Embedding model
make embed EMB_MODEL=text-embedding-3-small

# Retrieval parameters
make retriever VEC_TOPK=50 BM25_TOPK=100 FINAL_K=10

# Answer model
make answer RAG_CHAT_MODEL=gpt-4
```

### Key Parameters

- **EMB_MODEL**: OpenAI embedding model (default: `text-embedding-3-large`)
- **RERANKER**: Reranking model (default: `Alibaba-NLP/gte-multilingual-reranker-base`)
- **VEC_TOPK**: Vector search results (default: 100)
- **BM25_TOPK**: BM25 search results (default: 150)
- **FINAL_K**: Final reranked results (default: 8)
- **RAG_CHAT_MODEL**: Chat model (default: `gpt-5-mini`)

## API Usage

### Retriever Service (Port 8000)
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "Your question", "k": 5}'
```

### Answer Service (Port 8010)
```bash
curl -X POST "http://localhost:8010/answer" \
  -H "Content-Type: application/json" \
  -d '{"question": "Your question"}'
```

## Available Commands

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make venv` | Create Python virtual environment |
| `make deps` | Install dependencies |
| `make prep` | Extract text/images from PDFs |
| `make chunk` | Chunk pages into retrievable segments |
| `make embed` | Build FAISS vector index |
| `make bm25` | Build BM25 keyword index |
| `make rebuild` | Full pipeline rebuild (prep + chunk + embed + bm25) |
| `make quick-rebuild` | Skip PDF processing (chunk + embed + bm25) |
| `make retriever` | Start retrieval API server |
| `make retriever-once` | Single CLI retrieval test |
| `make answer` | One-shot question answering |
| `make answer-api` | Start Q&A API server |
| `make clean-index` | Remove generated indexes |

## Directory Structure

```
rag_prep/
├── raw_pdfs/          # Input PDF documents
├── data/              # Processed data
│   ├── pages.jsonl    # Extracted text/images
│   └── chunks.jsonl   # Text chunks
└── index/             # Search indexes
    ├── faiss.index    # Vector index
    ├── bm25.pkl       # BM25 index
    └── index_info.json # Index metadata
```

## Dependencies

The system automatically installs required Python packages including:
- PyMuPDF, pdf2image, pytesseract (PDF processing)
- sentence-transformers, faiss-cpu (vector search)
- rank-bm25 (keyword search)
- fastapi, uvicorn (web APIs)
- openai (language models)
- torch, numpy, scikit-learn (ML libraries)