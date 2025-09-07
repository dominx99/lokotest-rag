# LokotestRAG - Qdrant-Powered RAG Pipeline

A comprehensive Retrieval-Augmented Generation (RAG) system that processes PDF documents and provides intelligent question-answering capabilities using Qdrant vector database, BM25 keyword search, and OpenAI's language models.

## 🚀 Features

- **PDF Processing**: Extract text and images from PDFs with OCR fallback
- **Text Chunking**: Intelligent document segmentation for optimal retrieval
- **Qdrant Integration**: Cloud-native vector database for scalable search
- **Hybrid Retrieval**: Combines Qdrant vector search with BM25 keyword search
- **Reranking**: Multilingual document reranking for improved relevance
- **Enhanced Q&A**: Refactored answering system with specialized question handling
- **Cloud Ready**: Full Qdrant Cloud integration

## ⚡ Quick Start

```bash
# 1. Set up environment variables
export QDRANT_URL="https://your-cluster.qdrant.io"
export QDRANT_API_KEY="your-api-key" 
export OPENAI_API_KEY="your-openai-key"

# 2. Install dependencies
make deps

# 3. Build the complete pipeline (place PDFs in rag_prep/raw_pdfs/ first)
make rebuild

# 4. Start services
make retriever      # Retrieval API on :8000
make answer-api     # Q&A API on :8010

# 5. Test single query
make answer QUESTION="Your question here"
```

## 🏗️ Prerequisites

- **Python 3.8+**
- **Qdrant Cloud Account**: Get from [cloud.qdrant.io](https://cloud.qdrant.io)
- **OpenAI API Key**: Set `OPENAI_API_KEY` environment variable
- **PDFs**: Place documents in `rag_prep/raw_pdfs/` directory

## 📋 Pipeline Components

### 1. Document Preparation (`make prep`)
Extracts text and images from PDFs, saving structured data to `rag_prep/data/pages.jsonl`

### 2. Text Chunking (`make chunk`)
Segments documents into retrievable chunks, output: `rag_prep/data/chunks.jsonl`

### 3. Qdrant Indexing (`make index`)
Builds Qdrant collection using OpenAI embeddings, stores vectors in cloud

### 4. BM25 Indexing (`make bm25`)
Creates keyword search index with number/unit awareness, saves to `rag_prep/index/bm25.pkl`

## ⚙️ Configuration

### Required Environment Variables
```bash
export QDRANT_URL="https://your-cluster.qdrant.io"
export QDRANT_API_KEY="your-qdrant-api-key"
export OPENAI_API_KEY="your-openai-api-key"
```

### Optional Configuration
```bash
# Collection settings
export QDRANT_COLLECTION="rag_documents"

# Embedding model
export EMB_MODEL="text-embedding-3-small"

# Retrieval parameters  
export VEC_TOPK=100
export BM25_TOPK=150
export FINAL_K=8

# Answer model
export RAG_CHAT_MODEL="gpt-5-mini"
```

### Key Parameters

- **QDRANT_URL**: Qdrant Cloud cluster URL
- **QDRANT_API_KEY**: Qdrant Cloud API key
- **QDRANT_COLLECTION**: Collection name (default: `rag_documents`)
- **EMB_MODEL**: OpenAI embedding model (default: `text-embedding-3-small`)
- **RERANKER**: Reranking model (default: `jinaai/jina-reranker-v2-base-multilingual`)
- **RAG_CHAT_MODEL**: Chat model (default: `gpt-5-mini`)

## 🌐 API Usage

### Retriever Service (Port 8000)
```bash
# Search documents
curl "http://localhost:8000/search?q=your+question"
```

### Answer Service (Port 8010)
```bash
# Get detailed answer
curl "http://localhost:8010/ask?q=your+question"

# Health check
curl "http://localhost:8010/health"
```

## 📝 Available Commands

### Setup
- `make deps` - Install dependencies
- `make qdrant-health` - Test Qdrant connection

### Data Pipeline
- `make prep` - Extract text from PDFs
- `make chunk` - Chunk text documents
- `make index` - Build Qdrant collection
- `make bm25` - Build BM25 index
- `make rebuild` - Complete rebuild (prep + chunk + index + bm25)

### Services
- `make retriever` - Start retriever API (:8000)
- `make answer` - Ask a question
- `make answer-api` - Start answer API (:8010)
- `make qa` - Quick question with auto-start retriever

### Testing
- `make test-search` - Test direct search
- `make test-integration` - Full integration test

### Cleanup
- `make clean-index` - Remove local indexes
- `make clean-qdrant` - Remove Qdrant collection

## 📁 Directory Structure

```
rag_prep/
├── raw_pdfs/          # Input PDF documents
├── data/              # Processed data
│   ├── pages.jsonl    # Extracted text/images
│   └── chunks.jsonl   # Text chunks
└── index/             # Local indexes
    ├── bm25.pkl       # BM25 keyword index
    └── qdrant_info.json # Qdrant collection info
```

## 🧩 Core Components

### Python Files
- `answer_rag.py` - Enhanced RAG answering system
- `build_qdrant_index.py` - Qdrant collection builder
- `serve_qdrant_retriever.py` - Qdrant-based retriever service
- `qdrant_store.py` - Qdrant vector store wrapper
- `test_qdrant_integration.py` - Integration testing

### Processing Pipeline
- `prep_pdfs.py` - PDF text extraction
- `chunk_texts.py` - Text chunking
- `build_bm25.py` - BM25 index building

## 🔧 Dependencies

The system uses these key packages:
- **qdrant-client** - Qdrant vector database client
- **sentence-transformers** - Text embeddings and reranking
- **rank-bm25** - Keyword search
- **fastapi, uvicorn** - Web APIs
- **openai** - Language models
- **PyMuPDF, pdf2image** - PDF processing
- **torch, numpy, scikit-learn** - ML libraries

## 🚨 Troubleshooting

### Common Issues

1. **Qdrant connection fails**
   ```bash
   make qdrant-health
   # Check QDRANT_URL and QDRANT_API_KEY
   ```

2. **No search results**
   ```bash
   make status  # Check collection status
   make rebuild  # Rebuild if needed
   ```

3. **Retriever service not running**
   ```bash
   # Start retriever first
   make retriever
   # Then ask questions
   make answer QUESTION="your question"
   ```

## ✨ Advantages Over FAISS

- **Cloud Native**: Managed Qdrant Cloud service
- **Persistent**: Data survives restarts
- **Scalable**: Handles large document collections
- **Production Ready**: Built for enterprise workloads
- **No Local Storage**: Vector data stored in cloud
- **Better Performance**: Optimized vector operations

## 📊 System Status

Check your system status:
```bash
make status
```

This shows:
- Qdrant connection status
- Collection information
- Available data files
- Index files present
