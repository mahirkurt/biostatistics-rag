# Biostatistics RAG Microservice - AI Agent Instructions

## Architecture Overview

This is a **standalone FastAPI microservice** providing biostatistics guidance through RAG (Retrieval-Augmented Generation). The system uses **BGE-M3 + PubMedBERT** for hybrid retrieval with domain-specific reranking.

**Key Architecture Points:**

- **Microservice Pattern**: Runs on port 8002, independent from other services
- **Primary Embedder**: BGE-M3 (BAAI/bge-m3, 1024-dim) - State-of-the-art multilingual
- **Reranker**: PubMedBERT (pritamdeka/S-PubMedBert-MS-MARCO) - Medical domain specialized
- **Vector Store**: ChromaDB 1.3.5 with persistent storage at `rag_system/chroma_db/`
- **Documents**: 45 biostatistics textbooks in `textbooks/` directory
- **LLM**: Claude Sonnet 4.5 for Turkish biostatistics answers

## Critical Windows/ChromaDB Fix

**ALWAYS set these environment variables AND platform mocks BEFORE any ChromaDB import:**

```python
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

# Platform mocking for ChromaDB 1.3.5 on Windows
import platform
platform.system = lambda: "Windows"
platform.win32_ver = lambda: ("10", "10.0.19041", "", "Multiprocessor Free")
```

This fixes ChromaDB 1.3.x Rust backend encoding issues on Windows.

## Key Components

### 1. Query Interface (`rag_system/query_interface.py`)

The `BiostatisticsRAG` class provides:

- Full RAG with LLM: `query(query, top_k)` → Returns answer + sources
- Search only: `search_only(query, top_k)` → Returns documents without LLM
- Multi-query retrieval for comprehensive coverage
- Hybrid retrieval with PubMedBERT reranking
- Claude Sonnet 4.5 for Turkish biostatistics answers

### 2. Embedding Models (`rag_system/embedding_models.py`)

**Primary Models (PyTorch-based):**

| Model | Class | Dimension | Usage |
|-------|-------|-----------|-------|
| BGE-M3 | `BGEModel` | 1024 | Primary embedder for indexing & retrieval |
| PubMedBERT | `PubMedBERTModel` | 768 | Reranker for medical domain relevance |

### 3. Hybrid Retriever (`rag_system/retrievers.py`)

Two-stage retrieval:

1. **Stage 1**: BGE-M3 retrieves top-50 documents from ChromaDB
2. **Stage 2**: PubMedBERT reranks to top-20 based on medical domain relevance

### 4. Configuration (`rag_system/config/biostat_rag_config.py`)

```python
PRIMARY_MODEL_NAME = "BAAI/bge-m3"  # 1024-dim
SECONDARY_MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"  # Reranker
COLLECTION_NAME = "biostatistics_main"
TOP_K_RETRIEVE = 50  # Primary retrieval
TOP_K_FINAL = 20     # After reranking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
```

## API Endpoints

```python
# Full RAG query with LLM (returns Turkish answer)
POST /query
{
  "query": "Üç grup karşılaştırması için hangi test?",
  "top_k": 5
}
# Response: {"answer": "...", "sources": [...], "query": "..."}

# Retrieval only (no LLM)
POST /search
{
  "query": "power analysis",
  "top_k": 5
}

# Health check
GET /health

# Collection statistics
GET /stats
```

## Development Workflow

1. **Setup**: Install from `requirements.txt`
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure**: Set `ANTHROPIC_API_KEY` in `.env`

3. **Index PDFs**: Run BGE-M3 indexer
   ```bash
   cd rag_system/indexers
   python run_indexer_bge_m3.py
   ```

4. **Run API**: 
   ```bash
   python api_server.py
   ```

5. **Test**: 
   ```bash
   curl http://localhost:8002/health
   ```

## Indexing

### BGE-M3 Indexer

```bash
cd rag_system/indexers
python run_indexer_bge_m3.py
```

### Available Indexers

| Indexer | Description | Model |
|---------|-------------|-------|
| `run_indexer_bge_m3.py` | **RECOMMENDED** - PyTorch BGE-M3 | BAAI/bge-m3 |
| `run_indexer_pure_onnx.py` | Pure ONNX (no PyTorch) | MiniLM-L6-v2 |
| `run_indexer_lightweight.py` | Original lightweight | sentence_transformers |

## Project Structure

```
Biostatistics/
├── textbooks/                    # 45 PDF textbooks
├── rag_system/
│   ├── embedding_models.py       # ← BGEModel, PubMedBERTModel
│   ├── retrievers.py             # ← HybridRetriever
│   ├── query_interface.py        # ← BiostatisticsRAG
│   ├── embeddings/               # Alternative embedders
│   │   ├── bge_m3.py
│   │   ├── pure_onnx_minilm.py
│   │   └── ...
│   ├── indexers/
│   │   ├── run_indexer_bge_m3.py    # ← RECOMMENDED
│   │   ├── run_indexer_pure_onnx.py
│   │   └── ...
│   ├── chroma_db/               # Vector DB (generated)
│   └── config/
│       └── biostat_rag_config.py
├── api_server.py                # FastAPI service (port 8002)
├── requirements.txt
└── .env                         # ANTHROPIC_API_KEY
```

## Multi-RAG Client Pattern

This service is consumed by other projects via `MultiRAGClient`:

```python
from multi_rag_client import MultiRAGClient

client = MultiRAGClient(
    biostatistics_url="http://localhost:8002"
)
result = client.query_biostatistics("power analysis?")
```

## Testing

```python
# Quick test
from rag_system.query_interface import BiostatisticsRAG

rag = BiostatisticsRAG(verbose=True)
results = rag.search_only("power analysis", top_k=3)
print(f"Found {len(results)} results")

# Full RAG with answer
result = rag.query("Üç grup için hangi test?")
print(result["answer"])
```

## Dependencies

Key dependencies in `requirements.txt`:

```
sentence-transformers>=2.2.2
torch>=2.1.0
transformers>=4.36.0
chromadb>=0.5.23
anthropic>=0.18.0
PyMuPDF>=1.24.0
```

## Deployment Notes

- **Port**: 8002 (don't conflict with Medical Writing on 8001)
- **Single-machine only**: Not designed for distributed deployment
- **Turkish-optimized**: LLM prompts generate Turkish answers
- **PyTorch required**: BGE-M3 and PubMedBERT need PyTorch
