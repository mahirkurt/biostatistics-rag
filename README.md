# Biostatistics RAG Microservice

**Standalone** domain-specific RAG system for biostatistics guidance with **BGE-M3 + PubMedBERT** hybrid retrieval.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API Key

Edit `.env`:

```env
ANTHROPIC_API_KEY=your_claude_api_key
```

### 3. Index PDFs (First Time)

```bash
cd rag_system/indexers
python run_indexer_bge_m3.py
```

⏱️ Takes ~30-60 minutes for 45 PDFs

### 4. Start API Server

```bash
python api_server.py
```

🌐 Server: http://localhost:8002

## 📖 Architecture

- **Embeddings**: BGE-M3 (BAAI/bge-m3, 1024-dim)
- **Reranker**: PubMedBERT (medical domain specialized)
- **Vector DB**: ChromaDB 1.3.5 with Windows fixes
- **LLM**: Claude Sonnet 4.5 (Turkish responses)
- **Documents**: 45 biostatistics textbooks

### Hybrid Retrieval Flow

1. **Stage 1 - BGE-M3**: Retrieves top-50 semantically similar documents
2. **Stage 2 - PubMedBERT**: Reranks documents based on medical/scientific domain knowledge

This ensures both semantic relevance and domain-specific accuracy.

## 🔌 API Usage

### Python Client

```python
import requests

response = requests.post(
    "http://localhost:8002/query",
    json={
        "query": "Üç grup karşılaştırması için hangi test?",
        "top_k": 5
    }
)
print(response.json()["answer"])
```

### From Other Projects

```python
from multi_rag_client import MultiRAGClient

client = MultiRAGClient(biostatistics_url="http://localhost:8002")
result = client.query_biostatistics("What is power analysis?")
```

## 📡 Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service info |
| `/health` | GET | Health check |
| `/query` | POST | Full RAG query with LLM |
| `/search` | POST | Retrieval only (no LLM) |
| `/stats` | GET | Collection statistics |

## 📁 Project Structure

```
Biostatistics/
├── textbooks/                    # 45 PDF textbooks
├── rag_system/
│   ├── embedding_models.py       # BGE-M3 + PubMedBERT
│   ├── retrievers.py             # Hybrid retriever
│   ├── query_interface.py        # RAG query interface
│   ├── config/
│   │   └── biostat_rag_config.py
│   ├── indexers/
│   │   └── run_indexer_bge_m3.py # Main indexer
│   ├── chroma_db/                # Vector DB (generated)
│   └── embeddings/               # Alternative embedders
├── api_server.py                 # FastAPI server (port 8002)
├── requirements.txt
├── .env
└── README.md
```

## 🐛 Troubleshooting

### ChromaDB Windows Fix

If you see encoding errors, ensure these are set before any ChromaDB import:

```python
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

import platform
platform.system = lambda: "Windows"
platform.win32_ver = lambda: ("10", "10.0.19041", "", "Multiprocessor Free")
```

### Memory Issues

Reduce batch size in indexer (default: 50) if OOM errors occur.

## 🎯 Use Cases

- Statistical test selection
- Sample size calculation guidance
- Power analysis recommendations
- Data analysis plan development
- Statistical reporting assistance

---

Built with ❤️ by Dr. Mahir Kurt + Claude
