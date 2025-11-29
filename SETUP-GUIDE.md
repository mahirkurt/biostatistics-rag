# Multi-RAG Microservices Architecture

## Overview
Her RAG sistemi **bağımsız FastAPI servisi** olarak çalışır. Her proje kendi virtual environment'ında, kendi portunda çalışır.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CF Project (Main)                        │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           MultiRAGClient                             │  │
│  │  - Tüm RAG servislerine HTTP istekleri gönderir     │  │
│  │  - Sonuçları birleştirir                            │  │
│  │  - Claude ile sentezler                              │  │
│  └──────────────────────────────────────────────────────┘  │
│                          │                                  │
│            ┌─────────────┴─────────────┐                   │
│            │                           │                   │
└────────────┼───────────────────────────┼───────────────────┘
             │                           │
             ▼                           ▼
    ┌────────────────┐         ┌────────────────┐
    │ Medical Writing│         │ Biostatistics  │
    │  RAG Service   │         │  RAG Service   │
    │                │         │                │
    │ Port: 8001     │         │ Port: 8002     │
    │ Model: BGE-M3  │         │ Model: BGE-M3  │
    │ Rerank: PubMed │         │ Rerank: Nomic  │
    │ Docs: 25 PDFs  │         │ Docs: 45 PDFs  │
    └────────────────┘         └────────────────┘
```

## Setup Instructions

### 1. Medical Writing RAG Servisi

```bash
# 1. Klasöre git
cd "D:\Repositories\Medical Writing"

# 2. Virtual environment oluştur
python -m venv .venv
.venv\Scripts\activate

# 3. Dependency'leri yükle
pip install fastapi uvicorn chromadb sentence-transformers torch anthropic python-dotenv PyPDF2

# 4. API key ekle (.env dosyası)
echo "ANTHROPIC_API_KEY=your_key_here" > .env

# 5. Index oluştur (ilk seferinde)
python rag_system/indexers/run_indexer.py

# 6. Servisi başlat
cd rag_system
python api_server.py
# Servis çalışıyor: http://localhost:8001
```

### 2. Biostatistics RAG Servisi

```bash
# 1. Klasöre git
cd "D:\Repositories\Biostatistics"

# 2. Virtual environment oluştur
python -m venv .venv
.venv\Scripts\activate

# 3. Dependency'leri yükle
pip install fastapi uvicorn chromadb sentence-transformers torch anthropic python-dotenv PyPDF2

# 4. API key ekle (.env dosyası)
echo "ANTHROPIC_API_KEY=your_key_here" > .env

# 5. Index oluştur (ilk seferinde)
python rag_system/indexers/run_indexer.py

# 6. Servisi başlat
cd rag_system
python api_server.py
# Servis çalışıyor: http://localhost:8002
```

### 3. CF Projesinden Kullan

```python
from scripts.multi_rag_client import MultiRAGClient

# Client oluştur
client = MultiRAGClient(
    medical_writing_url="http://localhost:8001",
    biostatistics_url="http://localhost:8002"
)

# Servisleri kontrol et
status = client.check_services()
print(status)  # {'medical_writing': True, 'biostatistics': True}

# Medical Writing'den sor
result = client.query_medical_writing(
    "How to write discussion section?"
)
print(result['answer'])

# Biostatistics'ten sor
result = client.query_biostatistics(
    "Which test for comparing three groups?"
)
print(result['answer'])

# İkisini birleştir
result = client.query_both_rags(
    "How to report statistical results in manuscript?"
)
print(result['answer'])
```

## API Endpoints

### Medical Writing (Port 8001)

- `GET /` - Service info
- `GET /health` - Health check
- `POST /query` - Query with LLM answer
  ```json
  {
    "query": "How to write methods section?",
    "top_k": 5,
    "use_reranking": true
  }
  ```
- `POST /search` - Search without LLM (just retrieval)
- `GET /stats` - Collection statistics

### Biostatistics (Port 8002)

- Same endpoints as Medical Writing

## Advantages

✅ **Bağımsızlık**: Her RAG kendi environment'ında, kendi dependency'leriyle
✅ **Ölçeklenebilirlik**: Her servisi ayrı makinede çalıştırabilirsin
✅ **Hata izolasyonu**: Bir RAG çökerse diğerleri çalışmaya devam eder
✅ **Kolay güncellemeler**: Bir RAG'ı update ederken diğerleri etkilenmez
✅ **Multiple projects**: CF, diğer projeler hepsi aynı RAG'ları kullanabilir
✅ **Load balancing**: İleride multiple instance çalıştırabilirsin

## Production Deployment

### Docker ile (Önerilen)

Her RAG için Dockerfile:
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8001

CMD ["uvicorn", "rag_system.api_server:app", "--host", "0.0.0.0", "--port", "8001"]
```

Docker Compose:
```yaml
version: '3.8'
services:
  medical-writing-rag:
    build: ./Medical Writing
    ports:
      - "8001:8001"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    volumes:
      - ./Medical Writing/rag_system/chroma_db:/app/rag_system/chroma_db
  
  biostatistics-rag:
    build: ./Biostatistics
    ports:
      - "8002:8002"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    volumes:
      - ./Biostatistics/rag_system/chroma_db:/app/rag_system/chroma_db
```

### Cloud Deployment

- **Azure**: Container Instances veya App Service
- **AWS**: ECS veya Lambda (with function URLs)
- **Google Cloud**: Cloud Run
- **Railway/Render**: En kolay deployment

## Usage in CF Project

```python
# CF projesindeki thesis query'e ekle
from scripts.multi_rag_client import MultiRAGClient

# Thesis query sırasında ek kaynak
rag_client = MultiRAGClient()

# Eğer thesis'te istatistik sorusu varsa
if "statistical" in user_query.lower():
    biostat_info = rag_client.search_biostatistics(user_query, top_k=3)
    # Thesis context'e ekle

# Eğer writing sorusu varsa
if "writing" in user_query.lower() or "manuscript" in user_query.lower():
    writing_info = rag_client.search_medical_writing(user_query, top_k=3)
    # Thesis context'e ekle
```

## Next Steps

1. ✅ API servisleri oluşturuldu
2. ⏳ Medical Writing'de ChromaDB index'i oluştur
3. ⏳ Biostatistics'te ChromaDB index'i oluştur
4. ⏳ Her iki servisi başlat
5. ⏳ CF projesinden test et
6. ⏳ Docker image'ları oluştur (opsiyonel)
7. ⏳ Cloud'a deploy et (opsiyonel)
