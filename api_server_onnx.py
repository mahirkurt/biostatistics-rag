"""
Biostatistics RAG - FastAPI Service (Pure ONNX Edition)
Standalone microservice for biostatistics guidance
No PyTorch dependency!
"""
import os
import sys
from pathlib import Path

# CRITICAL: Fix ChromaDB encoding issue on Windows
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

# Platform mocking for ChromaDB
import platform
platform.system = lambda: "Windows"
platform.win32_ver = lambda: ("10", "10.0.19041", "", "Multiprocessor Free")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Use Pure ONNX query interface (no PyTorch!)
from rag_system.query_interface_onnx import BiostatisticsRAG

# Initialize FastAPI
app = FastAPI(
    title="Biostatistics RAG API",
    description="Biostatistics guidance and statistical methods search (Pure ONNX Edition)",
    version="2.0.0"
)

# CORS for cross-project access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG (lazy loading)
rag_system = None


def get_rag():
    global rag_system
    if rag_system is None:
        rag_system = BiostatisticsRAG(verbose=False)
    return rag_system


# Request/Response models
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    use_reranking: Optional[bool] = True  # Kept for API compatibility


class SourceDocument(BaseModel):
    content: str
    source: str
    score: float
    chunk_id: Optional[int] = 0


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    query: str


class SearchResult(BaseModel):
    content: str
    source: str
    score: float


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]


# Root endpoint
@app.get("/")
async def root():
    return {
        "service": "Biostatistics RAG",
        "status": "running",
        "version": "2.0.0",
        "embedding": "Pure ONNX MiniLM",
        "note": "No PyTorch dependency!"
    }


# Health check
@app.get("/health")
async def health_check():
    try:
        rag = get_rag()
        stats = rag.get_stats()
        return {
            "status": "healthy",
            "embedding_model": "MiniLM-L6-v2 (Pure ONNX)",
            "collection": stats["name"],
            "documents": stats["count"]
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


# Main query endpoint
@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Query the biostatistics RAG system with LLM answer generation
    
    Example:
    ```json
    {
        "query": "Üç grup karşılaştırması için hangi istatistiksel test kullanmalıyım?",
        "top_k": 5
    }
    ```
    
    Returns:
    - answer: Claude-generated Turkish answer
    - sources: Retrieved source documents
    - query: Original query
    """
    try:
        rag = get_rag()
        result = rag.query(
            query=request.query,
            top_k=request.top_k,
            use_reranking=request.use_reranking
        )
        
        return QueryResponse(
            answer=result["answer"],
            sources=[
                SourceDocument(
                    content=doc["content"],
                    source=doc["metadata"].get("source", "Unknown"),
                    score=doc.get("score", 0.0),
                    chunk_id=doc["metadata"].get("chunk_index", 0)
                )
                for doc in result["sources"]
            ],
            query=request.query
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


# Search without LLM (just retrieval)
@app.post("/search", response_model=SearchResponse)
async def search_only(request: QueryRequest):
    """
    Search documents without LLM answer generation
    Returns only relevant chunks
    
    Example:
    ```json
    {
        "query": "power analysis",
        "top_k": 10
    }
    ```
    """
    try:
        rag = get_rag()
        docs = rag.search_only(
            query=request.query,
            top_k=request.top_k
        )
        
        return SearchResponse(
            query=request.query,
            results=[
                SearchResult(
                    content=doc["content"],
                    source=doc["metadata"].get("source", "Unknown"),
                    score=doc.get("score", 0.0)
                )
                for doc in docs
            ]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# Collection stats
@app.get("/stats")
async def get_stats():
    """Get collection statistics"""
    try:
        rag = get_rag()
        stats = rag.get_stats()
        
        return {
            "collection_name": stats["name"],
            "total_documents": stats["count"],
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2 (Pure ONNX)",
            "embedding_dimension": 1024,  # Padded from 384
            "llm_model": "claude-sonnet-4-20250514"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_server_onnx:app",
        host="0.0.0.0",
        port=8002,  # Biostatistics on 8002
        reload=True
    )
