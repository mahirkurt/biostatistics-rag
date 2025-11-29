"""
Biostatistics RAG - FastAPI Service
Standalone microservice for biostatistics guidance

Features:
- API Key authentication for public access
- Rate limiting to prevent abuse
- CORS for cross-origin requests
"""
import os
import sys
import time
import hashlib
import secrets
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta

# Fix ChromaDB encoding issue on Windows
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

from fastapi import FastAPI, HTTPException, Depends, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag_system.query_interface import BiostatisticsRAG

# =============================================================================
# API Key Management
# =============================================================================

# API Keys - stored in environment or generated
API_KEYS_FILE = PROJECT_ROOT / ".api_keys"
VALID_API_KEYS = set()

def load_api_keys():
    """Load API keys from file and environment"""
    global VALID_API_KEYS
    
    # From environment
    env_key = os.getenv("BIOSTAT_API_KEY")
    if env_key:
        VALID_API_KEYS.add(env_key.strip())
    
    # From file
    if API_KEYS_FILE.exists():
        with open(API_KEYS_FILE, "r") as f:
            for line in f:
                key = line.strip()
                if key and not key.startswith("#"):
                    VALID_API_KEYS.add(key)
    
    # Generate default key if none exist
    if not VALID_API_KEYS:
        default_key = generate_api_key()
        save_api_key(default_key, "default")
        VALID_API_KEYS.add(default_key)
        print(f"\n🔑 Generated default API key: {default_key}")
        print(f"   Save this key! It's stored in: {API_KEYS_FILE}\n")

def generate_api_key() -> str:
    """Generate a secure API key"""
    return f"bstat_{secrets.token_urlsafe(32)}"

def save_api_key(key: str, name: str = "unnamed"):
    """Save API key to file"""
    with open(API_KEYS_FILE, "a") as f:
        f.write(f"# {name} - created {datetime.now().isoformat()}\n")
        f.write(f"{key}\n")

# API Key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(
    request: Request,
    api_key: Optional[str] = Depends(api_key_header)
):
    """Verify API key for protected endpoints"""
    # Allow local requests without API key
    client_ip = request.client.host if request.client else "unknown"
    if client_ip in ["127.0.0.1", "localhost", "::1"]:
        return True
    
    # Require API key for external requests
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required. Add 'X-API-Key' header."
        )
    
    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    
    return True

# =============================================================================
# Rate Limiting
# =============================================================================

class RateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(self, requests_per_minute: int = 10, requests_per_day: int = 100):
        self.requests_per_minute = requests_per_minute
        self.requests_per_day = requests_per_day
        self.minute_requests = defaultdict(list)
        self.day_requests = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> tuple[bool, str]:
        """Check if request is allowed"""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        day_ago = now - timedelta(days=1)
        
        # Clean old entries
        self.minute_requests[client_id] = [
            t for t in self.minute_requests[client_id] if t > minute_ago
        ]
        self.day_requests[client_id] = [
            t for t in self.day_requests[client_id] if t > day_ago
        ]
        
        # Check limits
        if len(self.minute_requests[client_id]) >= self.requests_per_minute:
            return False, f"Rate limit exceeded. Max {self.requests_per_minute}/minute."
        
        if len(self.day_requests[client_id]) >= self.requests_per_day:
            return False, f"Daily limit exceeded. Max {self.requests_per_day}/day."
        
        # Record request
        self.minute_requests[client_id].append(now)
        self.day_requests[client_id].append(now)
        
        return True, ""

rate_limiter = RateLimiter(requests_per_minute=10, requests_per_day=200)

async def check_rate_limit(request: Request):
    """Check rate limit for request"""
    client_ip = request.client.host if request.client else "unknown"
    
    # No limit for localhost
    if client_ip in ["127.0.0.1", "localhost", "::1"]:
        return True
    
    allowed, message = rate_limiter.is_allowed(client_ip)
    if not allowed:
        raise HTTPException(status_code=429, detail=message)
    
    return True

# =============================================================================
# FastAPI App
# =============================================================================

# Load API keys on startup
load_api_keys()

# Initialize FastAPI
app = FastAPI(
    title="Biostatistics RAG API",
    description="""
## Biostatistics RAG API

AI-powered biostatistics guidance using RAG (Retrieval-Augmented Generation).

### Features
- 📚 67,000+ chunks from 45 biostatistics textbooks
- 🔍 Semantic search with BGE-M3 embeddings
- 🏥 Medical domain reranking with PubMedBERT
- 📊 Evidence strength scoring
- 🔗 Cross-source verification

### Authentication
External requests require an API key in the `X-API-Key` header.
Local requests (localhost) don't need authentication.

### Rate Limits
- 10 requests per minute
- 200 requests per day
""",
    version="2.0.0"
)

# CORS for cross-origin access
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
        rag_system = BiostatisticsRAG()
    return rag_system


# Request/Response models
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    use_reranking: Optional[bool] = True
    use_semantic_analysis: Optional[bool] = True  # NEW: Enable semantic analysis


class SourceDocument(BaseModel):
    content: str
    source: str
    score: float
    chunk_id: int


class AnalysisInfo(BaseModel):
    evidence_strength: dict
    thematic_clusters: List[str]
    cross_source_agreement: float
    contradictions_found: int


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    query: str
    analysis: Optional[AnalysisInfo] = None  # NEW: Analysis metadata


# Health check
@app.get("/")
async def root():
    return {
        "service": "Biostatistics RAG",
        "status": "running",
        "version": "2.0.0",
        "features": [
            "BGE-M3 embeddings",
            "PubMedBERT reranking",
            "Semantic analysis",
            "Cross-source verification"
        ]
    }

@app.get("/health")
async def health_check():
    try:
        rag = get_rag()
        return {
            "status": "healthy",
            "embedding_model": "BAAI/bge-m3",
            "reranker_model": "pritamdeka/S-PubMedBert-MS-MARCO",
            "collection": "biostatistics_main"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


# Main query endpoint
@app.post("/query", response_model=QueryResponse, dependencies=[Depends(verify_api_key), Depends(check_rate_limit)])
async def query_rag(request: QueryRequest):
    """
    Query the biostatistics RAG system with semantic analysis
    
    Example:
    ```
    {
        "query": "What statistical test for comparing three groups?",
        "top_k": 5,
        "use_reranking": true,
        "use_semantic_analysis": true
    }
    ```
    """
    try:
        rag = get_rag()
        result = rag.query(
            query=request.query,
            top_k=request.top_k,
            use_reranking=request.use_reranking,
            use_semantic_analysis=request.use_semantic_analysis
        )
        
        # Build response with optional analysis
        response_data = {
            "answer": result["answer"],
            "sources": [
                SourceDocument(
                    content=doc["content"],
                    source=doc["source"],
                    score=doc["score"],
                    chunk_id=doc.get("chunk_id", 0)
                )
                for doc in result["sources"]
            ],
            "query": request.query
        }
        
        # Add analysis if available
        if result.get("analysis"):
            response_data["analysis"] = AnalysisInfo(
                evidence_strength=result["analysis"]["evidence_strength"],
                thematic_clusters=result["analysis"]["thematic_clusters"],
                cross_source_agreement=result["analysis"]["cross_source_agreement"],
                contradictions_found=result["analysis"]["contradictions_found"]
            )
        
        return QueryResponse(**response_data)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


# Analyze sources endpoint (NEW)
@app.post("/analyze", dependencies=[Depends(verify_api_key), Depends(check_rate_limit)])
async def analyze_sources(request: QueryRequest):
    """
    Analyze sources without generating an answer
    
    Returns detailed evidence analysis including:
    - Evidence strength score
    - Thematic clustering
    - Cross-source consistency
    - Potential contradictions
    """
    try:
        rag = get_rag()
        analysis = rag.analyze_sources(
            query=request.query,
            top_k=request.top_k
        )
        return analysis
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# Search without LLM
@app.post("/search", dependencies=[Depends(verify_api_key), Depends(check_rate_limit)])
async def search_only(request: QueryRequest):
    """Search documents without LLM answer generation"""
    try:
        rag = get_rag()
        docs = rag.retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            use_reranking=request.use_reranking
        )
        
        return {
            "query": request.query,
            "results": [
                {
                    "content": doc["content"],
                    "source": doc["metadata"]["source"],
                    "score": doc["score"]
                }
                for doc in docs
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# Collection stats
@app.get("/stats")
async def get_stats():
    """Get collection statistics"""
    try:
        rag = get_rag()
        collection = rag.retriever.collection
        
        return {
            "collection_name": collection.name,
            "total_chunks": collection.count(),
            "embedding_model": "BAAI/bge-m3",
            "reranker_model": "pritamdeka/S-PubMedBert-MS-MARCO"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8002,  # Biostatistics on 8002
        reload=False  # Disable reload to prevent VS Code terminal issues
    )
