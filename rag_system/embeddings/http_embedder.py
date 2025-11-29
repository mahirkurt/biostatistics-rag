"""
HTTP-based Embedding Client
============================
Alternative to local BGE-M3 when torch has issues
Can connect to a remote embedding service
"""

import numpy as np
import requests
from typing import List, Union


class HTTPEmbedder:
    """Client for remote embedding service"""
    
    def __init__(self, service_url: str = "http://localhost:8003"):
        """
        Args:
            service_url: URL of embedding service
        """
        self.service_url = service_url.rstrip('/')
        self.dimension = 1024  # BGE-M3 dimension
        
        # Test connection
        try:
            response = requests.get(f"{self.service_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"[OK] Connected to embedding service at {service_url}")
            else:
                print(f"[WARNING] Service responded with status {response.status_code}")
        except Exception as e:
            print(f"[WARNING] Could not connect to service: {e}")
            print("         Service must be running before indexing")
    
    def encode(
        self, 
        texts: Union[str, List[str]], 
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode texts to embeddings via HTTP
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for processing
            normalize: Whether to normalize embeddings
            
        Returns:
            numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                response = requests.post(
                    f"{self.service_url}/embed",
                    json={
                        "texts": batch,
                        "normalize": normalize
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    embeddings = response.json()["embeddings"]
                    all_embeddings.extend(embeddings)
                else:
                    raise Exception(f"Service error: {response.status_code}")
                    
            except Exception as e:
                print(f"[ERROR] Batch {i//batch_size + 1} failed: {e}")
                # Return zero vectors as fallback
                all_embeddings.extend([[0.0] * self.dimension] * len(batch))
        
        return np.array(all_embeddings, dtype=np.float32)


# Simple embedding service code (to run on another machine)
EMBEDDING_SERVICE_CODE = '''
"""
Simple Embedding Service
Run this on a machine where PyTorch works properly
Usage: python embedding_service.py
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np
import uvicorn

app = FastAPI(title="BGE-M3 Embedding Service")

# Load model at startup
print("Loading BGE-M3 model...")
model = SentenceTransformer("BAAI/bge-m3")
print(f"Model loaded (dimension: {model.get_sentence_embedding_dimension()})")


class EmbedRequest(BaseModel):
    texts: List[str]
    normalize: bool = True


@app.get("/health")
def health():
    return {"status": "ok", "model": "BAAI/bge-m3", "dimension": 1024}


@app.post("/embed")
def embed(request: EmbedRequest):
    """Generate embeddings for texts"""
    embeddings = model.encode(
        request.texts,
        normalize_embeddings=request.normalize,
        show_progress_bar=False
    )
    
    # Convert to list for JSON serialization
    return {
        "embeddings": embeddings.tolist(),
        "count": len(request.texts)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
'''


def save_embedding_service_code(output_path: str = "embedding_service.py"):
    """Save the embedding service code to a file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(EMBEDDING_SERVICE_CODE)
    print(f"[OK] Embedding service code saved to: {output_path}")
    print("     Run on another machine: python embedding_service.py")
