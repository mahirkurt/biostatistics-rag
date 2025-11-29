"""
OpenAI Embeddings Adapter
==========================
Use OpenAI's text-embedding-3-large as alternative to BGE-M3
Requires: OPENAI_API_KEY environment variable

Pros:
- No local model loading issues
- Fast, reliable
- Good quality (3072-dim, can truncate to 1024)

Cons:
- Costs money (~$0.00013 per 1K tokens)
- Requires internet connection
- Different embedding space than BGE-M3
"""

import numpy as np
from typing import List, Union
import os


class OpenAIEmbedder:
    """OpenAI embeddings adapter for RAG system"""
    
    def __init__(self, api_key: str = None, model: str = "text-embedding-3-large"):
        """
        Initialize OpenAI embedder
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: OpenAI embedding model name
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install: pip install openai")
        
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        
        # OpenAI models support dimension reduction
        if model == "text-embedding-3-large":
            self.dimension = 1024  # Reduced from 3072 to match BGE-M3
        else:
            self.dimension = 1536  # text-embedding-3-small
        
        print(f"[OK] OpenAI embedder initialized ({model}, {self.dimension}D)")
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 100,  # OpenAI allows large batches
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode texts using OpenAI API
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size (OpenAI handles up to 2048 texts)
            normalize: Whether to normalize embeddings
            
        Returns:
            numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        all_embeddings = []
        total_tokens = 0
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                    dimensions=self.dimension  # Request specific dimension
                )
                
                # Extract embeddings
                batch_embeddings = [
                    item.embedding for item in response.data
                ]
                all_embeddings.extend(batch_embeddings)
                
                total_tokens += response.usage.total_tokens
                
            except Exception as e:
                print(f"[ERROR] Batch {i//batch_size + 1} failed: {e}")
                # Return zero vectors as fallback
                all_embeddings.extend([[0.0] * self.dimension] * len(batch))
        
        embeddings = np.array(all_embeddings, dtype=np.float32)
        
        # OpenAI embeddings are already normalized, but ensure consistency
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-9)
        
        print(f"[INFO] Processed {len(texts)} texts, {total_tokens} tokens")
        print(f"       Estimated cost: ${total_tokens * 0.00013 / 1000:.4f}")
        
        return embeddings


# Cost estimator
def estimate_indexing_cost(num_documents: int, avg_chars_per_doc: int = 50000):
    """
    Estimate OpenAI embedding cost for indexing
    
    Args:
        num_documents: Number of documents
        avg_chars_per_doc: Average characters per document
        
    Returns:
        Estimated cost in USD
    """
    # Rough estimate: 1 token ≈ 4 chars
    total_tokens = (num_documents * avg_chars_per_doc) / 4
    cost_per_1k_tokens = 0.00013  # text-embedding-3-large
    estimated_cost = (total_tokens / 1000) * cost_per_1k_tokens
    
    print(f"Estimated indexing cost:")
    print(f"  Documents: {num_documents}")
    print(f"  Total tokens (approx): {total_tokens:,.0f}")
    print(f"  Cost: ${estimated_cost:.2f}")
    
    return estimated_cost


if __name__ == "__main__":
    # Estimate cost for 45 biostatistics PDFs
    estimate_indexing_cost(45, avg_chars_per_doc=568000 // 2)
