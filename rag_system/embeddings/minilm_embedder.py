"""
Lightweight MiniLM Embedder
============================
Uses all-MiniLM-L6-v2 (80MB) instead of BGE-M3 (2GB)
Much faster to load, fewer torch issues

Pros:
- Small model (80MB vs 2GB)
- Loads quickly
- Less likely to have torch issues
- Good quality for general text

Cons:
- Lower dimension (384 vs 1024)
- Slightly lower quality than BGE-M3
- Not optimized for biostatistics specifically
"""

import numpy as np
from typing import List, Union


class MiniLMEmbedder:
    """Lightweight embedder using MiniLM-L6-v2"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize MiniLM embedder
        
        Args:
            model_name: HuggingFace model name
        """
        print(f"[*] Loading lightweight model: {model_name}")
        print("    This is much smaller than BGE-M3 (80MB vs 2GB)")
        
        try:
            from sentence_transformers import SentenceTransformer
            
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            
            print(f"[OK] Model loaded (dimension: {self.dimension})")
            
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            raise
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode texts to embeddings
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for processing
            normalize: Whether to normalize embeddings
            show_progress: Show progress bar
            
        Returns:
            numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        return embeddings.astype(np.float32)


# Adapter to match BGE-M3 dimension (pad with zeros)
class MiniLMAdapter:
    """
    Adapter that pads MiniLM embeddings to 1024-dim
    This allows using existing ChromaDB collection
    """
    
    def __init__(self):
        self.embedder = MiniLMEmbedder()
        self.dimension = 1024  # Match BGE-M3
        print("[INFO] Will pad 384-dim to 1024-dim to match BGE-M3")
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        """Encode and pad to 1024 dimensions"""
        
        # Get 384-dim embeddings
        embeddings = self.embedder.encode(texts, batch_size, normalize=False)
        
        # Pad to 1024-dim with zeros
        padded = np.zeros((embeddings.shape[0], 1024), dtype=np.float32)
        padded[:, :384] = embeddings
        
        # Normalize after padding if requested
        if normalize:
            norms = np.linalg.norm(padded, axis=1, keepdims=True)
            padded = padded / np.maximum(norms, 1e-9)
        
        return padded


# Test function
def test_minilm():
    """Test MiniLM embedder"""
    try:
        print("Testing MiniLM embedder...")
        embedder = MiniLMEmbedder()
        
        texts = [
            "What is the difference between t-test and ANOVA?",
            "How to calculate p-value?"
        ]
        
        embeddings = embedder.encode(texts, show_progress=False)
        print(f"[TEST] Generated {embeddings.shape[0]} embeddings")
        print(f"       Dimension: {embeddings.shape[1]}")
        
        print("\nTesting adapter (with padding)...")
        adapter = MiniLMAdapter()
        padded = adapter.encode(texts)
        print(f"[TEST] Padded dimension: {padded.shape[1]}")
        
        return True
    except Exception as e:
        print(f"[TEST FAILED] {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_minilm()
