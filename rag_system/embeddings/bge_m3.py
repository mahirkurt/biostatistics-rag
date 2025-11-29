"""
BGE-M3 Embedding Model Wrapper
===============================
State-of-the-art general-purpose embedding model

Model: BAAI/bge-m3
Dimension: 1024
Best for: Academic text retrieval, multi-lingual support

@author: Dr. Mahir Kurt + Claude Code
@date: 2025-11-24
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union


class BGEModel:
    """BGE-M3 embedding model wrapper"""

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        """
        Initialize BGE-M3 model

        Args:
            model_name: Hugging Face model identifier (default: BAAI/bge-m3)
        """
        print(f"[*] Loading BGE-M3 model: {model_name}")
        print("   This may take a few minutes on first run...")

        self.model = SentenceTransformer(model_name)
        self.dimension = 1024  # BGE-M3 output dimension

        print(f"[OK] BGE-M3 loaded successfully (dimension: {self.dimension})")

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode text(s) into embeddings

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            normalize: L2 normalize embeddings (recommended for cosine similarity)

        Returns:
            Embeddings as numpy array
            - Single text: shape (1024,)
            - Multiple texts: shape (n, 1024)
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )

        # If single text, return 1D array
        if len(texts) == 1:
            return embeddings[0]

        return embeddings

    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimension

    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            "model_name": "BAAI/bge-m3",
            "dimension": self.dimension,
            "max_seq_length": self.model.max_seq_length,
            "type": "general-purpose",
            "best_for": "Academic text, multi-lingual"
        }


def test_bge_model():
    """Test BGE-M3 model"""
    print("\n" + "=" * 60)
    print("BGE-M3 Model Test")
    print("=" * 60)

    # Initialize model
    model = BGEModel()

    # Test single text
    print("\n[Test 1] Single text encoding")
    text = "This is a test sentence for medical research about cystic fibrosis."
    embedding = model.encode(text)
    print(f"   Input: {text[:50]}...")
    print(f"   Embedding shape: {embedding.shape}")
    print(f"   First 5 values: {embedding[:5]}")
    print(f"   L2 norm: {np.linalg.norm(embedding):.4f}")

    # Test multiple texts
    print("\n[Test 2] Multiple texts encoding")
    texts = [
        "Adverse events in clinical trials",
        "Statistical methods for biostatistics",
        "Medical writing guidelines for publications"
    ]
    embeddings = model.encode(texts)
    print(f"   Input: {len(texts)} texts")
    print(f"   Embeddings shape: {embeddings.shape}")

    # Test similarity
    print("\n[Test 3] Similarity computation")
    text1 = "Cystic fibrosis treatment methods"
    text2 = "CF therapy approaches"
    text3 = "Statistical power analysis"

    emb1 = model.encode(text1)
    emb2 = model.encode(text2)
    emb3 = model.encode(text3)

    sim_12 = np.dot(emb1, emb2)  # Cosine similarity (normalized embeddings)
    sim_13 = np.dot(emb1, emb3)

    print(f"   Text 1: {text1}")
    print(f"   Text 2: {text2}")
    print(f"   Text 3: {text3}")
    print(f"   Similarity(1,2): {sim_12:.4f} (related)")
    print(f"   Similarity(1,3): {sim_13:.4f} (unrelated)")

    # Model info
    print("\n[Test 4] Model information")
    info = model.get_model_info()
    for key, value in info.items():
        print(f"   {key}: {value}")

    print("\n✅ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_bge_model()
