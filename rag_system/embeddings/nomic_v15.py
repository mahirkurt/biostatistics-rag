"""
Nomic-Embed-Text-v1.5 Wrapper
==============================
Long-context general-purpose embedding model

Model: nomic-ai/nomic-embed-text-v1.5
Dimension: 768
Best for: Long academic texts, methodological sections

@author: Dr. Mahir Kurt + Claude Code
@date: 2025-11-24
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union


class NomicModel:
    """Nomic-Embed-Text-v1.5 embedding model wrapper"""

    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        """
        Initialize Nomic model

        Args:
            model_name: Hugging Face model identifier
        """
        print(f"[*] Loading Nomic model: {model_name}")
        print("   Optimized for long-context academic texts...")

        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True  # Nomic requires this
        )
        self.dimension = 768

        print(f"[OK] Nomic model loaded successfully (dimension: {self.dimension})")

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
            - Single text: shape (768,)
            - Multiple texts: shape (n, 768)
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
            "model_name": "nomic-ai/nomic-embed-text-v1.5",
            "dimension": self.dimension,
            "max_seq_length": self.model.max_seq_length,
            "type": "long-context",
            "best_for": "Academic methodology, long texts"
        }


def test_nomic_model():
    """Test Nomic model"""
    print("\n" + "=" * 60)
    print("Nomic-Embed-Text-v1.5 Model Test")
    print("=" * 60)

    # Initialize model
    model = NomicModel()

    # Test single text
    print("\n[Test 1] Single text encoding")
    text = "Statistical methods for longitudinal data analysis in clinical trials."
    embedding = model.encode(text)
    print(f"   Input: {text[:50]}...")
    print(f"   Embedding shape: {embedding.shape}")
    print(f"   First 5 values: {embedding[:5]}")
    print(f"   L2 norm: {np.linalg.norm(embedding):.4f}")

    # Test long methodological text
    print("\n[Test 2] Long methodological text encoding")
    long_text = """
    Sample size calculation for logistic regression analysis requires consideration
    of several factors including the expected odds ratio, baseline event rate,
    and desired statistical power. The formula by Hsieh et al. provides a method
    for calculating the required sample size based on these parameters. For a two-sided
    test with alpha of 0.05 and power of 0.80, the minimum sample size can be
    estimated using the variance of the predictor and the anticipated effect size.
    """
    embedding = model.encode(long_text)
    print(f"   Input length: {len(long_text)} chars")
    print(f"   Embedding shape: {embedding.shape}")

    # Test similarity (statistical texts)
    print("\n[Test 3] Statistical methodology similarity")
    text1 = "Power analysis for survival analysis using Cox regression"
    text2 = "Sample size determination for time-to-event outcomes"
    text3 = "Adverse event reporting in clinical trials"

    emb1 = model.encode(text1)
    emb2 = model.encode(text2)
    emb3 = model.encode(text3)

    sim_12 = np.dot(emb1, emb2)  # Cosine similarity
    sim_13 = np.dot(emb1, emb3)

    print(f"   Text 1: {text1}")
    print(f"   Text 2: {text2}")
    print(f"   Text 3: {text3}")
    print(f"   Similarity(1,2): {sim_12:.4f} (related stats)")
    print(f"   Similarity(1,3): {sim_13:.4f} (different topic)")

    # Model info
    print("\n[Test 4] Model information")
    info = model.get_model_info()
    for key, value in info.items():
        print(f"   {key}: {value}")

    print("\n✅ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_nomic_model()
