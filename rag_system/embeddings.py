"""
Standalone Embeddings Module
============================
BGE-M3 and PubMedBERT models for Biostatistics RAG

BGE-M3: Primary embedder (1024-dim) - General-purpose multilingual
PubMedBERT: Secondary reranker - Medical/scientific domain

@author: Dr. Mahir Kurt + Claude Code
@date: 2025-11-25
"""

import os
import locale

# Windows encoding fix - must be before sentence_transformers import
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"
try:
    locale.setlocale(locale.LC_ALL, "C")
except:
    pass

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union


class BGEModel:
    """
    BGE-M3 embedding model
    
    State-of-the-art multilingual embedding model
    - Dimension: 1024
    - Best for: Academic text retrieval, multi-lingual support
    - Usage: Primary embedder for ChromaDB indexing and retrieval
    """

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        """
        Initialize BGE-M3 model

        Args:
            model_name: HuggingFace model name
        """
        print(f"[*] Loading {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dimension = 1024
        print(f"✅ {model_name} loaded (dim: {self.dimension})")

    def encode(
        self, 
        texts: Union[str, List[str]], 
        normalize: bool = True, 
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode texts to embeddings

        Args:
            texts: Single text or list of texts
            normalize: Normalize embeddings (recommended for cosine similarity)
            batch_size: Batch size for encoding
            show_progress: Show progress bar

        Returns:
            Numpy array of embeddings
            - Single text: shape (1024,)
            - Multiple texts: shape (n, 1024)
        """
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False
            
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=normalize,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        if single_input:
            return embeddings[0]
        return embeddings

    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimension

    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "max_seq_length": self.model.max_seq_length,
            "type": "primary-embedder",
            "best_for": "Academic text, multilingual"
        }


class PubMedBERTModel:
    """
    PubMedBERT model for medical/scientific domain reranking
    
    Specialized for biomedical and scientific text
    - Dimension: 768
    - Best for: Medical/scientific domain reranking
    - Usage: Secondary reranker after primary BGE-M3 retrieval
    """

    def __init__(self, model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO"):
        """
        Initialize PubMedBERT model

        Args:
            model_name: HuggingFace model name
        """
        print(f"[*] Loading {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dimension = 768
        print(f"✅ {model_name} loaded (dim: {self.dimension})")

    def encode(
        self, 
        texts: Union[str, List[str]], 
        normalize: bool = True, 
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode texts to embeddings

        Args:
            texts: Single text or list of texts
            normalize: Normalize embeddings (recommended for cosine similarity)
            batch_size: Batch size for encoding
            show_progress: Show progress bar

        Returns:
            Numpy array of embeddings
            - Single text: shape (768,)
            - Multiple texts: shape (n, 768)
        """
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False
            
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=normalize,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        if single_input:
            return embeddings[0]
        return embeddings

    def rerank(self, query: str, documents: List[str]) -> List[float]:
        """
        Rerank documents for a query using PubMedBERT

        Args:
            query: Query text
            documents: List of documents to rerank

        Returns:
            List of relevance scores (cosine similarity)
        """
        query_emb = self.encode(query, normalize=True)
        doc_embs = self.encode(documents, normalize=True)

        # Compute cosine similarity
        scores = np.dot(doc_embs, query_emb.T).flatten()
        return scores.tolist()

    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimension

    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "max_seq_length": self.model.max_seq_length,
            "type": "reranker",
            "best_for": "Medical/biomedical domain"
        }


def test_models():
    """Test both embedding models"""
    print("\n" + "=" * 60)
    print("EMBEDDING MODELS TEST")
    print("=" * 60)
    
    # Test BGE-M3
    print("\n[1] Testing BGE-M3 Primary Embedder")
    bge = BGEModel()
    
    test_texts = [
        "Power analysis determines the sample size needed for statistical significance.",
        "ANOVA is used to compare means across multiple groups.",
        "Regression analysis predicts outcomes based on predictor variables."
    ]
    
    embeddings = bge.encode(test_texts)
    print(f"   Shape: {embeddings.shape}")
    print(f"   Model info: {bge.get_model_info()}")
    
    # Test PubMedBERT
    print("\n[2] Testing PubMedBERT Reranker")
    pubmed = PubMedBERTModel()
    
    query = "What statistical test should I use for comparing three groups?"
    
    scores = pubmed.rerank(query, test_texts)
    print(f"   Reranking scores for query: '{query}'")
    for i, (text, score) in enumerate(zip(test_texts, scores), 1):
        print(f"   {i}. Score {score:.4f}: {text[:50]}...")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_models()
