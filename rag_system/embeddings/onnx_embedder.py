"""
ONNX-based BGE-M3 Embedder
===========================
Alternative to PyTorch-based SentenceTransformer
Uses ONNX Runtime (lighter, fewer dependencies)

Setup:
pip install optimum onnx onnxruntime transformers tokenizers
"""

import numpy as np
from typing import List, Union
from pathlib import Path


class ONNXEmbedder:
    """BGE-M3 using ONNX Runtime instead of PyTorch"""
    
    def __init__(self, model_name: str = "BAAI/bge-m3", cache_dir: str = None):
        """
        Initialize ONNX-based embedder
        
        Args:
            model_name: HuggingFace model name
            cache_dir: Cache directory for models
        """
        print(f"[*] Loading {model_name} with ONNX Runtime...")
        
        try:
            from optimum.onnxruntime import ORTModelForFeatureExtraction
            from transformers import AutoTokenizer
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )
            
            # Load ONNX model
            self.model = ORTModelForFeatureExtraction.from_pretrained(
                model_name,
                export=True,  # Auto-convert to ONNX if needed
                cache_dir=cache_dir
            )
            
            self.dimension = 1024
            print(f"[OK] ONNX model loaded (dimension: {self.dimension})")
            
        except ImportError as e:
            print(f"[ERROR] ONNX dependencies missing: {e}")
            print("Install: pip install optimum onnx onnxruntime")
            raise
        except Exception as e:
            print(f"[ERROR] Failed to load ONNX model: {e}")
            raise
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode texts to embeddings
        
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
            
            # Tokenize
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Get embeddings
            outputs = self.model(**inputs)
            
            # Mean pooling
            embeddings = self._mean_pooling(
                outputs.last_hidden_state,
                inputs['attention_mask']
            )
            
            # Normalize if requested
            if normalize:
                embeddings = embeddings / np.linalg.norm(
                    embeddings,
                    axis=1,
                    keepdims=True
                )
            
            all_embeddings.append(embeddings.numpy())
        
        return np.vstack(all_embeddings)
    
    def _mean_pooling(self, token_embeddings, attention_mask):
        """Mean pooling with attention mask"""
        import torch
        
        input_mask_expanded = (
            attention_mask
            .unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )
        
        return torch.sum(
            token_embeddings * input_mask_expanded, 1
        ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Test function
def test_onnx_embedder():
    """Test ONNX embedder"""
    try:
        embedder = ONNXEmbedder()
        
        texts = [
            "What is the difference between t-test and ANOVA?",
            "How to calculate p-value?"
        ]
        
        embeddings = embedder.encode(texts)
        print(f"[TEST] Generated {embeddings.shape[0]} embeddings")
        print(f"       Dimension: {embeddings.shape[1]}")
        print(f"       Sample values: {embeddings[0][:5]}")
        
        return True
    except Exception as e:
        print(f"[TEST FAILED] {e}")
        return False


if __name__ == "__main__":
    test_onnx_embedder()
