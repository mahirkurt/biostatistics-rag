"""
Pure ONNX MiniLM Embedder (No PyTorch!)
========================================
Uses ONNX Runtime directly without sentence-transformers
This avoids all PyTorch import issues on Python 3.13

@author: Dr. Mahir Kurt + Claude Code
@date: 2025-11-25
"""

import numpy as np
from typing import List, Union
from pathlib import Path
import os


class PureONNXMiniLM:
    """
    MiniLM embedder using pure ONNX Runtime
    No PyTorch dependency at all
    """
    
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    DIMENSION = 384
    
    def __init__(self):
        print(f"[*] Loading MiniLM with pure ONNX Runtime...")
        print("    No PyTorch dependency!")
        
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            print(f"[ERROR] Missing dependency: {e}")
            print("Install: pip install onnxruntime tokenizers huggingface_hub")
            raise
        
        # Download tokenizer
        print("    Loading tokenizer...", end=" ", flush=True)
        tokenizer_path = hf_hub_download(
            repo_id=self.MODEL_NAME,
            filename="tokenizer.json"
        )
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self.tokenizer.enable_truncation(max_length=256)
        print("OK")
        
        # Download or convert ONNX model
        print("    Loading ONNX model...", end=" ", flush=True)
        try:
            # Try to download pre-converted ONNX model
            onnx_path = hf_hub_download(
                repo_id="sentence-transformers/all-MiniLM-L6-v2",
                filename="onnx/model.onnx",
                subfolder=""
            )
        except Exception:
            # Use optimum to export if not available
            onnx_path = self._export_to_onnx()
        
        # Create ONNX session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            onnx_path,
            sess_options,
            providers=['CPUExecutionProvider']
        )
        print("OK")
        
        self.dimension = self.DIMENSION
        print(f"[OK] Pure ONNX MiniLM loaded (dimension: {self.dimension})")
    
    def _export_to_onnx(self) -> str:
        """Export model to ONNX using optimum (one-time)"""
        print("\n    Converting to ONNX (one-time operation)...")
        
        try:
            from optimum.onnxruntime import ORTModelForFeatureExtraction
            
            cache_dir = Path.home() / ".cache" / "onnx_models" / "minilm"
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            onnx_path = cache_dir / "model.onnx"
            
            if not onnx_path.exists():
                model = ORTModelForFeatureExtraction.from_pretrained(
                    self.MODEL_NAME,
                    export=True
                )
                model.save_pretrained(str(cache_dir))
            
            return str(onnx_path)
            
        except Exception as e:
            raise RuntimeError(f"Failed to export ONNX model: {e}")
    
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
            numpy array of embeddings (N x 384)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Tokenize
            encoded = self.tokenizer.encode_batch(batch)
            
            input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
            attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
            token_type_ids = np.zeros_like(input_ids, dtype=np.int64)
            
            # Run ONNX inference
            outputs = self.session.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids
                }
            )
            
            # Mean pooling
            token_embeddings = outputs[0]  # (batch, seq_len, hidden)
            
            # Apply attention mask for mean pooling
            mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
            sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
            sum_mask = np.sum(mask_expanded, axis=1)
            embeddings = sum_embeddings / np.maximum(sum_mask, 1e-9)
            
            all_embeddings.append(embeddings)
        
        result = np.vstack(all_embeddings).astype(np.float32)
        
        # Normalize
        if normalize:
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            result = result / np.maximum(norms, 1e-9)
        
        return result


class PureONNXMiniLMAdapter:
    """
    Adapter that pads 384-dim to 1024-dim for ChromaDB compatibility
    """
    
    def __init__(self):
        self.embedder = PureONNXMiniLM()
        self.dimension = 1024
        print("[INFO] Will pad 384-dim to 1024-dim for ChromaDB")
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        """Encode and pad to 1024 dimensions"""
        
        # Get 384-dim embeddings
        embeddings = self.embedder.encode(texts, batch_size, normalize=False)
        
        # Pad to 1024-dim
        padded = np.zeros((embeddings.shape[0], 1024), dtype=np.float32)
        padded[:, :384] = embeddings
        
        # Normalize after padding
        if normalize:
            norms = np.linalg.norm(padded, axis=1, keepdims=True)
            padded = padded / np.maximum(norms, 1e-9)
        
        return padded


def test_pure_onnx():
    """Test the pure ONNX embedder"""
    print("Testing Pure ONNX MiniLM (no PyTorch)...")
    print()
    
    try:
        embedder = PureONNXMiniLM()
        
        texts = [
            "What is the difference between t-test and ANOVA?",
            "How to calculate p-value?"
        ]
        
        embeddings = embedder.encode(texts)
        print(f"\n[TEST] Generated {embeddings.shape[0]} embeddings")
        print(f"       Dimension: {embeddings.shape[1]}")
        print(f"       Sample: {embeddings[0][:5]}")
        
        # Test adapter
        print("\nTesting adapter (with padding)...")
        adapter = PureONNXMiniLMAdapter()
        padded = adapter.encode(texts)
        print(f"[TEST] Padded dimension: {padded.shape[1]}")
        
        print("\n[SUCCESS] Pure ONNX embedder works!")
        return True
        
    except Exception as e:
        print(f"\n[FAILED] {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_pure_onnx()
