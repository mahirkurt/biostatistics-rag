"""
Embedding Models for Biostatistics RAG
=======================================

Available models:
- MiniLMEmbedder: Lightweight (80MB, 384-dim) - RECOMMENDED
- MiniLMAdapter: MiniLM with padding to 1024-dim
- BGEModel: State-of-the-art (BAAI/bge-m3, 1024-dim) - requires PyTorch
- NomicModel: Long-context (nomic-ai/nomic-embed-text-v1.5, 768-dim) - requires PyTorch

Note: BGEModel and NomicModel have issues on Python 3.13 + PyTorch 2.8
      Use MiniLM or lazy imports for those models.

@author: Dr. Mahir Kurt + Claude Code
@date: 2025-11-24
"""

# Safe imports (lightweight, no PyTorch issues)
from .minilm_embedder import MiniLMEmbedder, MiniLMAdapter

# Lazy imports for PyTorch-based models (avoid import on module load)
def get_bge_model():
    """Lazy import for BGEModel (requires PyTorch)"""
    from .bge_m3 import BGEModel
    return BGEModel

def get_nomic_model():
    """Lazy import for NomicModel (requires PyTorch)"""
    from .nomic_v15 import NomicModel
    return NomicModel

# For backward compatibility (will trigger PyTorch import)
BGEModel = None
NomicModel = None

def __getattr__(name):
    """Lazy loading for backward compatibility"""
    global BGEModel, NomicModel
    if name == "BGEModel":
        if BGEModel is None:
            from .bge_m3 import BGEModel as _BGE
            BGEModel = _BGE
        return BGEModel
    elif name == "NomicModel":
        if NomicModel is None:
            from .nomic_v15 import NomicModel as _Nomic
            NomicModel = _Nomic
        return NomicModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "MiniLMEmbedder",
    "MiniLMAdapter", 
    "get_bge_model",
    "get_nomic_model",
    "BGEModel",  # Lazy loaded
    "NomicModel",  # Lazy loaded
]

__version__ = "1.0.0"
