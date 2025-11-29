"""
Biostatistics RAG Configuration
================================
Configuration for biostatistics methodology knowledge base

@author: Dr. Mahir Kurt + Claude Code
@date: 2025-11-24
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths - Use absolute path to avoid working directory issues
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TEXTBOOKS_DIR = PROJECT_ROOT / "textbooks"
CHROMA_DB_PATH = PROJECT_ROOT / "rag_system" / "chroma_db"
LOG_DIR = PROJECT_ROOT / "rag_system" / "logs"

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Collection
COLLECTION_NAME = "biostatistics_main"

# Embedding Models
PRIMARY_MODEL_NAME = "BAAI/bge-m3"  # BGE-M3 (1024-dim) - Primary embedder
SECONDARY_MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"  # PubMedBERT - Medical domain reranker

# Retrieval Settings
TOP_K_RETRIEVE = 50  # Primary model retrieves top-50
TOP_K_FINAL = 20     # Re-rank to top-20

# Chunking Settings
CHUNK_SIZE = 1000      # Characters per chunk
CHUNK_OVERLAP = 200    # Overlap between chunks
SEMANTIC_CHUNKING = True  # Break at sentence boundaries

# LLM Settings
LLM_MODEL = "claude-sonnet-4-20250514"  # Claude Sonnet 4.5
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 8192

# Create directories
CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
TEXTBOOKS_DIR.mkdir(parents=True, exist_ok=True)

# Validation
def validate_config():
    """Validate configuration"""
    issues = []

    if not TEXTBOOKS_DIR.exists():
        issues.append(f"Textbooks directory not found: {TEXTBOOKS_DIR}")

    if not ANTHROPIC_API_KEY and "claude" in LLM_MODEL:
        issues.append("ANTHROPIC_API_KEY not set (required for Claude)")

    if not GOOGLE_API_KEY and "gemini" in LLM_MODEL:
        issues.append("GOOGLE_API_KEY not set (required for Gemini)")

    if issues:
        print("WARNING:  Configuration warnings:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("OK: Biostatistics RAG config validated")

    return len(issues) == 0


# Auto-validate on import
print("[Biostatistics RAG Configuration]")
print(f"   Project root: {PROJECT_ROOT}")
print(f"   Textbooks: {TEXTBOOKS_DIR}")
print(f"   ChromaDB: {CHROMA_DB_PATH}")
print(f"   Collection: {COLLECTION_NAME}")
print(f"   Primary model: {PRIMARY_MODEL_NAME}")
print(f"   Secondary model: {SECONDARY_MODEL_NAME}")

validate_config()
