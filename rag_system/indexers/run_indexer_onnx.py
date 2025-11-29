"""
ONNX-based Indexer for Biostatistics RAG
=========================================
Uses ONNX Runtime instead of PyTorch to avoid torch import issues

@author: Dr. Mahir Kurt + Claude Code
@date: 2025-11-25
"""

# FIX ChromaDB encoding BEFORE any imports
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

import sys
import gc
from pathlib import Path

# Mock platform functions to avoid Windows WMI errors
import platform
def _mock_system():
    return "Windows"
def _mock_win32_ver():
    return ("10", "10.0.19041", "", "Multiprocessor Free")
platform.system = _mock_system
platform.win32_ver = _mock_win32_ver

print("=" * 80)
print("BIOSTATISTICS RAG - ONNX INDEXER")
print("=" * 80)
print()

# Step 1: Setup paths
print("[1/5] Setting up paths...")
sys.path.insert(0, str(Path(__file__).parent.parent))
print("   OK")
print()

# Step 2: Import dependencies
print("[2/5] Importing dependencies...")
try:
    from PyPDF2 import PdfReader
    import chromadb
    print("   - PyPDF2 and ChromaDB: OK")
    
    # Import ONNX embedder
    from embeddings.onnx_embedder import ONNXEmbedder
    print("   - ONNXEmbedder: OK")
    print("   OK")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Step 3: Load config
print("[3/5] Loading configuration...")
try:
    from config.biostat_rag_config import (
        TEXTBOOKS_DIR, CHROMA_DB_PATH, COLLECTION_NAME,
        PRIMARY_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP
    )
    print(f"   Collection: {COLLECTION_NAME}")
    print(f"   PDFs: {TEXTBOOKS_DIR}")
    print(f"   Model: {PRIMARY_MODEL_NAME} (ONNX Runtime)")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Step 4: Initialize ChromaDB
print("[4/5] Initializing ChromaDB...")
try:
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    
    try:
        client.delete_collection(COLLECTION_NAME)
        print("   - Deleted existing collection")
    except Exception:
        pass
    
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    print("   OK: Collection ready")
    
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Step 5: Load ONNX model
print("[5/5] Loading ONNX model...")
print("   Converting PyTorch model to ONNX (first time takes ~2 minutes)...", flush=True)

try:
    model = ONNXEmbedder(PRIMARY_MODEL_NAME)
    print("   OK: Model ready")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Helper functions
def chunk_text(text: str, size: int = 1000, overlap: int = 200) -> list:
    """Chunk text with overlap - optimized for large texts"""
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + size, text_len)
        
        # Only search for delimiter in last 200 chars (optimization)
        if end < text_len:
            search_start = max(start, end - 200)
            search_text = text[search_start:end]
            
            for delimiter in ['. ', '? ', '! ', '\n\n']:
                last_delim = search_text.rfind(delimiter)
                if last_delim != -1:
                    end = search_start + last_delim + len(delimiter)
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap if end < text_len else end
    
    return chunks

# Find PDFs
pdf_files = list(TEXTBOOKS_DIR.glob("*.pdf"))
if not pdf_files:
    print("ERROR: No PDFs found")
    sys.exit(1)

print(f"Found {len(pdf_files)} PDFs")
print("WARNING: This will take 1-2 hours for 45 PDFs")
print()

# Index with progress
total_chunks = 0
failed = []

for idx, pdf_path in enumerate(pdf_files, 1):
    print(f"[{idx}/{len(pdf_files)}] {pdf_path.stem}")
    
    try:
        # Extract text
        print("   Extracting...", end=" ", flush=True)
        reader = PdfReader(str(pdf_path))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        print(f"OK ({len(text):,} chars)")
        
        if not text.strip():
            print("   [SKIP] No text extracted")
            failed.append((pdf_path.name, "No text"))
            continue
        
        # Chunk
        print("   Chunking...", end=" ", flush=True)
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"OK ({len(chunks)} chunks)")
        
        # Embed in batches
        print("   Embedding...", end=" ", flush=True)
        batch_size = 8  # Smaller batches for ONNX
        all_embeddings = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            embeddings = model.encode(batch, normalize=True)
            all_embeddings.append(embeddings)
            
            # Progress indicator
            if (i // batch_size + 1) % 10 == 0:
                print(f"{i + len(batch)}/{len(chunks)}", end="... ", flush=True)
        
        import numpy as np
        all_embeddings = np.vstack(all_embeddings)
        print("OK")
        
        # Add to ChromaDB
        print("   Adding to DB...", end=" ", flush=True)
        ids = [f"{pdf_path.stem}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{
            "source": pdf_path.name,
            "chunk_index": i,
            "total_chunks": len(chunks)
        } for i in range(len(chunks))]
        
        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=all_embeddings.tolist(),
            metadatas=metadatas
        )
        print("OK")
        
        total_chunks += len(chunks)
        print(f"   [SUCCESS] {len(chunks)} chunks indexed")
        
        # Cleanup
        del text, chunks, all_embeddings
        gc.collect()
        
    except Exception as e:
        print(f"\n   [FAILED] {e}")
        failed.append((pdf_path.name, str(e)))
        continue

print()
print("=" * 80)
print("INDEXING COMPLETE")
print("=" * 80)
print(f"Total chunks indexed: {total_chunks:,}")
print(f"Successfully processed: {len(pdf_files) - len(failed)}/{len(pdf_files)} PDFs")

if failed:
    print(f"\nFailed ({len(failed)}):")
    for name, reason in failed:
        print(f"  - {name}: {reason}")

print(f"\nCollection: {COLLECTION_NAME}")
print(f"Path: {CHROMA_DB_PATH}")
