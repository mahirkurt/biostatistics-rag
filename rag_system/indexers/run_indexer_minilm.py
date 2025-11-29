"""
MiniLM-based Indexer (Lightweight Alternative)
===============================================
Uses all-MiniLM-L6-v2 (80MB) with padding to 1024-dim
Faster loading, avoids PyTorch issues on this system

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

# Mock platform functions
import platform
platform.system = lambda: "Windows"
platform.win32_ver = lambda: ("10", "10.0.19041", "", "Multiprocessor Free")

print("=" * 80)
print("BIOSTATISTICS RAG - MINILM INDEXER")
print("=" * 80)
print("NOTE: Using lightweight MiniLM model (384-dim padded to 1024-dim)")
print()

# Setup paths
print("[1/5] Setting up paths...")
sys.path.insert(0, str(Path(__file__).parent.parent))
print("   OK\n")

# Import
print("[2/5] Importing dependencies...")
try:
    from PyPDF2 import PdfReader
    import chromadb
    
    # Direct import to avoid __init__.py triggering PyTorch
    import sys
    from pathlib import Path
    embeddings_dir = Path(__file__).parent.parent / "embeddings"
    sys.path.insert(0, str(embeddings_dir.parent))
    
    from embeddings.minilm_embedder import MiniLMAdapter
    print("   OK\n")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Load config
print("[3/5] Loading configuration...")
try:
    from config.biostat_rag_config import (
        TEXTBOOKS_DIR, CHROMA_DB_PATH, COLLECTION_NAME,
        CHUNK_SIZE, CHUNK_OVERLAP
    )
    print(f"   Collection: {COLLECTION_NAME}")
    print(f"   PDFs: {TEXTBOOKS_DIR}\n")
except Exception as e:
    print(f"   ERROR: {e}")
    sys.exit(1)

# ChromaDB
print("[4/5] Initializing ChromaDB...")
try:
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    try:
        client.delete_collection(COLLECTION_NAME)
        print("   - Deleted existing collection")
    except:
        pass
    
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    print("   OK\n")
except Exception as e:
    print(f"   ERROR: {e}")
    sys.exit(1)

# Load model
print("[5/5] Loading MiniLM model...")
try:
    model = MiniLMAdapter()
    print("   OK\n")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Chunking function
def chunk_text(text: str, size: int = 1000, overlap: int = 200) -> list:
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + size, text_len)
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
print("Estimated time: 1-2 hours\n")

# Index
total_chunks = 0
failed = []

for idx, pdf_path in enumerate(pdf_files, 1):
    print(f"[{idx}/{len(pdf_files)}] {pdf_path.stem}")
    
    try:
        # Extract
        print("   Extracting...", end=" ", flush=True)
        reader = PdfReader(str(pdf_path))
        text = "".join(page.extract_text() or "" for page in reader.pages)
        print(f"OK ({len(text):,} chars)")
        
        if not text.strip():
            print("   [SKIP] No text")
            failed.append((pdf_path.name, "No text"))
            continue
        
        # Chunk
        print("   Chunking...", end=" ", flush=True)
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"OK ({len(chunks)} chunks)")
        
        # Embed
        print("   Embedding...", end=" ", flush=True)
        embeddings = model.encode(chunks, batch_size=16, normalize=True)
        print("OK")
        
        # Add to DB
        print("   Saving...", end=" ", flush=True)
        ids = [f"{pdf_path.stem}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{
            "source": pdf_path.name,
            "chunk_index": i,
            "total_chunks": len(chunks)
        } for i in range(len(chunks))]
        
        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )
        print("OK")
        
        total_chunks += len(chunks)
        print(f"   [SUCCESS] {len(chunks)} chunks\n")
        
        del text, chunks, embeddings
        gc.collect()
        
    except Exception as e:
        print(f"\n   [FAILED] {e}\n")
        failed.append((pdf_path.name, str(e)))

print("=" * 80)
print("INDEXING COMPLETE")
print("=" * 80)
print(f"Total chunks: {total_chunks:,}")
print(f"Success: {len(pdf_files) - len(failed)}/{len(pdf_files)} PDFs")

if failed:
    print(f"\nFailed ({len(failed)}):")
    for name, reason in failed:
        print(f"  - {name}: {reason}")

print(f"\nCollection: {COLLECTION_NAME}")
print(f"Path: {CHROMA_DB_PATH}")
