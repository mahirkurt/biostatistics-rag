"""
Lightweight Biostatistics Indexer
==================================
Memory-efficient version with batch processing

@author: Dr. Mahir Kurt + Claude Code
@date: 2025-11-24
"""

import sys
import gc
from pathlib import Path

# Fix ChromaDB encoding issue on Windows + Memory optimization
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['CHROMA_DB_IMPL'] = 'sqlite'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

print("=" * 80)
print("BIOSTATISTICS RAG - LIGHTWEIGHT INDEXER")
print("=" * 80)
print()

# Step 1: Setup paths
print("[1/5] Setting up paths...")
sys.path.insert(0, "D:/Repositories/shared_rag_components")
sys.path.insert(0, str(Path(__file__).parent.parent))
print("   OK")
print()

# Step 2: Import minimal dependencies
print("[2/5] Importing dependencies...")
try:
    from PyPDF2 import PdfReader
    import chromadb
    print("   OK")
except Exception as e:
    print(f"   ERROR: {e}")
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
    print(f"   Model: {PRIMARY_MODEL_NAME}")
except Exception as e:
    print(f"   ERROR: {e}")
    sys.exit(1)

print()

# Step 4: Initialize ChromaDB (using simple pattern from CF project)
print("[4/5] Initializing ChromaDB...")
try:
    # Use simple initialization pattern that works in CF project
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
    print(f"   OK: Collection ready")
except Exception as e:
    print(f"   ERROR: {e}")
    sys.exit(1)

print()

# Step 5: NOW load the model (after ChromaDB is ready)
print("[5/5] Loading BGE-M3 model...")
print("   Model should be cached from previous indexing")
print("   Loading...", flush=True)

try:
    from embeddings import BGEModel
    model = BGEModel(PRIMARY_MODEL_NAME)
    print("   OK: Model loaded")
except Exception as e:
    print(f"   ERROR: {e}")
    sys.exit(1)

print()

# Helper functions
def chunk_text(text, size=1000, overlap=200):
    """Chunk text with overlap"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        if end < len(text):
            for delimiter in ['. ', '? ', '! ', '\n\n']:
                last_delim = text[start:end].rfind(delimiter)
                if last_delim != -1:
                    end = start + last_delim + len(delimiter)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap if end < len(text) else end
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
    print(f"[{idx}/{len(pdf_files)}] {pdf_path.name[:50]}")
    
    try:
        # Extract
        print("   Extracting...", end=" ", flush=True)
        reader = PdfReader(str(pdf_path))
        text = "".join(page.extract_text() or "" for page in reader.pages).strip()
        
        if not text:
            print("SKIP (no text)")
            continue
        
        print(f"OK ({len(text):,} chars)")
        
        # Chunk
        print("   Chunking...", end=" ", flush=True)
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"OK ({len(chunks)} chunks)")
        
        # Index in batches
        print("   Indexing...", end=" ", flush=True)
        BATCH_SIZE = 10
        
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i+BATCH_SIZE]
            
            for j, chunk in enumerate(batch):
                chunk_idx = i + j
                embedding = model.encode(chunk)
                
                if hasattr(embedding, 'tolist'):
                    embedding = embedding.tolist()
                
                collection.add(
                    ids=[f"{pdf_path.stem}_chunk_{chunk_idx}"],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{
                        "source": pdf_path.name,
                        "chunk_index": chunk_idx,
                        "total_chunks": len(chunks),
                        "type": "biostatistics"
                    }]
                )
            
            # Free memory after each batch
            gc.collect()
        
        total_chunks += len(chunks)
        print(f"OK")
        
    except Exception as e:
        print(f"ERROR: {e}")
        failed.append(pdf_path.name)
    
    # Force garbage collection between PDFs
    gc.collect()

# Summary
print()
print("=" * 80)
print("COMPLETE")
print("=" * 80)
print(f"Indexed: {len(pdf_files) - len(failed)}/{len(pdf_files)} PDFs")
print(f"Total chunks: {total_chunks:,}")
print(f"Collection: {COLLECTION_NAME}")

if failed:
    print(f"\nFailed ({len(failed)}):")
    for name in failed:
        print(f"  - {name}")

print("=" * 80)
