"""
Verbose Biostatistics Indexer Wrapper
======================================
Shows detailed progress for indexing process

@author: Dr. Mahir Kurt + Claude Code
@date: 2025-11-24
"""

import sys
import time
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 80)
print("BIOSTATISTICS RAG - VERBOSE INDEXER")
print("=" * 80)
print()

# Step 1: Import shared components
print("[Step 1/6] Importing shared components...")
sys.path.insert(0, "D:/Repositories/shared_rag_components")

try:
    print("   - Importing BGEModel...")
    from embeddings import BGEModel
    print("   OK: BGEModel imported")
except Exception as e:
    print(f"   ERROR: {e}")
    sys.exit(1)

try:
    print("   - Importing ChromaDB...")
    import chromadb
    from chromadb.config import Settings
    print("   OK: ChromaDB imported")
except Exception as e:
    print(f"   ERROR: {e}")
    sys.exit(1)

try:
    print("   - Importing PyPDF2...")
    from PyPDF2 import PdfReader
    print("   OK: PyPDF2 imported")
except Exception as e:
    print(f"   ERROR: {e}")
    sys.exit(1)

print()

# Step 2: Load configuration
print("[Step 2/6] Loading configuration...")
try:
    from config.biostat_rag_config import *
    print(f"   OK: Config loaded")
    print(f"   - Collection: {COLLECTION_NAME}")
    print(f"   - Primary model: {PRIMARY_MODEL_NAME}")
    print(f"   - ChromaDB path: {CHROMA_DB_PATH}")
except Exception as e:
    print(f"   ERROR: {e}")
    sys.exit(1)

print()

# Step 3: Initialize model
print("[Step 3/6] Loading BGE-M3 model...")
print("   Model should be cached from Medical Writing indexing")
start = time.time()

try:
    model = BGEModel(PRIMARY_MODEL_NAME)
    elapsed = time.time() - start
    print(f"   OK: Model loaded in {elapsed:.1f} seconds")
except Exception as e:
    print(f"   ERROR: {e}")
    sys.exit(1)

print()

# Step 4: Initialize ChromaDB
print("[Step 4/6] Initializing ChromaDB...")
try:
    client = chromadb.PersistentClient(
        path=str(CHROMA_DB_PATH),
        settings=Settings(anonymized_telemetry=False, allow_reset=True)
    )
    
    # Delete existing
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"   - Deleted existing collection")
    except:
        pass
    
    # Create new
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"   OK: Collection '{COLLECTION_NAME}' ready")
except Exception as e:
    print(f"   ERROR: {e}")
    sys.exit(1)

print()

# Step 5: Find PDFs
print("[Step 5/6] Finding PDF files...")
pdf_files = list(TEXTBOOKS_DIR.glob("*.pdf"))
print(f"   Found {len(pdf_files)} PDFs in {TEXTBOOKS_DIR}")

if not pdf_files:
    print("   ERROR: No PDF files found!")
    sys.exit(1)

print()

# Step 6: Index PDFs
print("[Step 6/6] Indexing PDFs...")
print(f"   Chunk size: {CHUNK_SIZE} chars (overlap: {CHUNK_OVERLAP})")
print(f"   WARNING: This will take 1-2 hours for 45 PDFs")
print()

total_chunks = 0
failed = []

for idx, pdf_path in enumerate(pdf_files, 1):
    print(f"[{idx}/{len(pdf_files)}] {pdf_path.name}")
    
    try:
        # Extract text
        print(f"   [1/3] Extracting text...", end=" ", flush=True)
        reader = PdfReader(str(pdf_path))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        text = text.strip()
        
        if not text:
            print("SKIP: No text")
            failed.append((pdf_path.name, "No text extracted"))
            continue
        
        print(f"OK ({len(text):,} chars)")
        
        # Chunk
        print(f"   [2/3] Chunking text...", end=" ", flush=True)
        chunks = []
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            if end < len(text):
                for delimiter in ['. ', '? ', '! ', '\n\n']:
                    last_delim = text[start:end].rfind(delimiter)
                    if last_delim != -1:
                        end = start + last_delim + len(delimiter)
                        break
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - CHUNK_OVERLAP if end < len(text) else end
        
        print(f"OK ({len(chunks)} chunks)")
        
        # Index
        print(f"   [3/3] Indexing...", end=" ", flush=True)
        for i, chunk in enumerate(chunks):
            embedding = model.encode(chunk)
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()
            
            doc_id = f"{pdf_path.stem}_chunk_{i}"
            metadata = {
                "source": pdf_path.name,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "type": "biostatistics"
            }
            
            collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[metadata]
            )
        
        total_chunks += len(chunks)
        print(f"OK ({len(chunks)} chunks)")
        
    except Exception as e:
        print(f"ERROR: {e}")
        failed.append((pdf_path.name, str(e)))
        continue
    
    print()

# Summary
print("=" * 80)
print("INDEXING COMPLETE")
print("=" * 80)
print(f"Total PDFs: {len(pdf_files)}")
print(f"Successful: {len(pdf_files) - len(failed)}")
print(f"Failed: {len(failed)}")
print(f"Total chunks: {total_chunks:,}")
print(f"Collection: {COLLECTION_NAME}")
print(f"ChromaDB path: {CHROMA_DB_PATH}")

if failed:
    print()
    print("Failed PDFs:")
    for name, error in failed:
        print(f"  - {name}: {error}")

print("=" * 80)
