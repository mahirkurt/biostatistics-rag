"""
Pure ONNX Indexer - PyTorch-free biostatistics PDF indexer

Uses:
- pure_onnx_minilm.py (no PyTorch, no sentence_transformers)
- PyPDF2 for PDF extraction  
- ChromaDB 1.3.5 for vector storage

This is the recommended indexer for Python 3.13 where PyTorch has issues.
"""

import os
import sys

# === CRITICAL: Set environment BEFORE any imports ===
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
# Note: CHROMA_DB_IMPL is deprecated in ChromaDB 1.3.5
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TRANSFORMERS_OFFLINE'] = '0'

# === Mock platform functions BEFORE chromadb import ===
import platform
_original_system = platform.system
_original_win32_ver = platform.win32_ver

def _safe_system():
    return "Windows"

def _safe_win32_ver():
    return ("10", "10.0.19041", "", "Multiprocessor Free")

platform.system = _safe_system
platform.win32_ver = _safe_win32_ver

print("=" * 60)
print("Pure ONNX Biostatistics Indexer")
print("No PyTorch, No sentence_transformers")
print("=" * 60)

# Now safe to import
import chromadb
import PyPDF2
import hashlib
import time
import gc
from pathlib import Path
from datetime import datetime

# Add parent to path for embeddings import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import our pure ONNX embedder (no PyTorch!)
from rag_system.embeddings.pure_onnx_minilm import PureONNXMiniLMAdapter

# === CONFIGURATION ===
TEXTBOOKS_DIR = Path(__file__).parent.parent.parent / "textbooks"
CHROMA_DB_PATH = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "biostatistics_main"

# Chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Batch settings (for memory efficiency)
PDF_BATCH_SIZE = 3  # Process 3 PDFs at a time
EMBED_BATCH_SIZE = 32  # Embed 32 chunks at a time


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from PDF using PyPDF2."""
    text = ""
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"    [ERROR] PDF extraction failed: {e}")
    return text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """Split text into overlapping chunks."""
    if not text or len(text) < 100:
        return []
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        
        # Clean chunk
        chunk = chunk.strip()
        if len(chunk) >= 100:  # Minimum chunk size
            chunks.append(chunk)
        
        # Move to next chunk with overlap
        start = end - overlap if end < text_len else text_len
        
        # Prevent infinite loop
        if start >= text_len - 50:
            break
    
    return chunks


def generate_chunk_id(pdf_name: str, chunk_index: int) -> str:
    """Generate unique ID for a chunk."""
    content = f"{pdf_name}_{chunk_index}"
    return hashlib.md5(content.encode()).hexdigest()


def index_pdfs():
    """Main indexing function."""
    print(f"\n[1/4] Initializing ChromaDB...")
    print(f"      Path: {CHROMA_DB_PATH}")
    
    # Ensure directory exists
    CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
    
    # Initialize ChromaDB (1.3.5 API - no Settings)
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    
    # Get or create collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    
    existing_count = collection.count()
    print(f"      Collection: {COLLECTION_NAME}")
    print(f"      Existing documents: {existing_count}")
    
    # Get existing IDs to skip
    existing_ids = set()
    if existing_count > 0:
        try:
            result = collection.get(include=[])
            existing_ids = set(result['ids'])
            print(f"      Will skip already indexed chunks")
        except:
            pass
    
    print(f"\n[2/4] Loading embedding model...")
    embedder = PureONNXMiniLMAdapter()  # 384-dim padded to 1024
    print(f"      Model: MiniLM-L6-v2 (Pure ONNX)")
    print(f"      Dimension: 1024 (padded from 384)")
    
    print(f"\n[3/4] Discovering PDFs...")
    pdf_files = list(TEXTBOOKS_DIR.glob("*.pdf"))
    print(f"      Found: {len(pdf_files)} PDFs in {TEXTBOOKS_DIR}")
    
    if not pdf_files:
        print("      [ERROR] No PDF files found!")
        return
    
    print(f"\n[4/4] Processing PDFs...")
    print("-" * 60)
    
    total_chunks_added = 0
    total_pdfs_processed = 0
    start_time = time.time()
    
    for i, pdf_path in enumerate(pdf_files, 1):
        pdf_name = pdf_path.name
        print(f"\n[{i}/{len(pdf_files)}] {pdf_name}")
        
        # Extract text
        print(f"    Extracting text...", end=" ", flush=True)
        text = extract_text_from_pdf(pdf_path)
        
        if not text or len(text) < 200:
            print(f"SKIP (no content)")
            continue
        print(f"OK ({len(text):,} chars)")
        
        # Chunk text
        print(f"    Chunking...", end=" ", flush=True)
        chunks = chunk_text(text)
        print(f"OK ({len(chunks)} chunks)")
        
        if not chunks:
            continue
        
        # Generate IDs and filter existing
        chunk_ids = [generate_chunk_id(pdf_name, j) for j in range(len(chunks))]
        new_chunks = []
        new_ids = []
        
        for chunk, chunk_id in zip(chunks, chunk_ids):
            if chunk_id not in existing_ids:
                new_chunks.append(chunk)
                new_ids.append(chunk_id)
        
        if not new_chunks:
            print("    SKIP (all chunks already indexed)")
            continue
        
        print(f"    New chunks to index: {len(new_chunks)}")
        
        # Embed and save in small batches for resilience
        print("    Processing...", end=" ", flush=True)
        chunks_added_this_pdf = 0
        
        for batch_start in range(0, len(new_chunks), EMBED_BATCH_SIZE):
            batch_end = min(batch_start + EMBED_BATCH_SIZE, len(new_chunks))
            batch_chunks = new_chunks[batch_start:batch_end]
            batch_ids = new_ids[batch_start:batch_end]
            
            # Embed this batch
            batch_embeddings = embedder.encode(batch_chunks)
            
            # Metadata for this batch
            batch_metadatas = [
                {
                    "source": pdf_name,
                    "chunk_index": batch_start + j,
                    "total_chunks": len(chunks)
                }
                for j in range(len(batch_chunks))
            ]
            
            # Save immediately to ChromaDB
            collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings.tolist(),
                documents=batch_chunks,
                metadatas=batch_metadatas
            )
            
            chunks_added_this_pdf += len(batch_chunks)
            existing_ids.update(batch_ids)
            print(".", end="", flush=True)
        
        print(f" OK ({chunks_added_this_pdf} saved)")
        
        total_chunks_added += chunks_added_this_pdf
        total_pdfs_processed += 1
        
        # Memory cleanup every few PDFs
        if i % PDF_BATCH_SIZE == 0:
            gc.collect()
            print(f"\n    [CHECKPOINT] {total_chunks_added} chunks indexed so far")
    
    # Final summary
    elapsed = time.time() - start_time
    final_count = collection.count()
    
    print("\n" + "=" * 60)
    print("INDEXING COMPLETE")
    print("=" * 60)
    print(f"PDFs processed:     {total_pdfs_processed}/{len(pdf_files)}")
    print(f"Chunks added:       {total_chunks_added}")
    print(f"Total in collection:{final_count}")
    print(f"Time elapsed:       {elapsed/60:.1f} minutes")
    print(f"Timestamp:          {datetime.now().isoformat()}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        index_pdfs()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Indexing stopped by user")
        print("Progress has been saved. Run again to continue.")
    except Exception as e:
        print(f"\n\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
