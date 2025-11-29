"""
Indexer Worker - Processes a single batch of PDFs
==================================================
Called by restart-based indexer for each batch

@author: Dr. Mahir Kurt + Claude Code
@date: 2025-11-25
"""

import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

import sys
import gc
from pathlib import Path

# Mock platform
import platform
platform.system = lambda: "Windows"
platform.win32_ver = lambda: ("10", "10.0.19041", "", "Multiprocessor Free")

# Setup paths - add both base and rag_system to path
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "rag_system"))

def main(batch_file_path):
    """Process PDFs listed in batch file"""
    
    # Load batch
    batch_file = Path(batch_file_path)
    if not batch_file.exists():
        print(f"ERROR: Batch file not found: {batch_file}")
        return 1
    
    pdf_paths = []
    with open(batch_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                pdf_paths.append(Path(line))
    
    if not pdf_paths:
        print("ERROR: No PDFs in batch file")
        return 1
    
    print(f"Worker processing {len(pdf_paths)} PDFs")
    print()
    
    # Import (after environment setup)
    try:
        from PyPDF2 import PdfReader
        import chromadb
        
        # Try different import paths
        try:
            from embeddings.minilm_embedder import MiniLMAdapter
            from config.biostat_rag_config import (
                CHROMA_DB_PATH, COLLECTION_NAME,
                CHUNK_SIZE, CHUNK_OVERLAP
            )
        except ImportError:
            from rag_system.embeddings.minilm_embedder import MiniLMAdapter
            from rag_system.config.biostat_rag_config import (
                CHROMA_DB_PATH, COLLECTION_NAME,
                CHUNK_SIZE, CHUNK_OVERLAP
            )
    except Exception as e:
        print(f"ERROR importing: {e}")
        print(f"sys.path: {sys.path[:3]}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Initialize ChromaDB
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"ChromaDB ready: {COLLECTION_NAME}")
    except Exception as e:
        print(f"ERROR ChromaDB: {e}")
        return 1
    
    # Load model
    try:
        print("Loading MiniLM model...")
        model = MiniLMAdapter()
        print("Model ready")
        print()
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return 1
    
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
    
    # Process each PDF
    total_chunks = 0
    success_count = 0
    
    for idx, pdf_path in enumerate(pdf_paths, 1):
        print(f"[{idx}/{len(pdf_paths)}] {pdf_path.name}")
        
        try:
            # Extract
            print("   Extract...", end=" ", flush=True)
            reader = PdfReader(str(pdf_path))
            text = ""
            for page in reader.pages:
                try:
                    text += page.extract_text() or ""
                except:
                    pass  # Skip problematic pages
            
            print(f"{len(text):,} chars", end=" | ", flush=True)
            
            if not text.strip():
                print("[SKIP] No text")
                continue
            
            # Chunk
            print("Chunk...", end=" ", flush=True)
            chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
            print(f"{len(chunks)} chunks", end=" | ", flush=True)
            
            # Embed
            print("Embed...", end=" ", flush=True)
            embeddings = model.encode(chunks, batch_size=16, normalize=True)
            print(f"OK", end=" | ", flush=True)
            
            # Save
            print("Save...", end=" ", flush=True)
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
            
            print(f"[SUCCESS]")
            total_chunks += len(chunks)
            success_count += 1
            
            # Cleanup
            del text, chunks, embeddings
            gc.collect()
            
        except Exception as e:
            print(f"[FAILED] {e}")
    
    print()
    print(f"Batch complete: {success_count}/{len(pdf_paths)} PDFs")
    print(f"Total chunks: {total_chunks}")
    
    return 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: indexer_worker.py <batch_file>")
        sys.exit(1)
    
    exit_code = main(sys.argv[1])
    sys.exit(exit_code)
