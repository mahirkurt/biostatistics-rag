"""
Single PDF Indexer - Process one PDF at a time

Usage: python run_single_pdf.py [pdf_name_or_index]

Examples:
    python run_single_pdf.py 1                    # Process 1st PDF
    python run_single_pdf.py "Biostatistics.pdf"  # Process by name
    python run_single_pdf.py next                 # Process next unindexed PDF
    python run_single_pdf.py all                  # Process all with auto-restart
"""

import os
import sys
import subprocess

# === CRITICAL: Set environment BEFORE any imports ===
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# === Mock platform functions BEFORE chromadb import ===
import platform
platform.system = lambda: "Windows"
platform.win32_ver = lambda: ("10", "10.0.19041", "", "Multiprocessor Free")

import chromadb
import PyPDF2
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from rag_system.embeddings.pure_onnx_minilm import PureONNXMiniLMAdapter

# === CONFIGURATION ===
TEXTBOOKS_DIR = Path(__file__).parent.parent.parent / "textbooks"
CHROMA_DB_PATH = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "biostatistics_main"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBED_BATCH_SIZE = 16  # Smaller batches for stability


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


def chunk_text(text: str) -> list:
    """Split text into overlapping chunks."""
    if not text or len(text) < 100:
        return []
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + CHUNK_SIZE, text_len)
        chunk = text[start:end].strip()
        if len(chunk) >= 100:
            chunks.append(chunk)
        start = end - CHUNK_OVERLAP if end < text_len else text_len
        if start >= text_len - 50:
            break
    
    return chunks


def generate_chunk_id(pdf_name: str, chunk_index: int) -> str:
    """Generate unique ID for a chunk."""
    return hashlib.md5(f"{pdf_name}_{chunk_index}".encode()).hexdigest()


def get_collection():
    """Get ChromaDB collection."""
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


def get_indexed_pdfs(collection) -> set:
    """Get set of already indexed PDF names."""
    indexed = set()
    try:
        result = collection.get(include=["metadatas"])
        for meta in result['metadatas']:
            if 'source' in meta:
                indexed.add(meta['source'])
    except:
        pass
    return indexed


def process_single_pdf(pdf_path: Path, collection, embedder):
    """Process a single PDF file."""
    pdf_name = pdf_path.name
    print(f"\n[PROCESSING] {pdf_name}")
    
    # Check if already fully indexed
    existing_ids = set()
    try:
        result = collection.get(include=[])
        existing_ids = set(result['ids'])
    except:
        pass
    
    # Extract text
    print("    Extracting text...", end=" ", flush=True)
    text = extract_text_from_pdf(pdf_path)
    if not text or len(text) < 200:
        print("SKIP (no content)")
        return 0
    print(f"OK ({len(text):,} chars)")
    
    # Chunk text
    print("    Chunking...", end=" ", flush=True)
    chunks = chunk_text(text)
    print(f"OK ({len(chunks)} chunks)")
    
    if not chunks:
        return 0
    
    # Filter already indexed
    chunk_ids = [generate_chunk_id(pdf_name, j) for j in range(len(chunks))]
    new_chunks = []
    new_ids = []
    
    for chunk, chunk_id in zip(chunks, chunk_ids):
        if chunk_id not in existing_ids:
            new_chunks.append(chunk)
            new_ids.append(chunk_id)
    
    if not new_chunks:
        print("    SKIP (already indexed)")
        return 0
    
    print(f"    New chunks: {len(new_chunks)}")
    
    # Embed in small batches
    print("    Embedding...", end=" ", flush=True)
    all_embeddings = []
    
    for i in range(0, len(new_chunks), EMBED_BATCH_SIZE):
        batch = new_chunks[i:i + EMBED_BATCH_SIZE]
        embeddings = embedder.encode(batch)
        all_embeddings.extend(embeddings.tolist())
        print(".", end="", flush=True)
    
    print(" OK")
    
    # Add to ChromaDB
    print("    Adding to ChromaDB...", end=" ", flush=True)
    
    metadatas = [
        {"source": pdf_name, "chunk_index": j, "total_chunks": len(chunks)}
        for j in range(len(new_chunks))
    ]
    
    collection.add(
        ids=new_ids,
        embeddings=all_embeddings,
        documents=new_chunks,
        metadatas=metadatas
    )
    
    print("OK")
    return len(new_chunks)


def main():
    args = sys.argv[1:] if len(sys.argv) > 1 else ["next"]
    mode = args[0].lower()
    
    print("=" * 50)
    print("Single PDF Indexer")
    print("=" * 50)
    
    # Get all PDFs
    pdf_files = sorted(TEXTBOOKS_DIR.glob("*.pdf"))
    print(f"Total PDFs: {len(pdf_files)}")
    
    # Initialize
    collection = get_collection()
    indexed_pdfs = get_indexed_pdfs(collection)
    print(f"Already indexed: {len(indexed_pdfs)} PDFs")
    print(f"Documents in DB: {collection.count()}")
    
    # Load embedder once
    print("\nLoading embedder...")
    embedder = PureONNXMiniLMAdapter()
    
    if mode == "all":
        # Process all with auto-restart
        for i, pdf_path in enumerate(pdf_files):
            if pdf_path.name in indexed_pdfs:
                print(f"\n[{i+1}/{len(pdf_files)}] SKIP {pdf_path.name}")
                continue
            
            print(f"\n[{i+1}/{len(pdf_files)}] Processing...")
            try:
                added = process_single_pdf(pdf_path, collection, embedder)
                if added > 0:
                    print(f"    Added {added} chunks")
            except KeyboardInterrupt:
                print("\n\n[INTERRUPTED] Progress saved. Run again to continue.")
                break
        
        print(f"\n[DONE] Total documents: {collection.count()}")
    
    elif mode == "next":
        # Find next unindexed PDF
        for pdf_path in pdf_files:
            if pdf_path.name not in indexed_pdfs:
                added = process_single_pdf(pdf_path, collection, embedder)
                print(f"\n[DONE] Added {added} chunks. Total: {collection.count()}")
                return
        
        print("\n[DONE] All PDFs already indexed!")
    
    elif mode.isdigit():
        # Process by index
        idx = int(mode) - 1
        if 0 <= idx < len(pdf_files):
            added = process_single_pdf(pdf_files[idx], collection, embedder)
            print(f"\n[DONE] Added {added} chunks. Total: {collection.count()}")
        else:
            print(f"[ERROR] Index {idx+1} out of range (1-{len(pdf_files)})")
    
    else:
        # Process by name
        for pdf_path in pdf_files:
            if mode in pdf_path.name.lower():
                added = process_single_pdf(pdf_path, collection, embedder)
                print(f"\n[DONE] Added {added} chunks. Total: {collection.count()}")
                return
        
        print(f"[ERROR] No PDF matching '{mode}'")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Progress saved.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
