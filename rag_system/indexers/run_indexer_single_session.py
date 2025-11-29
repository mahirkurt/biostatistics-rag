"""
Biostatistics Indexer - Single Session Mode
============================================
Tüm verileri tek bir ChromaDB session'ında yazar.
Disk I/O hatalarını önlemek için tasarlandı.

@author: Dr. Mahir Kurt + Claude Code
@date: 2025-11-26
"""

import os
import sys

# CRITICAL: Set encoding BEFORE any imports
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Windows locale fix
import locale
try:
    locale.setlocale(locale.LC_ALL, "C")
except:
    pass

# Platform mocking for ChromaDB 1.3.5 on Windows
import platform
platform.system = lambda: "Windows"
platform.win32_ver = lambda: ("10", "10.0.19041", "", "Multiprocessor Free")

from pathlib import Path
import warnings
import gc
warnings.filterwarnings("ignore", category=FutureWarning)

print("=" * 80)
print("BIOSTATISTICS RAG - SINGLE SESSION INDEXER")
print("=" * 80)
print()

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent))

print("Importing dependencies...")
try:
    import fitz  # PyMuPDF
    PDF_LIBRARY = "pymupdf"
    print("✓ Using PyMuPDF for PDF extraction")
except ImportError:
    from PyPDF2 import PdfReader
    PDF_LIBRARY = "pypdf2"
    print("✓ Using PyPDF2 for PDF extraction")

import chromadb
from embedding_models import BGEModel

print("✓ Dependencies loaded")

from config.biostat_rag_config import (
    TEXTBOOKS_DIR,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    PRIMARY_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

print(f"📚 Textbooks: {TEXTBOOKS_DIR}")
print(f"🤖 Model: {PRIMARY_MODEL_NAME}")
print(f"📦 Collection: {COLLECTION_NAME}")
print(f"💾 ChromaDB: {CHROMA_DB_PATH}")
print()

# Load model ONCE
print("Loading BGE-M3 embedding model...")
model = BGEModel(PRIMARY_MODEL_NAME)
print()


def extract_text_pymupdf(pdf_path: Path) -> str:
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        return text.strip()
    except Exception as e:
        return ""


def extract_text_pypdf2(pdf_path: Path) -> str:
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except:
        return ""


def extract_text(pdf_path: Path) -> str:
    if PDF_LIBRARY == "pymupdf":
        return extract_text_pymupdf(pdf_path)
    return extract_text_pypdf2(pdf_path)


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        if end < len(text):
            for delim in ['. ', '? ', '! ', '\n\n', '\n']:
                last = text[start:end].rfind(delim)
                if last > size // 2:
                    end = start + last + len(delim)
                    break
        chunk = text[start:end].strip()
        if chunk and len(chunk) > 50:
            chunks.append(chunk)
        start = end - overlap if end < len(text) else end
    return chunks


# Find PDFs
pdf_files = sorted(TEXTBOOKS_DIR.glob("*.pdf"))
if not pdf_files:
    print(f"❌ No PDF files found in {TEXTBOOKS_DIR}")
    sys.exit(1)

print(f"Found {len(pdf_files)} PDF files")
print()

# ============================================================================
# PHASE 1: Extract all text and create embeddings in memory
# ============================================================================
print("=" * 80)
print("PHASE 1: Extracting text and creating embeddings")
print("=" * 80)

all_ids = []
all_embeddings = []
all_documents = []
all_metadatas = []

total_chunks = 0
failed_pdfs = []

for idx, pdf_path in enumerate(pdf_files, 1):
    print(f"\n[{idx}/{len(pdf_files)}] {pdf_path.name}")
    
    try:
        # Extract
        print("  📖 Extracting...", end=" ")
        text = extract_text(pdf_path)
        if not text:
            print("⚠️ No text")
            failed_pdfs.append(pdf_path.name)
            continue
        print(f"✓ ({len(text):,} chars)")
        
        # Chunk
        chunks = chunk_text(text)
        if not chunks:
            print("  ⚠️ No chunks")
            failed_pdfs.append(pdf_path.name)
            continue
        print(f"  📄 {len(chunks)} chunks")
        
        # Embed in batches
        batch_size = 50
        pdf_embeddings = []
        
        for batch_start in range(0, len(chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(chunks))
            batch = chunks[batch_start:batch_end]
            
            emb = model.encode(batch, normalize=True, show_progress=False)
            pdf_embeddings.extend(emb.tolist())
            
            batch_num = batch_start // batch_size + 1
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            print(f"    Batch {batch_num}/{total_batches} ✓")
        
        # Store in memory
        for i, (chunk, emb) in enumerate(zip(chunks, pdf_embeddings)):
            all_ids.append(f"{pdf_path.stem}_chunk_{total_chunks + i}")
            all_embeddings.append(emb)
            all_documents.append(chunk)
            all_metadatas.append({
                "source": pdf_path.name,
                "chunk_id": total_chunks + i,
                "pdf_name": pdf_path.stem,
            })
        
        total_chunks += len(chunks)
        print(f"  ✅ Total so far: {total_chunks}")
        
        # Force garbage collection
        del text, chunks, pdf_embeddings
        gc.collect()
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted!")
        break
    except Exception as e:
        print(f"  ❌ Error: {e}")
        failed_pdfs.append(pdf_path.name)

print()
print("=" * 80)
print(f"PHASE 1 COMPLETE: {total_chunks} chunks in memory")
print("=" * 80)

if total_chunks == 0:
    print("❌ No chunks to index!")
    sys.exit(1)

# ============================================================================
# PHASE 2: Write all to ChromaDB in one session
# ============================================================================
print()
print("=" * 80)
print("PHASE 2: Writing to ChromaDB (single session)")
print("=" * 80)

# Clean and create fresh
print("\nInitializing ChromaDB...")
CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)

# Remove old DB completely
import shutil
if CHROMA_DB_PATH.exists():
    shutil.rmtree(CHROMA_DB_PATH)
    print("  Removed old database")
CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)

# Create client and collection
client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)
print(f"  Created collection: {COLLECTION_NAME}")

# Write in large batches
write_batch_size = 500
total_written = 0

print(f"\nWriting {total_chunks} chunks...")

for start in range(0, len(all_ids), write_batch_size):
    end = min(start + write_batch_size, len(all_ids))
    
    collection.add(
        ids=all_ids[start:end],
        embeddings=all_embeddings[start:end],
        documents=all_documents[start:end],
        metadatas=all_metadatas[start:end],
    )
    
    total_written = end
    print(f"  Written: {total_written}/{total_chunks}")

# Verify
final_count = collection.count()
print(f"\n✅ Verified: {final_count} documents in collection")

# Close client
del collection
del client
gc.collect()

# ============================================================================
# Summary
# ============================================================================
print()
print("=" * 80)
print("✅ INDEXING COMPLETE")
print("=" * 80)
print(f"Total chunks indexed: {final_count}")
print(f"Collection: {COLLECTION_NAME}")
print(f"ChromaDB path: {CHROMA_DB_PATH}")

if failed_pdfs:
    print(f"\n⚠️ Failed PDFs ({len(failed_pdfs)}):")
    for pdf in failed_pdfs[:10]:
        print(f"   - {pdf}")
    if len(failed_pdfs) > 10:
        print(f"   ... and {len(failed_pdfs) - 10} more")

print()
print("🚀 Run API server:")
print("   python api_server.py")
print("=" * 80)
