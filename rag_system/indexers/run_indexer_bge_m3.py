"""
Standalone Biostatistics Indexer with BGE-M3
=============================================
No external dependencies beyond requirements.txt
Uses BGE-M3 (PyTorch) for embeddings

@author: Dr. Mahir Kurt + Claude Code
@date: 2025-11-25
"""

import os
import sys

# CRITICAL: Set encoding BEFORE any imports
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Windows locale fix for transformers
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
warnings.filterwarnings("ignore", category=FutureWarning)

print("=" * 80)
print("BIOSTATISTICS RAG - BGE-M3 STANDALONE INDEXER")
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
from dotenv import load_dotenv
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

# Load environment
load_dotenv(override=True)

print(f"📚 Textbooks: {TEXTBOOKS_DIR}")
print(f"🤖 Model: {PRIMARY_MODEL_NAME}")
print(f"📦 Collection: {COLLECTION_NAME}")
print(f"💾 ChromaDB: {CHROMA_DB_PATH}")
print()

# Load model
print("Loading BGE-M3 embedding model (first time: ~5-10 min, 2.3 GB)...")
model = BGEModel(PRIMARY_MODEL_NAME)
print()

# Initialize ChromaDB
print("Initializing ChromaDB...")
CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)

# Use a function to get fresh connection
def init_chromadb():
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    return client

client = init_chromadb()
print(f"  ✓ Using local path: {CHROMA_DB_PATH}")

# Try to delete existing collection
try:
    client.delete_collection(COLLECTION_NAME)
    print(f"  Deleted existing collection: {COLLECTION_NAME}")
except Exception:
    pass

# Create initial collection
client.create_collection(
    name=COLLECTION_NAME, 
    metadata={"hnsw:space": "cosine"}
)
print(f"✅ Collection '{COLLECTION_NAME}' ready")

# Close initial client
del client
print()

# Find PDFs
pdf_files = list(TEXTBOOKS_DIR.glob("*.pdf"))
if not pdf_files:
    print(f"❌ No PDF files found in {TEXTBOOKS_DIR}")
    sys.exit(1)

print(f"Found {len(pdf_files)} PDF files")
print()


def extract_text_pymupdf(pdf_path: Path) -> str:
    """Extract text using PyMuPDF (faster, better quality)"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        return text.strip()
    except Exception as e:
        print(f"  ⚠️  PyMuPDF error: {e}")
        return ""


def extract_text_pypdf2(pdf_path: Path) -> str:
    """Extract text using PyPDF2 (fallback)"""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        print(f"  ⚠️  PyPDF2 error: {e}")
        return ""


def extract_text(pdf_path: Path) -> str:
    """Extract text from PDF using available library"""
    if PDF_LIBRARY == "pymupdf":
        return extract_text_pymupdf(pdf_path)
    else:
        return extract_text_pypdf2(pdf_path)


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """Split text into overlapping chunks with semantic boundaries"""
    if not text:
        return []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + size
        
        # Try to break at sentence boundary
        if end < len(text):
            for delimiter in ['. ', '? ', '! ', '\n\n', '\n']:
                last_delim = text[start:end].rfind(delimiter)
                if last_delim > size // 2:  # Only use if not too early
                    end = start + last_delim + len(delimiter)
                    break
        
        chunk = text[start:end].strip()
        if chunk and len(chunk) > 50:  # Skip very short chunks
            chunks.append(chunk)
        
        start = end - overlap if end < len(text) else end
    
    return chunks


# Process PDFs
total_chunks = 0
batch_size = 25  # Reduced batch size to avoid disk I/O issues
failed_pdfs = []

def get_collection():
    """Get fresh ChromaDB collection (reconnect each time)"""
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

for idx, pdf_path in enumerate(pdf_files, 1):
    print(f"[{idx}/{len(pdf_files)}] {pdf_path.name}")

    try:
        # Get fresh collection connection for each PDF
        collection = get_collection()
        
        # Extract text
        print("  📖 Extracting text...", end=" ")
        text = extract_text(pdf_path)
        
        if not text:
            print("⚠️  No text extracted")
            failed_pdfs.append(pdf_path.name)
            continue
        
        print(f"✓ ({len(text):,} chars)")

        # Chunk text
        chunks = chunk_text(text)
        print(f"  📄 {len(chunks)} chunks")

        if not chunks:
            print("  ⚠️  No chunks created")
            failed_pdfs.append(pdf_path.name)
            continue

        # Process in batches
        for batch_start in range(0, len(chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]

            # Generate embeddings
            batch_num = batch_start // batch_size + 1
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            print(f"    Encoding batch {batch_num}/{total_batches}...", end=" ")
            
            embeddings = model.encode(batch_chunks, normalize=True, show_progress=False)

            # Create IDs and metadata
            ids = [
                f"{pdf_path.stem}_chunk_{total_chunks + i}"
                for i in range(len(batch_chunks))
            ]

            metadatas = [
                {
                    "source": pdf_path.name,
                    "chunk_id": total_chunks + i,
                    "pdf_name": pdf_path.stem,
                }
                for i in range(len(batch_chunks))
            ]

            # Add to collection
            collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=batch_chunks,
                metadatas=metadatas,
            )

            total_chunks += len(batch_chunks)
            print(f"✅ → Total: {total_chunks}")

        print(f"  ✅ Complete\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted! Progress saved.")
        print(f"Total chunks indexed so far: {total_chunks}")
        sys.exit(0)
        
    except Exception as e:
        print(f"  ❌ Error: {e}\n")
        failed_pdfs.append(pdf_path.name)
        continue

# Summary
print("=" * 80)
print("✅ INDEXING COMPLETE")
print("=" * 80)
print(f"Total chunks indexed: {total_chunks}")
print(f"Collection: {COLLECTION_NAME}")
print(f"ChromaDB path: {CHROMA_DB_PATH}")

if failed_pdfs:
    print(f"\n⚠️  Failed PDFs ({len(failed_pdfs)}):")
    for pdf in failed_pdfs:
        print(f"   - {pdf}")

print()
print("🚀 You can now run the API server:")
print("   python api_server.py")
print()
print("Or test with interactive mode:")
print("   cd rag_system && python query_interface.py")
print("=" * 80)
